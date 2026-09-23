"""Config-driven full SUN RGB-D experiment, DDP, resumable checkpoints and fixed-10 AP."""
import argparse
import copy
from contextlib import nullcontext
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from detection.benchmark import load_config,build_model,postprocess
from detection.data import SUNRGBDFull,prepare_batch,collate_samples
from detection.loss import DetectionCriterion
from detection.evaluate import evaluate


def validate_resume_compatibility(checkpoint,config,world_size,steps_per_epoch,allow_migration=False):
    """Validate an exact resume or a controlled effective-batch-preserving migration."""
    if not checkpoint.get('epoch_complete',True):
        raise ValueError('Smoke/partial epoch checkpoint: use --initialize, not resume')
    if checkpoint['world_size']==world_size and checkpoint['config']==config:
        return None
    if not allow_migration:
        if checkpoint['world_size']!=world_size:
            raise ValueError('Exact resume requires identical world size; use --allow-resume-migration for a controlled change')
        raise ValueError('Exact resume requires identical config; use --initialize for a new experiment')

    previous=copy.deepcopy(checkpoint['config']); current=copy.deepcopy(config)
    previous_batch=previous['training']['batch_size_per_gpu']
    current_batch=current['training']['batch_size_per_gpu']
    previous['training']['batch_size_per_gpu']=None
    current['training']['batch_size_per_gpu']=None
    if previous!=current:
        raise ValueError('Resume migration only permits batch_size_per_gpu and world size to change')
    previous_effective=previous_batch*checkpoint['world_size']
    current_effective=current_batch*world_size
    if previous_effective!=current_effective:
        raise ValueError(f'Resume migration must preserve effective batch size: {previous_effective} != {current_effective}')
    completed_epochs=checkpoint['epoch']+1
    expected_global_step=completed_epochs*steps_per_epoch
    if checkpoint['global_step']!=expected_global_step:
        raise ValueError(f'Resume migration must preserve steps per epoch: expected global_step {expected_global_step}, got {checkpoint["global_step"]}')
    if world_size>len(checkpoint['rng_by_rank']):
        raise ValueError('Resume migration cannot create ranks without saved RNG state')
    return {'from_world_size':checkpoint['world_size'],'to_world_size':world_size,
            'from_batch_size_per_gpu':previous_batch,'to_batch_size_per_gpu':current_batch,
            'effective_batch_size':current_effective,'steps_per_epoch':steps_per_epoch,
            'bit_exact':False}


def describe_nonfinite_gradients(module):
    """Return compact, JSON-safe diagnostics for parameters with NaN/Inf gradients."""
    details=[]
    for name,parameter in module.named_parameters():
        if parameter.grad is None: continue
        gradient=parameter.grad.detach()
        values=gradient.coalesce().values() if gradient.is_sparse else gradient
        finite=torch.isfinite(values)
        if bool(finite.all()): continue
        details.append({'parameter':name,'shape':list(parameter.shape),
                        'nonfinite':int((~finite).sum()),
                        'nan':int(torch.isnan(values).sum()),
                        'posinf':int(torch.isposinf(values).sum()),
                        'neginf':int(torch.isneginf(values).sum())})
    return details


def split_microbatches(samples,size=None):
    """Keep the DataLoader's logical batch while bounding activation memory."""
    if size is None: return [samples]
    if size<1: raise ValueError('microbatch size must be positive')
    return [samples[start:start+size] for start in range(0,len(samples),size)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/utonia_sunrgbd_full.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ablation',choices=['A','B','C'])
    parser.add_argument('--epochs',type=int,help='Stop after this epoch; retain configured full cosine schedule')
    parser.add_argument('--max-steps',type=int,help='Smoke only: cap optimizer steps per epoch')
    parser.add_argument('--max-val-batches',type=int,help='Smoke only; metrics are explicitly marked subset')
    parser.add_argument('--profile-only',action='store_true',help='Run capped training steps, report peak memory, and exit without validation/checkpoints')
    parser.add_argument('--resume',type=Path,help='Exact same-world-size checkpoint resume, including optimizer/scheduler/RNG')
    parser.add_argument('--allow-resume-migration',action='store_true',
                        help='Allow a complete-epoch world-size/batch migration when effective batch and steps/epoch stay unchanged')
    parser.add_argument('--batch-size-per-gpu',type=int,help='Override the configured per-GPU batch size')
    parser.add_argument('--microbatch-size',type=int,
                        help='Split each logical per-GPU batch for gradient accumulation without changing optimizer steps')
    parser.add_argument('--max-nonfinite-gradient-skips',type=int,default=0,
                        help='Audited NaN/Inf-gradient updates to skip before failing (default: strict failure)')
    parser.add_argument('--initialize',type=Path,help='Model weights only, e.g. start B from A; reset optimizer')
    parser.add_argument('--evaluate-only',action='store_true')
    args=parser.parse_args()
    if args.resume and args.initialize: parser.error('Choose resume OR initialize')
    if args.profile_only and not args.max_steps: parser.error('--profile-only requires --max-steps')
    if args.profile_only and args.evaluate_only: parser.error('--profile-only cannot be combined with --evaluate-only')
    if args.max_nonfinite_gradient_skips<0: parser.error('--max-nonfinite-gradient-skips must be nonnegative')
    if args.microbatch_size is not None and args.microbatch_size<1: parser.error('--microbatch-size must be positive')
    cfg=load_config(args.config)
    if args.ablation: cfg['training']['ablation']=args.ablation
    if args.batch_size_per_gpu is not None:
        if args.batch_size_per_gpu<1: parser.error('--batch-size-per-gpu must be positive')
        cfg['training']['batch_size_per_gpu']=args.batch_size_per_gpu
    if args.microbatch_size is not None and args.microbatch_size>cfg['training']['batch_size_per_gpu']:
        parser.error('--microbatch-size cannot exceed --batch-size-per-gpu')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.set_num_threads(4); torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=60),device_id=device)
    seed=cfg['seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    args.output.mkdir(parents=True,exist_ok=True)
    dc,tc=cfg['dataset'],cfg['training']
    train=SUNRGBDFull(dc['processed_dir'],dc['train_split'],dc['num_points'],cfg['augmentation'],seed)
    val=SUNRGBDFull(dc['processed_dir'],dc['val_split'],dc['num_points'],None,seed)
    assert len(train)==dc['expected_train_scenes'] and len(val)==dc['expected_val_scenes']
    assert train.manifest['annotation_version']==dc['annotation_version']
    assert not set(train.ids)&set(val.ids)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=seed,drop_last=False)
    loader=DataLoader(train,batch_size=tc['batch_size_per_gpu'],sampler=sampler,num_workers=dc['workers_per_gpu'],
                      collate_fn=collate_samples,pin_memory=False)
    # No padded duplicated scenes in distributed validation.
    val_loader=DataLoader(Subset(val,list(range(rank,len(val),world))),batch_size=tc['batch_size_per_gpu'],
                         num_workers=dc['workers_per_gpu'],collate_fn=collate_samples)
    base=build_model(cfg).to(device)
    optimizer=torch.optim.AdamW(base.optimizer_groups(tc['head_lr'],tc['backbone_lr']),weight_decay=tc['weight_decay'])
    total_steps=tc['epochs']*len(loader)
    def schedule(step):
        if step<tc['warmup_steps']: return max(step+1,1)/max(tc['warmup_steps'],1)
        phase=min(1,(step-tc['warmup_steps'])/max(total_steps-tc['warmup_steps'],1))
        return tc['min_lr_ratio']+(1-tc['min_lr_ratio'])*(1+math.cos(math.pi*phase))/2
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    criterion=DetectionCriterion(**cfg['loss'])
    amp=tc['amp']; dtype={'bfloat16':torch.bfloat16,'float16':torch.float16}[tc['amp_dtype']]
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    start_epoch=0; global_step=0; best25=-1.; best50=-1.; resume_migration=None; nonfinite_gradient_events=[]
    if args.resume or args.initialize:
        ck=torch.load(args.resume or args.initialize,map_location='cpu',weights_only=True)
        base.load_state_dict(ck['model'])
        if args.resume:
            resume_migration=validate_resume_compatibility(ck,cfg,world,len(loader),args.allow_resume_migration)
            saved_microbatch=ck.get('microbatch_size')
            if saved_microbatch is not None and saved_microbatch!=(args.microbatch_size or tc['batch_size_per_gpu']):
                raise ValueError(f'Exact resume requires microbatch size {saved_microbatch}')
            optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
            start_epoch=ck['epoch']+1; global_step=ck['global_step']; best25=ck['best_AP25']; best50=ck['best_AP50']
            nonfinite_gradient_events=ck.get('nonfinite_gradient_events',[])
            rng=ck['rng_by_rank'][rank]; torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'],local); random.setstate(rng['python'])
    elif args.evaluate_only: parser.error('Evaluation requires --resume or --initialize')
    model=DDP(base,device_ids=[local],find_unused_parameters=not base.freeze_backbone) if world>1 else base
    writer=None
    if rank==0:
        metadata={'config':cfg,'run_epochs':args.epochs,'world_size':world,'train_scenes':len(train),'val_scenes':len(val),
            'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':sys.executable,'gpu':torch.cuda.get_device_name(),
            'max_steps':args.max_steps,'max_val_batches':args.max_val_batches,'short_run':args.epochs is not None,
            'resume_migration':resume_migration,'profile_only':args.profile_only,
            'max_nonfinite_gradient_skips':args.max_nonfinite_gradient_skips,
            'microbatch_size':args.microbatch_size or tc['batch_size_per_gpu']}
        (args.output/'run_config.json').write_text(json.dumps(metadata,indent=2))
        print('CONFIG',json.dumps(metadata),flush=True)
        if cfg['logging']['tensorboard']:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer=SummaryWriter(str(args.output/'tensorboard'))
            except ImportError: print('TensorBoard unavailable; CSV/JSONL are enabled.',flush=True)
    step_log=(args.output/f'loss_rank{rank}.jsonl').open('a' if args.resume else 'w')

    def evaluate_epoch(epoch):
        base.eval(); predictions={}; targets={}; sums={}; count=0
        with torch.no_grad():
            for batch,samples in enumerate(val_loader):
                if args.max_val_batches and batch>=args.max_val_batches: break
                points,bounds,target=prepare_batch(samples,device,dc['backbone_scale'],dc['voxel_size'])
                with torch.autocast('cuda',dtype=dtype,enabled=amp): output=base(points,bounds)
                _,parts=criterion(output,target)
                for k,v in parts.items(): sums[k]=sums.get(k,0)+float(v)*len(samples)
                count+=len(samples)
                predictions.update(postprocess(output,samples,cfg['evaluation']))
                for s in samples: targets[s['id']]={'boxes':s['boxes'],'labels':s['labels']}
                if rank==0 and (batch+1)%100==0: print('VAL',epoch,batch+1,'/',len(val_loader),flush=True)
        local_result={'predictions':predictions,'targets':targets,'sums':sums,'count':count}
        # Atomic files avoid gathering hundreds of MB of Python pickles through NCCL.
        part=args.output/f'val_rank{rank}.pt'; temporary=part.with_suffix('.part'); torch.save(local_result,temporary); temporary.replace(part)
        if world>1: dist.barrier()
        metrics=None
        if rank==0:
            merged_pred={}; merged_gt={}; totals={}; n=0
            for r in range(world):
                # Locally generated evaluation arrays, never third-party pickle data.
                shard=torch.load(args.output/f'val_rank{r}.pt',map_location='cpu',weights_only=False)
                assert not set(merged_pred)&set(shard['predictions'])
                merged_pred.update(shard['predictions']); merged_gt.update(shard['targets']); n+=shard['count']
                for k,v in shard['sums'].items(): totals[k]=totals.get(k,0)+v
            if not args.max_val_batches: assert n==len(val) and set(merged_gt)==set(val.ids)
            metrics=evaluate(merged_pred,merged_gt,thresholds=cfg['evaluation']['thresholds'],fixed_classes=True)
            nonempty={sid:t for sid,t in merged_gt.items() if len(t['boxes'])}
            metrics['nonempty_scene_comparison']=evaluate({sid:merged_pred[sid] for sid in nonempty},nonempty,
                thresholds=cfg['evaluation']['thresholds'],fixed_classes=True)
            metrics['nonempty_scene_comparison']['scenes']=len(nonempty)
            metrics['postprocessing']=cfg['evaluation']; metrics['validation_scenes']=n
            metrics['is_full_validation']=n==len(val); metrics['losses']={k:v/n for k,v in totals.items()}
            metrics['annotation_version']=dc['annotation_version']; metrics['empty_scene_policy']=dc['empty_scene_policy']
            (args.output/f'epoch_{epoch:03d}_metrics.json').write_text(json.dumps(metrics,indent=2))
            print('EVALUATION',epoch,'scenes',n,'AP25',metrics['thresholds']['0.25']['mAP'],'AP50',metrics['thresholds']['0.5']['mAP'],flush=True)
        if world>1:
            shared=[metrics]; dist.broadcast_object_list(shared,src=0); metrics=shared[0]
        return metrics

    if args.evaluate_only:
        evaluate_epoch(start_epoch)
        if world>1: dist.destroy_process_group()
        return
    run_epochs=args.epochs or tc['epochs']
    for epoch in range(start_epoch,run_epochs):
        train.set_epoch(epoch); sampler.set_epoch(epoch); model.train()
        sums={}; sample_count=0; started=time.time(); torch.cuda.reset_peak_memory_stats()
        for batch,samples in enumerate(loader):
            if args.max_steps and batch>=args.max_steps: break
            optimizer.zero_grad(set_to_none=True)
            values={}
            microbatches=split_microbatches(samples,args.microbatch_size)
            for micro_index,micro_samples in enumerate(microbatches):
                weight=len(micro_samples)/len(samples)
                points,bounds,target=prepare_batch(micro_samples,device,dc['backbone_scale'],dc['voxel_size'])
                sync=model.no_sync() if world>1 and micro_index+1<len(microbatches) else nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=dtype,enabled=amp):
                        output=model(points,bounds); loss,parts=criterion(output,target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at {epoch}/{batch} microbatch {micro_index}')
                    scaler.scale(loss*weight).backward()
                for k,v in parts.items(): values[k]=values.get(k,0.)+float(v.detach())*weight
                del points,bounds,target,output,loss,parts
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in base.parameters() if p.requires_grad],tc['grad_clip'],error_if_nonfinite=False)
            sample_count+=len(samples)
            for k,v in values.items(): sums[k]=sums.get(k,0)+v*len(samples)
            skipped_update=not bool(torch.isfinite(norm))
            gradient_details=[]
            if skipped_update:
                gradient_details=describe_nonfinite_gradients(base)
                event={'epoch':epoch,'batch':batch,'attempted_step':global_step+1,
                       'ids':[s['id'] for s in samples],'losses':values,
                       'parameters':gradient_details,'action':'skip_optimizer_update'}
                nonfinite_gradient_events.append(event)
                if rank==0:
                    with (args.output/'nonfinite_gradients.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
                    print('SKIP_NONFINITE_GRADIENT',json.dumps(event),flush=True)
                optimizer.zero_grad(set_to_none=True)
                if len(nonfinite_gradient_events)>args.max_nonfinite_gradient_skips:
                    raise FloatingPointError(
                        f'Nonfinite gradient skip limit exceeded at {epoch}/{batch}: '
                        f'{len(nonfinite_gradient_events)} > {args.max_nonfinite_gradient_skips}')
                scaler.update()
            else:
                scaler.step(optimizer); scaler.update()
            scheduler.step(); global_step+=1
            row={'epoch':epoch,'step':global_step,'ids':[s['id'] for s in samples],**values,
                 'grad_norm':None if skipped_update else float(norm),'head_lr':optimizer.param_groups[0]['lr'],
                 'optimizer_update_skipped':skipped_update,
                 'microbatch_size':args.microbatch_size or len(samples)}
            if gradient_details: row['nonfinite_gradients']=gradient_details
            step_log.write(json.dumps(row)+'\n'); step_log.flush()
            if rank==0 and (batch%cfg['logging']['every_steps']==0):
                print('TRAIN',epoch,batch,'/',len(loader),json.dumps(values),'sec',round(time.time()-started,1),flush=True)
                if writer:
                    for k,v in values.items(): writer.add_scalar('step/'+k,v,global_step)
        keys=sorted(sums); accum=torch.tensor([sample_count]+[sums[k] for k in keys],dtype=torch.float64,device=device)
        if world>1: dist.all_reduce(accum)
        means={k:float(accum[i+1]/accum[0]) for i,k in enumerate(keys)}
        peak=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device)
        if world>1: dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        if args.profile_only:
            if rank==0:
                print('PROFILE_COMPLETE',json.dumps({'ablation':tc['ablation'],'batch_size_per_gpu':tc['batch_size_per_gpu'],
                    'steps':min(args.max_steps,len(loader)),'peak_allocated_GiB':float(peak),
                    'seconds':time.time()-started,**{'train_'+k:v for k,v in means.items()}}),flush=True)
            step_log.close()
            if writer: writer.close()
            if world>1: dist.destroy_process_group()
            return
        metrics=evaluate_epoch(epoch)
        ap25=metrics['thresholds']['0.25']['mAP']; ap50=metrics['thresholds']['0.5']['mAP']
        improved25=ap25>best25; improved50=ap50>best50; best25=max(best25,ap25); best50=max(best50,ap50)
        rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(local),'python':random.getstate()}
        rngs=[None]*world
        if world>1: dist.all_gather_object(rngs,rng)
        else: rngs=[rng]
        if rank==0:
            record={'epoch':epoch,'train_samples_with_ddp_padding':int(accum[0]),**{'train_'+k:v for k,v in means.items()},
                **{'val_'+k:v for k,v in metrics['losses'].items()},'AP25':ap25,'AP50':ap50,
                'head_lr':optimizer.param_groups[0]['lr'],'backbone_lr':optimizer.param_groups[1]['lr'] if len(optimizer.param_groups)>1 else 0,
                'peak_allocated_GiB':float(peak),'seconds':time.time()-started,'validation_scenes':metrics['validation_scenes']}
            for t,stats in metrics['thresholds'].items():
                for cls,v in stats['per_class'].items(): record[f'AP{int(float(t)*100)}_{cls}']=v['AP'] or 0
            path=args.output/'epochs.csv'; exists=path.exists()
            with path.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=record.keys())
                if not exists: w.writeheader()
                w.writerow(record)
            if writer:
                for k,v in record.items(): writer.add_scalar('epoch/'+k,v,epoch)
                writer.flush()
            ck={'model':base.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'epoch':epoch,'epoch_complete':not args.max_steps or args.max_steps>=len(loader),
                'global_step':global_step,'config':cfg,'world_size':world,'rng_by_rank':rngs,'best_AP25':best25,'best_AP50':best50,
                'nonfinite_gradient_events':nonfinite_gradient_events,
                'microbatch_size':args.microbatch_size or tc['batch_size_per_gpu']}
            def save(name):
                path=args.output/name; tmp=path.with_suffix('.part'); torch.save(ck,tmp); tmp.replace(path)
            save('last.pt')
            if improved25: save('best_AP25.pt')
            if improved50: save('best_AP50.pt')
            print('EPOCH_COMPLETE',json.dumps(record),flush=True)
        if world>1: dist.barrier()
    if writer: writer.close()
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
