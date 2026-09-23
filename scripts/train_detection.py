"""Run real online smoke training, frozen-feature small-set overfit, and evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.data import SUNRGBDSubset,prepare_batch
from detection.model import UtoniaDetector
from detection.loss import DetectionCriterion
from detection.evaluate import evaluate
from detection.geometry import visualize


def digest(module):
    h=hashlib.sha256()
    for p in module.parameters(): h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cpu_record(record):
    return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in record.items()}


def device_record(record,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in record.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/utonia.pth')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/detection/training')
    p.add_argument('--phase',choices=['all','smoke','overfit','evaluate','finetune'],default='all')
    p.add_argument('--iterations',type=int,default=1500)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--queries',type=int,default=64)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',action='store_true',help='Enable BF16 autocast; full precision by default')
    p.add_argument('--resume',type=Path)
    p.add_argument('--seed',type=int,default=37)
    a=p.parse_args()
    if a.iterations<1 or a.batch_size<1: p.error('iterations and batch size must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.cuda.set_device(a.device)
    train_ds=SUNRGBDSubset(a.data_root,'train'); val_ds=SUNRGBDSubset(a.data_root,'val')
    train=[train_ds[i] for i in range(len(train_ds))]; val=[val_ds[i] for i in range(len(val_ds))]
    model=UtoniaDetector(a.checkpoint,freeze_backbone=a.phase!='finetune',queries=a.queries,max_tokens=a.max_tokens).to(a.device)
    optimizer=torch.optim.AdamW(model.optimizer_groups(a.head_lr,a.backbone_lr),weight_decay=1e-4)
    criterion=DetectionCriterion()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        python=sys.executable,backbone_frozen=model.freeze_backbone,
        train_ids=train_ds.ids,val_ids=val_ds.ids,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        box_format='cx cy cz dx dy dz yaw (full size, metres, CCW Z-up)',
        architecture='Utonia Stage3[432]+upsampled Stage4[576] ->192 ->3 decoder layers, 6 heads')
    (a.output/'config.json').write_text(json.dumps(config,indent=2))
    if a.resume:
        ck=torch.load(a.resume,map_location=a.device,weights_only=True)
        model.adapter.load_state_dict(ck['adapter']); model.head.load_state_dict(ck['head'])
        if 'backbone' in ck: model.backbone.load_state_dict(ck['backbone'])
    elif a.phase=='evaluate': p.error('evaluate requires --resume')
    def save(name,step):
        ck={'adapter':model.adapter.state_dict(),'head':model.head.state_dict(),'optimizer':optimizer.state_dict(),
            'step':step,'config':config,'backbone_sha256':backbone_sha}
        if not model.freeze_backbone: ck['backbone']=model.backbone.state_dict()
        torch.save(ck,a.output/name)
    backbone_sha=digest(model.backbone)
    log=(a.output/f'{a.phase}_loss.jsonl').open('w')
    def train_step(samples,records=None):
        model.train(); optimizer.zero_grad(set_to_none=True)
        points,bounds,targets=prepare_batch(samples,a.device)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.amp):
            out=model(points,bounds) if records is None else model.decode(records)
            loss,parts=criterion(out,targets)
        assert all(torch.isfinite(v).all() for k,v in out.items() if torch.is_tensor(v)), 'Nonfinite prediction'
        assert torch.isfinite(loss), 'Nonfinite loss'
        loss.backward()
        grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),'Bad gradients'
        for module in (model.adapter,model.head):
            assert any(v.grad is not None and v.grad.abs().max()>0 for v in module.parameters()),'Missing module gradient'
        grad_norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],10)
        optimizer.step()
        return {k:float(v.detach()) for k,v in parts.items()}|{'grad_norm':float(grad_norm)},out

    if a.phase in ('all','smoke'):
        # Separate forward test before any parameter updates.
        model.eval()
        points,bounds,targets=prepare_batch(train[:1],a.device)
        with torch.no_grad(): forward=model(points,bounds); initial_loss,_=criterion(forward,targets)
        shapes={k:list(v.shape) for k,v in forward.items() if torch.is_tensor(v)}
        print('FORWARD TEST',shapes,'loss',float(initial_loss),flush=True)
        assert shapes['boxes']==[1,a.queries,7]
        before=model.adapter.projection[1].weight.detach().clone()
        smoke=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(10):
            samples=[train[(i*a.batch_size+j)%len(train)] for j in range(a.batch_size)]
            row,_=train_step(samples)
            row.update(phase='online_smoke',step=i+1,ids=[s['id'] for s in samples])
            smoke.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            print('SMOKE',json.dumps(row),flush=True)
        delta=float((model.adapter.projection[1].weight-before).abs().max())
        assert delta>0 and digest(model.backbone)==backbone_sha
        assert all(p.grad is None for p in model.backbone.parameters())
        audit={'forward_shapes':shapes,'initial_loss':float(initial_loss),'iterations':smoke,
            'backward_and_optimizer_step_passed':True,'adapter_max_parameter_change':delta,
            'backbone_sha256_before_after':backbone_sha,'backbone_unchanged':True,
            'all_loss_predictions_gradients_finite':True,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'gpu':torch.cuda.get_device_name()}
        (a.output/'smoke_audit.json').write_text(json.dumps(audit,indent=2))
        save('smoke_head.pt',10)
        if a.phase=='smoke': return

    # Cache only when frozen, and only after the real online smoke iterations.
    records={}
    if model.freeze_backbone:
        model.eval()
        for sample in train+val:
            points,bounds,_=prepare_batch([sample],a.device)
            records[sample['id']]=model.encode(points,bounds)[0]
            r=records[sample['id']]
            print('ENCODE',sample['id'],'stage3',r['stage3_points'],'stage4',r['stage4_points'],
                  'memory',list(r['features'].shape),flush=True)
        torch.save({'backbone_sha256':backbone_sha,'config':config,
            'records':{k:cpu_record(v) for k,v in records.items()}},a.output/'frozen_features.pt')

    def run_eval(tag):
        model.eval(); metrics={}
        for split,samples in [('train_overfit',train),('val_unseen',val)]:
            predictions={}; targets={}
            for s in samples:
                with torch.no_grad():
                    if model.freeze_backbone: out=model.decode([records[s['id']]])
                    else:
                        points,bounds,_=prepare_batch([s],a.device); out=model(points,bounds)
                probs=out['logits'][0].softmax(-1)
                scores,labels=probs[:,:10].max(-1)
                boxes=out['boxes'][0].cpu().numpy()
                pred={'boxes':boxes,'labels':labels.cpu().numpy(),'scores':scores.cpu().numpy()}
                predictions[s['id']]=pred; targets[s['id']]={'boxes':s['boxes'],'labels':s['labels']}
                if tag=='final':
                    dest=a.output/'predictions'/split; dest.mkdir(parents=True,exist_ok=True)
                    np.savez(dest/f"{s['id']:06d}_all_queries.npz",**pred)
                    keep=pred['scores']>=0.3
                    np.save(dest/f"{s['id']:06d}_boxes.npy",boxes[keep])
                    rows=[{'box':b.tolist(),'label':int(l),'score':float(sc)} for b,l,sc in zip(boxes[keep],pred['labels'][keep],pred['scores'][keep])]
                    (dest/f"{s['id']:06d}_boxes.json").write_text(json.dumps(rows,indent=2))
                    visualize(dest/f"{s['id']:06d}_boxes.html",s['coord'],s['color'],boxes[keep],pred['labels'][keep],pred['scores'][keep],
                        title=f"{split} {s['id']:06d}: learned Utonia detector; score >=0.3")
            metrics[split]=evaluate(predictions,targets)
        metrics['scope']='8 training / 4 held-out official split scenes only; not full SUN RGB-D benchmark; V2 annotations'
        (a.output/f'{tag}_metrics.json').write_text(json.dumps(metrics,indent=2))
        print('EVAL',tag,json.dumps({k:{t:v['mAP_present_classes'] for t,v in m['thresholds'].items()} for k,m in metrics.items() if isinstance(m,dict)}),flush=True)
        return metrics

    if a.phase!='evaluate':
        # Baseline evaluated once; validation never participates in optimizer updates.
        run_eval('initial')
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.iterations,eta_min=1e-5)
        started=time.time()
        rng=np.random.default_rng(a.seed)
        for step in range(a.iterations):
            ix=rng.choice(len(train),min(a.batch_size,len(train)),replace=False)
            samples=[train[i] for i in ix]
            cached=[records[s['id']] for s in samples] if model.freeze_backbone else None
            row,_=train_step(samples,cached); scheduler.step()
            row.update(phase='frozen_overfit' if model.freeze_backbone else 'finetune',step=step+1,
                       ids=[s['id'] for s in samples],lr=optimizer.param_groups[0]['lr'])
            log.write(json.dumps(row)+'\n'); log.flush()
            if step==0 or (step+1)%50==0:
                print('TRAIN',json.dumps(row),'elapsed',round(time.time()-started,1),flush=True)
            if (step+1)%250==0: save('latest_head.pt',step+1)
        if model.freeze_backbone: assert digest(model.backbone)==backbone_sha
        save('detector_head.pt',a.iterations)
    run_eval('final')
    print('DONE',a.output,flush=True)


if __name__=='__main__': main()
