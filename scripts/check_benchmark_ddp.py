"""Bounded two-GPU architecture/AMP/DDP correctness check on existing real scenes."""
import argparse,json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from detection.benchmark import load_config,build_model
from detection.data import SUNRGBDSubset,prepare_batch
from detection.loss import DetectionCriterion

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ablation',choices=['A','B','C'],default='A'); a=p.parse_args()
    local=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local); torch.set_num_threads(4); torch.manual_seed(37)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    cfg=load_config(ROOT/'configs/utonia_sunrgbd_full.yaml'); base=build_model(cfg,a.ablation).cuda(local)
    ddp=DistributedDataParallel(base,device_ids=[local],find_unused_parameters=a.ablation!='A')
    dataset=SUNRGBDSubset(Path.home()/'dabaset/SUNRGBD','train')
    samples=[dataset[(local*4+i)%len(dataset)] for i in range(4)]
    points,bounds,targets=prepare_batch(samples,f'cuda:{local}')
    optimizer=torch.optim.AdamW(base.optimizer_groups(1e-4,1e-5))
    before=base.adapter.projections[0][1].weight.detach().clone()
    ddp.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=ddp(points,bounds); loss,parts=DetectionCriterion(**cfg['loss'])(output,targets)
    loss.backward()
    for projection in base.adapter.projections:
        assert projection[1].weight.grad is not None and torch.isfinite(projection[1].weight.grad).all()
    torch.nn.utils.clip_grad_norm_(base.parameters(),1,error_if_nonfinite=True); optimizer.step()
    assert (base.adapter.projections[0][1].weight-before).abs().max()>0
    value=base.adapter.projections[0][1].weight.detach().clone(); other=value.clone()
    dist.broadcast(other,0); assert torch.equal(value,other)
    result={'rank':local,'ablation':a.ablation,'losses':{k:float(v.detach()) for k,v in parts.items()},
        'boxes_shape':list(output['boxes'].shape),'amp':'head BF16; sparse backbone FP32','batch_size_per_gpu':4,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'DDP_parameters_identical':True,'passed':True}
    out=ROOT/'outputs/benchmark/audit'; out.mkdir(parents=True,exist_ok=True)
    (out/f'ddp_{a.ablation}_rank{local}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    dist.destroy_process_group()
if __name__=='__main__': main()
