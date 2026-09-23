"""One isolated full-backbone backward/step check; does not alter overfit weights."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.data import SUNRGBDSubset,prepare_batch
from detection.model import UtoniaDetector
from detection.loss import DetectionCriterion


def main():
    torch.set_num_threads(8); torch.manual_seed(37); torch.cuda.set_device(1)
    sample=SUNRGBDSubset(Path.home()/'dabaset/SUNRGBD','train')[1]
    model=UtoniaDetector(ROOT/'checkpoints/utonia.pth',freeze_backbone=False).to('cuda:1').train()
    ck=torch.load(ROOT/'outputs/detection/training/smoke_head.pt',map_location='cpu',weights_only=True)
    model.adapter.load_state_dict(ck['adapter']); model.head.load_state_dict(ck['head'])
    opt=torch.optim.AdamW(model.optimizer_groups(1e-4,1e-5))
    points,bounds,target=prepare_batch([sample],'cuda:1')
    output=model(points,bounds)
    loss,_=DetectionCriterion()(output,target)
    assert torch.isfinite(loss)
    loss.backward()
    grads=[(name,p) for name,p in model.backbone.named_parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(p.grad).all() for _,p in grads)
    name,param=next((n,p) for n,p in grads if p.grad.abs().max()>0)
    before=param.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(),10)
    opt.step()
    delta=float((param-before).abs().max())
    assert delta>0
    report={'sample':sample['id'],'loss':float(loss.detach()),'backbone_parameters_with_gradient':len(grads),
        'checked_parameter':name,'parameter_max_change':delta,'backbone_lr':1e-5,'head_lr':1e-4,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'gpu':torch.cuda.get_device_name(),
        'passed':True,'note':'Isolated single-step check; trained overfit checkpoint is untouched.'}
    (ROOT/'outputs/detection/finetune_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
