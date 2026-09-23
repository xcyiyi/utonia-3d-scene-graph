"""Predict oriented boxes using Utonia + the trained adapter/head; no GT input needed."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.infer import load_input
from detection.data import prepare_batch
from detection.model import UtoniaDetector
from detection.geometry import visualize,CLASSES
from detection.benchmark import build_model,postprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--head',type=Path,default=ROOT/'outputs/detection/training/detector_head.pt')
    p.add_argument('--backbone',type=Path,default=ROOT/'checkpoints/utonia.pth')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/detection/inference')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--score-threshold',type=float,default=0.3)
    a=p.parse_args()
    if not 0<=a.score_threshold<=1: p.error('score threshold must be in [0,1]')
    torch.set_num_threads(8); torch.manual_seed(37); np.random.seed(37); torch.cuda.set_device(a.device)
    ck=torch.load(a.head,map_location='cpu',weights_only=True)
    config=ck['config']
    benchmark='model' in ck
    if benchmark:
        config['model']['checkpoint']=str(a.backbone)
        model=build_model(config).to(a.device).eval(); model.load_state_dict(ck['model'])
    else:
        model=UtoniaDetector(a.backbone,queries=config['queries'],max_tokens=config['max_tokens']).to(a.device).eval()
        model.adapter.load_state_dict(ck['adapter']); model.head.load_state_dict(ck['head'])
        if 'backbone' in ck: model.backbone.load_state_dict(ck['backbone'])
    sample=load_input(a.input)
    sample['id']=int(a.input.stem) if a.input.stem.isdigit() else 0
    if benchmark and len(sample['coord'])!=config['dataset']['num_points']:
        rng=np.random.default_rng(np.random.SeedSequence([config['seed'],0,sample['id']]))
        ids=rng.choice(len(sample['coord']),config['dataset']['num_points'],replace=len(sample['coord'])<config['dataset']['num_points'])
        for k in ('coord','color','normal'): sample[k]=sample[k][ids]
    # Empty targets are a batching API detail; annotations are never read from input.
    sample.update(boxes=np.empty((0,7),np.float32),labels=np.empty(0,np.int64))
    options={'scale':config['dataset']['backbone_scale'],'voxel_size':config['dataset']['voxel_size']} if benchmark else {}
    points,bounds,_=prepare_batch([sample],a.device,**options)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.time()
    amp=benchmark and config['training']['amp']
    dtype=torch.bfloat16 if not benchmark or config['training']['amp_dtype']=='bfloat16' else torch.float16
    with torch.no_grad(),torch.autocast('cuda',dtype=dtype,enabled=amp): out=model(points,bounds)
    torch.cuda.synchronize(); elapsed=time.time()-started
    probs=out['logits'][0].softmax(-1); scores,labels=probs[:,:10].max(-1)
    boxes=out['boxes'][0].cpu().numpy(); scores=scores.cpu().numpy(); labels=labels.cpu().numpy()
    assert np.isfinite(boxes).all() and (boxes[:,3:6]>0).all()
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez(a.output/'all_queries.npz',boxes=boxes,scores=scores,labels=labels)
    if benchmark:
        pred=postprocess(out,[sample],config['evaluation'])[sample['id']]
        boxes,scores,labels=pred['boxes'],pred['scores'],pred['labels']
    keep=scores>=a.score_threshold
    np.save(a.output/'boxes.npy',boxes[keep])
    rows=[{'box':b.tolist(),'class_id':int(l),'class':CLASSES[l],'score':float(s)} for b,l,s in zip(boxes[keep],labels[keep],scores[keep])]
    (a.output/'boxes.json').write_text(json.dumps(rows,indent=2))
    visualize(a.output/'boxes.html',sample['coord'],sample['color'],boxes[keep],labels[keep],scores[keep],title='Utonia SUN RGB-D detector' if benchmark else 'Utonia learned detector (small-set experimental head)')
    report={'input':str(a.input),'head':str(a.head),'box_format':'cx cy cz dx dy dz yaw; full dimensions; metres; CCW about Z',
            'coordinate':'input upright-depth metric coordinates preserved','num_boxes':int(keep.sum()),
            'forward_seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'gpu':torch.cuda.get_device_name(),'score_threshold':a.score_threshold,'GT_used':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
