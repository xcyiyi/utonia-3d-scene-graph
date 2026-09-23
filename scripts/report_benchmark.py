"""Render saved benchmark validation predictions and short-training loss curves."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from detection.geometry import corners,EDGES,CLASSES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--scenes',type=int,nargs='+',default=[1110,1156,270,1232])
    p.add_argument('--score-threshold',type=float,default=.1)
    a=p.parse_args(); out=a.run/'visualizations'; out.mkdir(exist_ok=True)
    metadata=json.loads((a.run/'run_config.json').read_text())
    data=Path(metadata['config']['dataset']['processed_dir'])
    predictions={}; targets={}
    for path in sorted(a.run.glob('val_rank*.pt')):
        # Only consume this run's locally generated arrays.
        shard=torch.load(path,map_location='cpu',weights_only=False)
        predictions.update({sid:shard['predictions'][sid] for sid in a.scenes if sid in shard['predictions']})
        targets.update({sid:shard['targets'][sid] for sid in a.scenes if sid in shard['targets']})
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    for ax,sid in zip(axes.flat,a.scenes):
        with np.load(data/f'{sid:06d}.npz') as d: xyz=d['coord'][::4]; rgb=d['color'][::4]
        pred=predictions[sid]; gt=targets[sid]
        # Per-class proposals share geometry: show the highest class score per box.
        _,ix=np.unique(pred['boxes'],axis=0,return_index=True)
        visible=[]
        for i in ix:
            same=(pred['boxes']==pred['boxes'][i]).all(1); candidates=np.where(same)[0]
            best=candidates[pred['scores'][candidates].argmax()]
            if pred['scores'][best]>=a.score_threshold: visible.append(best)
        bs=pred['boxes'][visible]; ls=pred['labels'][visible]; ss=pred['scores'][visible]
        np.savez(out/f'{sid:06d}_boxes.npz',boxes=bs,labels=ls,scores=ss)
        ax.scatter(xyz[:,0],xyz[:,1],c=rgb/255,s=1)
        interactive=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=1,color=rgb),name='RGB points'))
        for name,boxes,labels,scores,color in [('GT',gt['boxes'],gt['labels'],np.ones(len(gt['boxes'])),'green'),('prediction',bs,ls,ss,'red')]:
            for i,(box,label,score) in enumerate(zip(boxes,labels,scores)):
                c=corners(box); loop=c[[0,1,2,3,0]]
                ax.plot(loop[:,0],loop[:,1],color=color,label=name if i==0 else None)
                lines=np.array([v for u,w in EDGES for v in (c[u],c[w],[np.nan]*3)])
                interactive.add_trace(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines',line=dict(color=color,width=5),name=f'{name}: {CLASSES[label]} {score:.3f}'))
        ax.set_title(f'{sid:06d}: GT={len(gt["boxes"])}, pred={len(bs)}, score≥{a.score_threshold}')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_aspect('equal'); ax.legend()
        interactive.update_layout(title=f'{sid:06d}: GT green / predictions red, threshold {a.score_threshold}',scene=dict(aspectmode='data'))
        interactive.write_html(out/f'{sid:06d}_overlay.html',include_plotlyjs=True)
    fig.tight_layout(); fig.savefig(out/'val_overlays.png',dpi=150); plt.close(fig)
    records=[json.loads(line) for line in (a.run/'loss_rank0.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key in zip(axes.flat,['total','class','center','size','heading','iou']):
        values=np.array([r[key] for r in records]); step=np.array([r['step'] for r in records]); window=min(25,len(values))
        ax.plot(step,values,alpha=.2); ax.plot(step[window-1:],np.convolve(values,np.ones(window)/window,mode='valid'))
        ax.set_title(key); ax.set_xlabel('optimizer step (rank 0)'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'loss_curves.png',dpi=150); plt.close(fig)
    print(json.dumps({'output':str(out),'scenes':a.scenes,'visualization_score_threshold':a.score_threshold}))


if __name__=='__main__': main()
