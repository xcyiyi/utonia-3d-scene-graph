"""Static overview supplement to the four interactive error-analysis scenes."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from detection.geometry import corners

root=Path.home()/'dabaset/SUNRGBD'
ids=json.loads((root/'processed_v2/manifest.json').read_text())['splits']['val']
fig,axes=plt.subplots(2,2,figsize=(13,11))
for ax,sid in zip(axes.flat,ids):
    d=np.load(root/f'processed_v2/{sid:06d}.npz')
    pred=np.load(ROOT/f'outputs/detection/training/predictions/val_unseen/{sid:06d}_all_queries.npz')
    xyz=d['coord'][::3]; ax.scatter(xyz[:,0],xyz[:,1],c=d['color'][::3]/255,s=1)
    for boxes,color,label in [(d['boxes'],'green','GT'),(pred['boxes'][pred['scores']>=.3],'red','pred score>=0.3')]:
        for i,box in enumerate(boxes):
            c=corners(box)[[0,1,2,3,0]]; ax.plot(c[:,0],c[:,1],color=color,label=label if i==0 else None)
    ax.legend(); ax.set_aspect('equal'); ax.set_title(f'Validation {sid:06d}'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
fig.tight_layout(); fig.savefig(ROOT/'outputs/benchmark/audit/all_val_overlays.png',dpi=150)
