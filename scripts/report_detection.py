"""Render final experiment plots and verify exported prediction consistency."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.geometry import corners


def main():
    out=ROOT/'outputs/detection'; train=out/'training'
    rows=[json.loads(l) for l in (train/'overfit_loss.jsonl').read_text().splitlines()]
    losses=np.array([r['total'] for r in rows])
    assert len(rows)==1500 and np.isfinite(losses).all()
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(np.arange(1,len(losses)+1),losses,alpha=.35,label='per batch')
    ax.plot(np.arange(25,len(losses)+1),np.convolve(losses,np.ones(25)/25,'valid'),label='25-step mean')
    ax.set(xlabel='Optimizer step (frozen Utonia features)',ylabel='Detection loss',yscale='log')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'loss_curve.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for row,(sid,split) in enumerate([(6307,'train_overfit'),(1110,'val_unseen')]):
        data=np.load(Path.home()/f'dabaset/SUNRGBD/processed_v2/{sid:06d}.npz')
        pred=np.load(train/f'predictions/{split}/{sid:06d}_all_queries.npz')
        keep=pred['scores']>=.3
        for ax,boxes,title in zip(axes[row],[data['boxes'],pred['boxes'][keep]],['GT','Predicted score >= 0.3']):
            xyz=data['coord']; ax.scatter(xyz[::3,0],xyz[::3,1],c=data['color'][::3]/255,s=1)
            for box in boxes:
                p=corners(box)[[0,1,2,3,0]]; ax.plot(p[:,0],p[:,1],linewidth=1.5)
            ax.set_title(f'{split} {sid:06d}: {title} ({len(boxes)} boxes)')
            ax.set_aspect('equal'); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    fig.tight_layout(); fig.savefig(out/'box_comparison.png',dpi=150); plt.close(fig)
    cached=np.load(train/'predictions/val_unseen/001110_all_queries.npz')
    standalone=np.load(out/'standalone_val_001110/all_queries.npz')
    error=float(np.max(np.abs(cached['boxes']-standalone['boxes'])))
    np.testing.assert_allclose(cached['boxes'],standalone['boxes'],rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(cached['scores'],standalone['scores'],rtol=1e-4,atol=1e-4)
    summary={'overfit_steps':len(rows),'mean_loss_first_50':float(losses[:50].mean()),
        'mean_loss_last_50':float(losses[-50:].mean()),'all_1500_losses_finite':True,
        'standalone_vs_cached_max_box_difference':error,
        'standalone_checkpoint_reload_passed':True,
        'metrics':json.loads((train/'final_metrics.json').read_text())}
    (out/'acceptance_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))


if __name__=='__main__': main()
