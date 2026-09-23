"""Audit the existing prototype without changing its model or evaluation protocol."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.geometry import CLASSES,corners,EDGES,iou3d
from detection.evaluate import evaluate


def main():
    out=ROOT/'outputs/benchmark/audit'; out.mkdir(parents=True,exist_ok=True)
    data_root=Path.home()/'dabaset/SUNRGBD'
    manifest=json.loads((data_root/'processed_v2/manifest.json').read_text())
    targets={}; data={}; predictions={}
    for sid in manifest['splits']['val']:
        d=dict(np.load(data_root/f'processed_v2/{sid:06d}.npz'))
        data[sid]=d; targets[sid]={'boxes':d['boxes'],'labels':d['labels']}
        predictions[sid]=dict(np.load(ROOT/f'outputs/detection/training/predictions/val_unseen/{sid:06d}_all_queries.npz'))
    sanity={}
    for name,center_std,size_std,yaw_std in [('GT',0,0,0),('small_center',.02,0,0),('small_size',0,.03,0),
            ('small_yaw',0,0,3),('large_center',.5,0,0),('large_size',0,.6,0),('large_yaw',0,0,45)]:
        rng=np.random.default_rng(37); pred={}
        for sid,t in targets.items():
            b=t['boxes'].copy(); b[:,:3]+=rng.normal(0,center_std,b[:,:3].shape)
            b[:,3:6]*=np.exp(rng.normal(0,size_std,b[:,3:6].shape))
            b[:,6]+=rng.normal(0,np.deg2rad(yaw_std),len(b))
            pred[sid]={'boxes':b,'labels':t['labels'],'scores':np.ones(len(b))}
        m=evaluate(pred,targets)
        sanity[name]={t:v['mAP_present_classes'] for t,v in m['thresholds'].items()}
    assert all(abs(v-1)<1e-6 for v in sanity['GT'].values())
    (out/'gt_evaluation_sanity.json').write_text(json.dumps(sanity,indent=2))
    # Independently reconstruct VoteNet's published depth-frame corners.
    max_corner_error=0
    for t in targets.values():
        for b in t['boxes']:
            l,w,h=b[3:6]/2; angle=-b[6]
            c,s=np.cos(-angle),np.sin(-angle)
            R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            ref=np.array([[-l,l,l,-l,-l,l,l,-l],[w,w,-w,-w,w,w,-w,-w],[h,h,h,h,-h,-h,-h,-h]]).T@R.T+b[:3]
            dist=np.linalg.norm(ref[:,None]-corners(b)[None],axis=-1)
            max_corner_error=max(max_corner_error,float(dist.min(1).max()))
    assert max_corner_error<1e-5
    rows=[]; gtrows=[]; scenes={}; fixes={k:[] for k in ['original','center','size','yaw','center_size','all_geometry']}
    for sid,p in predictions.items():
        d=data[sid]; gt=targets[sid]; boxes=p['boxes']; g=gt['boxes']
        ious=np.array([[iou3d(b,t) for t in g] for b in boxes])
        for i,b in enumerate(boxes):
            j=int(ious[i].argmax()) if ious[i].max()>0 else int(np.linalg.norm(g[:,:3]-b[:3],axis=1).argmin())
            yaw_error=abs((b[6]-g[j,6]+np.pi/2)%np.pi-np.pi/2)
            rows.append({'scene':sid,'query':i,'GT_index':j,'GT_class':CLASSES[gt['labels'][j]],'pred_class':CLASSES[p['labels'][i]],
                'confidence':float(p['scores'][i]),'center_error_m':float(np.linalg.norm(b[:3]-g[j,:3])),
                'size_MAE_m':float(abs(b[3:6]-g[j,3:6]).mean()),'yaw_error_deg':float(np.rad2deg(yaw_error)),
                'best_GT_IoU':float(ious[i,j]),'score_ge_0.3':bool(p['scores'][i]>=.3)})
        for j,t in enumerate(g):
            same=p['labels']==gt['labels'][j]; visible=p['scores']>=.3
            gtrows.append({'scene':sid,'GT_index':j,'GT_class':CLASSES[gt['labels'][j]],
                'best_IoU_any_class_all_queries':float(ious[:,j].max()),
                'best_IoU_correct_class_all_queries':float(ious[same,j].max()) if same.any() else 0.,
                'best_IoU_visible':float(ious[visible,j].max()) if visible.any() else 0.})
        # Controlled one-factor oracles use a fixed geometry-only Hungarian assignment.
        # Diagnostic only: never feeds predictions or evaluation for model selection.
        ii,jj=linear_sum_assignment(-ious)
        for i,j in zip(ii,jj):
            b,t=boxes[i],g[j]
            for name in fixes:
                altered=b.copy()
                if name in ('center','center_size','all_geometry'): altered[:3]=t[:3]
                if name in ('size','center_size','all_geometry'): altered[3:6]=t[3:6]
                if name in ('yaw','all_geometry'): altered[6]=t[6]
                fixes[name].append(iou3d(altered,t))
        scenes[sid]={'GT':len(g),'queries':len(boxes),'visible_predictions':int((p['scores']>=.3).sum()),
            'best_geometric_IoU':float(ious.max()),'GT_with_any_query_IoU50':int((ious.max(0)>=.5).sum()),
            'visible_queries_IoU25_with_same_GT_over_one':sum(max(0,int(((ious[:,j]>=.25)&(p['scores']>=.3)).sum())-1) for j in range(len(g)))}
        ids=np.linspace(0,len(d['coord'])-1,min(15000,len(d['coord'])),dtype=int); xyz=d['coord'][ids]
        fig=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',marker=dict(size=1,color=d['color'][ids]),name='point cloud'))
        for prefix,bs,ls,ss,color in [('GT',g,gt['labels'],np.ones(len(g)),'lime'),('PRED',boxes,p['labels'],p['scores'],'red')]:
            for i,(b,l,score) in enumerate(zip(bs,ls,ss)):
                c=corners(b); lines=np.array([v for a,z in EDGES for v in (c[a],c[z],[np.nan]*3)])
                fig.add_trace(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines',line=dict(color=color,width=5),
                    name=f'{prefix} {i} {CLASSES[l]} {score:.3f}',visible=True if prefix=='GT' or score>=.3 else 'legendonly'))
        fig.update_layout(title=f'{sid:06d}: green GT / red predictions; low scores toggle in legend',scene=dict(aspectmode='data'))
        fig.write_html(out/f'{sid:06d}_overlay.html',include_plotlyjs=True)
    for name,items in [('predictions',rows),('gt_coverage',gtrows)]:
        with (out/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=items[0].keys()); w.writeheader(); w.writerows(items)
    summary={'coordinate':'GT, points and decoded predictions all original upright_depth XYZ, metres',
        'center':'geometric cuboid center','size':'full dx dy dz; official coeffs half-size [w,l,h] swapped and doubled',
        'yaw':'CCW about +Z, zero +X; internal yaw = negative of VoteNet heading; cuboid periodicity pi',
        'independent_votenet_corner_max_error':max_corner_error,
        'train_inference_decode':'same DetectionHead.forward; previous independent saved-checkpoint predictions exactly matched',
        'stages':'existing Stage3/4 concat; FPS queries+features, XYZ Fourier PE, relative centers already present',
        'scenes':scenes,'one_factor_oracle_fixed_geometry_assignment':{k:{'mean_iou':float(np.mean(v)),'matched_IoU50_count':int((np.array(v)>=.5).sum())} for k,v in fixes.items()},
        'GT_sanity':sanity,'warning':'oracle corrections are diagnostics, not detector performance'}
    (out/'audit.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
