"""Full official train/val preprocessing using the audited projection/box functions."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
import open3d as o3d
from prepare_sunrgbd import project_rgbd,convert_boxes,CLASSES


def worker(task):
    root,output,sid,m,count=task
    dest=output/f'{sid:06d}.npz'
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f:
            if int(f['preprocess_version'])!=1 or len(f['coord'])!=count: raise ValueError(f'Incompatible cached scene {sid}')
            return sid,len(f['coord']),f['labels'].tolist(),False
    xyz,rgb=project_rgbd(root,m)
    if len(xyz)==0 or not np.isfinite(xyz).all(): raise ValueError(f'Invalid point cloud {sid}')
    ix=np.random.default_rng(sid).choice(len(xyz),count,replace=len(xyz)<count)
    xyz,rgb=xyz[ix],rgb[ix]
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.15,max_nn=30))
    cloud.orient_normals_towards_camera_location(np.zeros(3))
    normals=np.asarray(cloud.normals).astype(np.float32)
    boxes,labels=convert_boxes(m)
    if not np.isfinite(boxes).all(): raise ValueError(f'Nonfinite GT {sid}')
    # Official V1 has two zero-height training annotations (7334, 8481).
    # Exclude zero-volume targets explicitly; never alter validation boxes silently.
    valid=(boxes[:,3:6]>0).all(1)
    if not valid.all() and sid not in (7334,8481): raise ValueError(f'Unexpected degenerate GT {sid}')
    boxes,labels=boxes[valid],labels[valid]
    tmp=dest.with_suffix('.part')
    with tmp.open('wb') as f: np.savez_compressed(f,coord=xyz,color=rgb,normal=normals,boxes=boxes,labels=labels,preprocess_version=np.array(1))
    tmp.replace(dest)
    return sid,len(xyz),labels.tolist(),True


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--version',choices=['v1','v2'],default='v1'); p.add_argument('--workers',type=int,default=12)
    p.add_argument('--num-points',type=int,default=50000); a=p.parse_args()
    meta_file=a.root/('SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat' if a.version=='v1' else 'SUNRGBDMeta3DBB_v2.mat')
    meta=loadmat(meta_file,simplify_cells=True)['SUNRGBDMeta']
    split=loadmat(a.root/'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat',simplify_cells=True)
    def canonical(s): return Path(str(s).split('SUNRGBD/')[-1]).as_posix().rstrip('/')
    index={canonical(m['sequenceName']):i+1 for i,m in enumerate(meta)}
    splits={key:sorted(index[canonical(path)] for path in split[source]) for key,source in [('train','alltrain'),('val','alltest')]}
    assert len(splits['train'])==5285 and len(splits['val'])==5050
    assert not set(splits['train'])&set(splits['val']) and len(index)==10335
    output=a.root/f'processed_full_{a.version}'; output.mkdir(exist_ok=True)
    (output/'split_ids.json').write_text(json.dumps(splits,indent=2))
    tasks=[(a.root,output,i+1,m,a.num_points) for i,m in enumerate(meta)]
    started=time.time(); results={}; errors=[]
    with ProcessPoolExecutor(a.workers) as pool:
        futures={pool.submit(worker,t):t[2] for t in tasks}
        for future in as_completed(futures):
            try:
                sid,n,labels,new=future.result(); results[sid]={'points':n,'labels':labels}
            except Exception as e: errors.append({'id':futures[future],'error':str(e)})
            done=len(results)+len(errors)
            if done%100==0 or done==len(tasks):
                status={'completed':len(results),'failed':len(errors),'total':len(tasks),'seconds':round(time.time()-started,1),'errors':errors}
                (output/'progress.json').write_text(json.dumps(status,indent=2))
                print(json.dumps({k:v for k,v in status.items() if k!='errors'}),flush=True)
    if errors: raise RuntimeError(f'{len(errors)} failed preprocessing; see progress.json; rerun resumes completed files')
    stats={}
    for name,ids in splits.items():
        c=Counter(l for sid in ids for l in results[sid]['labels'])
        stats[name]={'scenes':len(ids),'empty_scenes':sum(not results[sid]['labels'] for sid in ids),
                     'class_counts':{name:c[i] for i,name in enumerate(CLASSES)}}
    manifest={'annotation_version':a.version,'classes':CLASSES,'splits':splits,'stats':stats,'num_points':a.num_points,
        'preprocessing':'audited project_rgbd and convert_boxes; original metric XYZ; full sizes; CCW yaw',
        'empty_scene_policy':'keep all official split scenes as negatives; report this versus implementations that drop empty scenes',
        'invalid_box_policy':'exclude the two audited zero-height V1 train boxes in scenes 7334 and 8481; validation unchanged',
        'subset_only':False,'source_metadata':str(meta_file)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(stats,indent=2))


if __name__=='__main__': main()
