"""Native Python port of official SUN RGB-D depth projection and VoteNet V2 boxes."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from PIL import Image
from scipy.io import loadmat
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from detection.geometry import CLASSES, inside, visualize


def project_rgbd(root, m):
    def rawpath(k): return root / ('SUNRGBD/'+m[k].split('/SUNRGBD/')[1])
    encoded=np.asarray(Image.open(rawpath('depthpath')),dtype=np.uint16)
    depth=np.minimum(((encoded>>3)|(encoded<<13)).astype(np.float32)/1000,8)
    rgb=np.asarray(Image.open(rawpath('rgbpath')).convert('RGB'))
    if rgb.shape[:2]!=depth.shape: raise ValueError('RGB/depth shape mismatch')
    y,x=np.indices(depth.shape,dtype=np.float32); k=m['K']; valid=depth>0
    xyz=np.stack(((x+1-k[0,2])*depth/k[0,0],depth,-(y+1-k[1,2])*depth/k[1,1]),-1)
    return (xyz[valid]@m['Rtilt'].T).astype(np.float32),rgb[valid]


def convert_boxes(m):
    gt=m.get('groundtruth3DBB',[]); gt=[gt] if isinstance(gt,dict) else gt
    boxes=[]; labels=[]
    for g in gt:
        if g['classname'] not in CLASSES: continue
        size=2*np.abs(np.asarray(g['coeffs']))[[1,0,2]]
        yaw=np.arctan2(g['orientation'][1],g['orientation'][0])
        boxes.append(np.r_[g['centroid'],size,yaw]); labels.append(CLASSES.index(g['classname']))
    return np.asarray(boxes,dtype=np.float32).reshape(-1,7),np.asarray(labels,dtype=np.int64)


def prepare(root, ids_path, output, num_points):
    meta = loadmat(root/'SUNRGBDMeta3DBB_v2.mat',simplify_cells=True)['SUNRGBDMeta']
    split_meta = loadmat(root/'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat',simplify_cells=True)
    splits = json.loads(ids_path.read_text())
    assert not set(splits['train']) & set(splits['val'])
    output.mkdir(parents=True,exist_ok=True)
    processed = root/'processed_v2'
    processed.mkdir(exist_ok=True)
    reports = []
    for split, ids in splits.items():
        official = {str(p).split('/SUNRGBD/')[-1] for p in split_meta['alltrain' if split=='train' else 'alltest']}
        for sid in ids:
            m = meta[sid-1]
            assert m['sequenceName'].split('SUNRGBD/')[-1] in official
            def rawpath(k): return root / ('SUNRGBD/'+m[k].split('/SUNRGBD/')[1])
            encoded = np.asarray(Image.open(rawpath('depthpath')), dtype=np.uint16)
            depth = ((encoded >> 3) | (encoded << 13)).astype(np.float32)/1000
            depth = np.minimum(depth,8.0)
            rgb = np.asarray(Image.open(rawpath('rgbpath')).convert('RGB'))
            assert rgb.shape[:2] == depth.shape
            y,x = np.indices(depth.shape,dtype=np.float32)
            k = m['K']
            # MATLAB pixels start at one; use the official depth coordinate order.
            xyz = np.stack(((x+1-k[0,2])*depth/k[0,0],depth,-(y+1-k[1,2])*depth/k[1,1]),axis=-1)
            valid = depth>0
            xyz = (xyz[valid] @ m['Rtilt'].T).astype(np.float32)
            rgb = rgb[valid]
            ix = np.random.default_rng(sid).choice(len(xyz),min(num_points,len(xyz)),replace=False)
            xyz,rgb = xyz[ix],rgb[ix]
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
            cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15,max_nn=30))
            cloud.orient_normals_towards_camera_location(np.zeros(3))
            normal = np.asarray(cloud.normals).astype(np.float32)
            gt = m.get('groundtruth3DBB',[])
            gt = [gt] if isinstance(gt,dict) else gt
            boxes,labels = [],[]
            for g in gt:
                if g['classname'] not in CLASSES: continue
                # VoteNet SUNObject3d swaps coeffs x/y: l=coeffs[1], w=coeffs[0].
                size = 2*np.abs(np.asarray(g['coeffs']))[[1,0,2]]
                yaw = np.arctan2(g['orientation'][1],g['orientation'][0])
                boxes.append(np.r_[g['centroid'],size,yaw])
                labels.append(CLASSES.index(g['classname']))
            boxes = np.asarray(boxes,dtype=np.float32).reshape(-1,7)
            labels = np.asarray(labels,dtype=np.int64)
            assert len(xyz)>1000 and np.isfinite(xyz).all() and np.isfinite(normal).all()
            assert len(boxes) and np.isfinite(boxes).all() and (boxes[:,3:6]>0).all()
            counts = [int(inside(xyz,b).sum()) for b in boxes]
            assert max(counts)>20, f'Boxes do not overlap points: {sid}, {counts}'
            dest = processed/f'{sid:06d}.npz'
            np.savez_compressed(dest,coord=xyz,color=rgb,normal=normal,boxes=boxes,labels=labels)
            report = {'id':sid,'split':split,'points':len(xyz),'boxes':len(boxes),
                'classes':[CLASSES[i] for i in labels],'points_in_each_box':counts,
                'xyz_min':xyz.min(0).tolist(),'xyz_max':xyz.max(0).tolist(),
                'sequence':m['sequenceName'],'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
            reports.append(report)
            print(json.dumps(report),flush=True)
            visualize(output/f'{sid:06d}_gt.html',xyz,rgb,boxes,labels,title=f'SUN RGB-D {sid:06d} {split}: GT V2')
    manifest = {'annotation_version':'SUNRGBDMeta3DBB_v2','classes':CLASSES,
        'box_format':'cx cy cz dx dy dz yaw; metres; FULL dimensions; yaw CCW about +Z',
        'coordinate':'upright_depth: X right, Y forward, Z up; original origin retained',
        'depth':'official uint16 bit rotation, /1000, cap 8m, discard zero, K 1-based, Rtilt',
        'normals':'Open3D estimate, radius .15m / max_nn 30, oriented toward camera origin',
        'splits':splits,'samples':reports,'subset_only':True}
    (processed/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'dataset_sanity.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--ids',type=Path,default=ROOT/'outputs/detection/subset_ids.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/detection/gt')
    p.add_argument('--num-points',type=int,default=20000)
    a=p.parse_args(); prepare(a.root,a.ids,a.output,a.num_points)
