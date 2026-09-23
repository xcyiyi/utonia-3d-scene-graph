import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SUNRGBDSubset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)/'processed_v2'
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.ids = self.manifest['splits'][split]

    def __len__(self): return len(self.ids)
    def __getitem__(self, index):
        sid = self.ids[index]
        with np.load(self.root/f'{sid:06d}.npz',allow_pickle=False) as f:
            sample = {k:f[k].copy() for k in f.files}
        sample['id'] = sid
        return sample


class SUNRGBDFull(Dataset):
    def __init__(self,processed_dir,split,num_points=20000,augmentation=None,seed=37):
        self.root=Path(processed_dir).expanduser()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest.get('subset_only',True): raise ValueError('Full benchmark requires full dataset manifest')
        self.ids=self.manifest['splits'][split]; self.num_points=num_points
        self.augmentation=augmentation if split=='train' else None; self.seed=seed; self.epoch=0
    def __len__(self): return len(self.ids)
    def set_epoch(self,epoch): self.epoch=epoch
    def __getitem__(self,index):
        from .augmentation import augment
        sid=self.ids[index]
        with np.load(self.root/f'{sid:06d}.npz',allow_pickle=False) as f:
            sample={k:f[k].copy() for k in ('coord','color','normal','boxes','labels')}
        sample['id']=sid
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch if self.augmentation else 0,sid]))
        ix=rng.choice(len(sample['coord']),self.num_points,replace=len(sample['coord'])<self.num_points)
        for k in ('coord','color','normal'): sample[k]=sample[k][ix]
        if self.augmentation: sample=augment(sample,rng,self.augmentation)
        return sample


def collate_samples(samples): return samples


def prepare_batch(samples, device, scale=0.5,voxel_size=0.01):
    """Deterministic 1cm backbone voxels; carry original metric XYZ separately.

    Detection boxes never use the shifted/scaled backbone input coordinates.
    First point per voxel replaces the demo's random representative for caching.
    """
    points = {k:[] for k in ('coord','origin_coord','grid_coord','feat')}
    offsets, targets, sizes = [], [], []
    for sample in samples:
        xyz = sample['coord'].astype(np.float32)
        coord = xyz*scale
        low, high = coord.min(0), coord.max(0)
        coord -= np.array([(low[0]+high[0])/2,(low[1]+high[1])/2,low[2]],dtype=np.float32)
        grid = np.floor((coord-coord.min(0))/voxel_size).astype(np.int64)
        _, ix = np.unique(grid,axis=0,return_index=True)
        feat = np.concatenate((coord[ix],sample['color'][ix]/255,sample['normal'][ix]),axis=1).astype(np.float32)
        for k,v in dict(coord=coord[ix],origin_coord=xyz[ix],grid_coord=grid[ix],feat=feat).items():
            points[k].append(torch.from_numpy(v).to(device))
        sizes.append(len(ix)); offsets.append(sum(sizes))
        targets.append({'boxes':torch.as_tensor(sample['boxes'],device=device),
                        'labels':torch.as_tensor(sample['labels'],device=device)})
    points = {k:torch.cat(v) for k,v in points.items()}
    points['offset'] = torch.tensor(offsets,device=device,dtype=torch.long)
    bounds = torch.tensor(np.stack([[s['coord'].min(0),s['coord'].max(0)] for s in samples]),device=device)
    return points,bounds,targets
