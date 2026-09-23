import numpy as np


def geometric_transform(sample,flip=False,angle=0.,scale=1.):
    sample={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in sample.items()}
    xyz,normal,boxes=sample['coord'],sample['normal'],sample['boxes']
    if flip:
        xyz[:,0]*=-1; normal[:,0]*=-1; boxes[:,0]*=-1; boxes[:,6]=np.pi-boxes[:,6]
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)
    sample['coord']=(xyz@rotation.T)*scale
    sample['normal']=normal@rotation.T
    boxes[:,:3]=(boxes[:,:3]@rotation.T)*scale
    boxes[:,3:6]*=scale; boxes[:,6]+=angle
    boxes[:,6]=(boxes[:,6]+np.pi)%(2*np.pi)-np.pi
    return sample


def augment(sample,rng,config):
    sample=geometric_transform(sample,rng.random()<config['flip_probability'],
        rng.uniform(-config['rotation_degrees'],config['rotation_degrees'])*np.pi/180,
        rng.uniform(*config['scale_range']))
    rgb=sample['color'].astype(np.float32)/255
    rgb*=rng.uniform(*config['brightness_range'],size=3)
    rgb+=rng.uniform(-config['rgb_shift'],config['rgb_shift'],size=3)
    rgb+=rng.uniform(-config['point_color_jitter'],config['point_color_jitter'],size=(len(rgb),1))
    rgb=np.clip(rgb,0,1)
    rgb[rng.random(len(rgb))<config['color_dropout']]=0
    sample['color']=(rgb*255).astype(np.float32)
    return sample
