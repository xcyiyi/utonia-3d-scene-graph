"""Shared benchmark configuration, model construction and detection postprocessing."""
from pathlib import Path
import numpy as np
import torch
import yaml
from .model import UtoniaDetector
from .geometry import CLASSES,inside,corners
from .box_ops import pairwise_iou_giou


def load_config(path):
    config=yaml.safe_load(Path(path).read_text())
    assert config['dataset']['classes']==CLASSES
    assert config['evaluation']['classes']=='fixed10'
    assert config['model']['feature_stages']==[2,3,4]
    assert config['model']['feature_dimensions']==[216,432,576]
    assert config['model']['position_encoding']=='xyz_fourier'
    assert config['model']['box_parameterization']=='relative_center_softplus_size_double_angle'
    assert config['training']['optimizer']=='AdamW' and config['training']['scheduler']=='cosine'
    return config


def build_model(config,ablation=None):
    ablation=ablation or config['training']['ablation']
    if ablation not in ('A','B','C'): raise ValueError(ablation)
    m=config['model']
    return UtoniaDetector(m['checkpoint'],freeze_backbone=ablation=='A',pretrained=ablation!='C',
        **{k:m[k] for k in ['dim','layers','attention_heads','queries','max_tokens','adapter_type','query_type']})


def postprocess(output,samples,config):
    """VoteNet/3DETR protocol: objectness-filtered class-aware AABB NMS, then per-class proposals.

    AP itself uses oriented 3D IoU. Foreground probabilities already equal
    conditional semantic probability times objectness for our 11-way softmax.
    """
    results={}
    for batch,sample in enumerate(samples):
        boxes=output['boxes'][batch].detach().float().cpu().numpy()
        probs=output['logits'][batch].detach().float().softmax(-1).cpu().numpy()[:,:10]
        objectness=probs.sum(-1); predicted_class=probs.argmax(-1)
        keep=np.ones(len(boxes),dtype=bool)
        if config['remove_empty_boxes']:
            keep=np.array([inside(sample['coord'],box).sum()>=config['min_points_in_box'] for box in boxes])
        if not keep.any(): keep[objectness.argmax()]=True
        c=np.stack([corners(b) for b in boxes]); low=c.min(1); high=c.max(1)
        inter=np.maximum(0,np.minimum(high[:,None],high[None])-np.maximum(low[:,None],low[None])).prod(-1)
        vol=(high-low).prod(-1); overlaps=inter/np.maximum(vol[:,None]+vol[None]-inter,1e-8)
        order=list(np.where(keep)[0][np.argsort(-objectness[keep])]); selected=[]
        while order:
            best=order.pop(0); selected.append(best)
            order=[i for i in order if predicted_class[i]!=predicted_class[best] or overlaps[best,i]<=config['nms_iou']]
        proposals=[]
        for i in selected:
            if objectness[i]<=config['score_threshold']: continue
            for label in (range(10) if config['per_class_proposals'] else [predicted_class[i]]):
                proposals.append((i,label,float(probs[i,label])))
        results[sample['id']]={'boxes':np.array([boxes[i] for i,_,_ in proposals],np.float32).reshape(-1,7),
            'labels':np.array([label for _,label,_ in proposals],np.int64),
            'scores':np.array([score for _,_,score in proposals],np.float32)}
    return results
