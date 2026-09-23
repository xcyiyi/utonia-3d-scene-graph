import numpy as np
from .geometry import CLASSES, iou3d


def evaluate(predictions, targets, thresholds=(0.25,0.5), score_threshold=0.3, fixed_classes=False):
    """Exact oriented 3D IoU and all-point interpolated AP over supplied detections.

    AP uses all supplied scores. Fixed-threshold P/R uses score_threshold.
    Absent classes have AP=None; fixed_classes=True includes them as zero in
    the ten-class mean, while legacy mode averages only present classes.
    """
    results={}
    # Compute geometry once per scene, shared by thresholds and all classes.
    from .box_ops import pairwise_iou_giou
    import torch
    overlaps_by_scene={}
    for sid,p in predictions.items():
        unique,inverse=np.unique(p['boxes'],axis=0,return_inverse=True)
        with torch.no_grad():
            overlaps_by_scene[sid]=pairwise_iou_giou(torch.as_tensor(unique),torch.as_tensor(targets[sid]['boxes']))[0].numpy()[inverse]
    for threshold in thresholds:
        per_class={}; aggregate=np.zeros(3,dtype=int)
        for label,name in enumerate(CLASSES):
            gt={sid:t['boxes'][t['labels']==label] for sid,t in targets.items()}
            count=sum(len(v) for v in gt.values())
            detections=sorted([(float(score),sid,i) for sid,p in predictions.items()
                for i,(cl,score) in enumerate(zip(p['labels'],p['scores'])) if cl==label],key=lambda x:-x[0])
            used={sid:set() for sid in gt}; tp=[]; scores=[]
            for score,sid,index in detections:
                overlaps=overlaps_by_scene[sid][index,targets[sid]['labels']==label]
                best=int(np.argmax(overlaps)) if len(overlaps) else -1
                hit=best>=0 and overlaps[best]>=threshold and best not in used[sid]
                if hit: used[sid].add(best)
                tp.append(int(hit)); scores.append(score)
            tp=np.asarray(tp); scores=np.asarray(scores)
            fixed_tp=int(tp[scores>=score_threshold].sum())
            fixed_fp=int((scores>=score_threshold).sum())-fixed_tp
            aggregate+=np.array([fixed_tp,fixed_fp,count-fixed_tp])
            ap=None
            if count:
                recall=np.cumsum(tp)/count
                precision=np.cumsum(tp)/np.maximum(np.arange(len(tp))+1,1)
                mr=np.r_[0,recall,1]; mp=np.r_[0,precision,0]
                mp=np.maximum.accumulate(mp[::-1])[::-1]
                ap=float(np.sum(np.diff(mr)*mp[1:]))
            per_class[name]={'gt':count,'AP':ap,'TP':fixed_tp,'FP':fixed_fp,'FN':count-fixed_tp}
        tp,fp,fn=map(int,aggregate)
        aps=[p['AP'] for p in per_class.values() if p['AP'] is not None]
        results[str(threshold)]={'mAP_present_classes':float(np.mean(aps)) if aps else None,
            'num_present_classes':len(aps),'TP':tp,'FP':fp,'FN':fn,
            'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'per_class':per_class}
        if fixed_classes:
            results[str(threshold)]['mAP']=float(sum(p['AP'] or 0 for p in per_class.values())/len(CLASSES))
            results[str(threshold)]['class_count']=len(CLASSES)
            del results[str(threshold)]['mAP_present_classes']
    return {'IoU':'exact oriented 3D cuboid intersection','AP':'all-point interpolation over supplied scored detections; postprocessing is caller-controlled',
            'score_threshold_for_precision_recall':score_threshold,'thresholds':results}
