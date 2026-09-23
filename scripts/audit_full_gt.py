"""Fixed-ten-class GT-as-predictions audit over the full official V1 validation split."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.prepare_sunrgbd import convert_boxes,CLASSES
from detection.evaluate import evaluate


def main():
    torch.set_num_threads(2); root=Path.home()/'dabaset/SUNRGBD'
    meta=loadmat(root/'SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat',simplify_cells=True)['SUNRGBDMeta']
    split=loadmat(root/'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat',simplify_cells=True)
    canonical=lambda s:Path(str(s).split('SUNRGBD/')[-1]).as_posix().rstrip('/')
    index={canonical(m['sequenceName']):i+1 for i,m in enumerate(meta)}
    sets={k:sorted(index[canonical(s)] for s in split[v]) for k,v in [('train','alltrain'),('val','alltest')]}
    summary={}; predictions={}; targets={}; processed_checked=0
    processed=root/'processed_full_v1'
    for name,ids in sets.items():
        counts=Counter(); empty=0
        for sid in ids:
            boxes,labels=convert_boxes(meta[sid-1])
            assert np.isfinite(boxes).all(),sid
            valid=(boxes[:,3:6]>0).all(1)
            if not valid.all():
                assert name=='train' and sid in (7334,8481),sid
                boxes,labels=boxes[valid],labels[valid]
            if (processed/'manifest.json').exists():
                with np.load(processed/f'{sid:06d}.npz') as saved:
                    np.testing.assert_allclose(saved['boxes'],boxes,atol=1e-6)
                    np.testing.assert_array_equal(saved['labels'],labels)
                processed_checked+=1
            counts.update(labels.tolist()); empty+=len(boxes)==0
            if name=='val':
                targets[sid]={'boxes':boxes,'labels':labels}
                predictions[sid]={'boxes':boxes.copy(),'labels':labels.copy(),'scores':np.ones(len(boxes))}
        summary[name]={'scenes':len(ids),'empty':empty,'class_counts':{c:counts[i] for i,c in enumerate(CLASSES)}}
    metrics=evaluate(predictions,targets,fixed_classes=True)
    report={'metadata_scene_counts':summary,'processed_data_ready':(root/'processed_full_v1/manifest.json').exists(),
            'processed_scenes_checked':processed_checked,'GT_evaluation':metrics,'annotation_version':'v1'}
    path=ROOT/'outputs/benchmark/audit/full_gt_v1_sanity.json'; path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert all(v['mAP']>=.99 for v in metrics['thresholds'].values()),'Investigate GT evaluation before training'


if __name__=='__main__': main()
