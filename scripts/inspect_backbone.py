"""Measure the actual pretrained encoder hierarchy before designing detection."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utonia

torch.set_num_threads(8)
utonia.utils.set_seed(37)
model = utonia.load(str(ROOT / 'checkpoints/utonia.pth')).cuda().eval()
with np.load(ROOT / 'data/sample1.npz') as f:
    raw = {k: f[k].copy() for k in ('coord', 'color', 'normal')}
point = utonia.transform.default(scale=0.5)(raw)
keys = ('coord', 'grid_coord', 'feat', 'batch', 'offset', 'inverse', 'pooling_parent', 'pooling_inverse')

def describe(p):
    return {k: ({'shape': list(p[k].shape), 'dtype': str(p[k].dtype),
                 **({'values': p[k].tolist()} if k == 'offset' else {})}
                if isinstance(p.get(k), torch.Tensor) else
                ('Point (previous stage)' if k in p else None)) for k in keys}

report = {'gpu': torch.cuda.get_device_name(), 'input': describe(point), 'stages': []}
with torch.no_grad():
    out = model({k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in point.items()})
    for stage in range(4, -1, -1):
        assert torch.isfinite(out.feat).all()
        report['stages'].append({'stage': stage, **describe(out)})
        if 'pooling_parent' in out:
            out = out.pooling_parent
report['stages'].reverse()
report['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30
path = ROOT / 'outputs/detection/backbone_shapes.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
