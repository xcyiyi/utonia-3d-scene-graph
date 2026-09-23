import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import utonia
from detection.data import SUNRGBDSubset,prepare_batch

torch.set_num_threads(8); torch.cuda.set_device(0)
model=utonia.load(str(ROOT/'checkpoints/utonia.pth')).cuda().eval()
data=SUNRGBDSubset(Path.home()/'dabaset/SUNRGBD','val'); records=[]
for sample in data:
    points,bounds,_=prepare_batch([sample],'cuda')
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out=model(points); stages={}
        for s in range(4,-1,-1):
            stages[s]={'points':len(out.feat),'dim':out.feat.shape[1],'xyz':list(out.origin_coord.shape),
                       'physical_grid_metres':.02*2**s}
            if 'pooling_parent' in out: out=out.pooling_parent
    records.append({'id':sample['id'],'stages':stages,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
path=ROOT/'outputs/benchmark/audit/stage_profile.json'; path.write_text(json.dumps(records,indent=2)); print(json.dumps(records,indent=2))
