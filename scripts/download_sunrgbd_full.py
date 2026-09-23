"""Fetch all official split RGB/depth members, reuse existing files, resume by member.

One coalesced HTTP range per ZIP member; verify decompressed length and CRC before
atomic commit. A persistent receipt records provenance. No unrelated ZIP members.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import struct
import threading
import time
import zipfile
import zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.io import loadmat
from download_sunrgbd import RemoteZip


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--workers',type=int,default=16); a=p.parse_args()
    url='https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
    raw=RemoteZip(url,a.root/'range_cache')
    with zipfile.ZipFile(raw) as z: info={i.filename:i for i in z.infolist()}
    meta=loadmat(a.root/'SUNRGBDMeta3DBB_v2.mat',simplify_cells=True)['SUNRGBDMeta']
    names=sorted(set(Path('SUNRGBD/'+m[k].split('/SUNRGBD/')[1]).as_posix() for m in meta for k in ('depthpath','rgbpath')))
    missing=[n for n in names if n not in info]
    if missing: raise RuntimeError(f'Official metadata paths absent in ZIP: {missing[:10]}')
    local=threading.local(); started=time.time(); completed=[]; errors=[]
    def fetch(name):
        i=info[name]; dest=a.root/name
        if dest.exists() and dest.stat().st_size==i.file_size and zlib.crc32(dest.read_bytes())==i.CRC:
            return {'path':name,'bytes':i.file_size,'crc32':i.CRC,'reused':True}
        if not hasattr(local,'session'):
            local.session=requests.Session()
            local.session.mount('https://',HTTPAdapter(max_retries=Retry(total=8,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
        # Local ZIP headers may contain extra fields beyond the central header.
        start=i.header_offset; end=min(raw.size-1,start+30+len(name.encode())+len(i.extra)+i.compress_size+255)
        r=local.session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(30,180))
        r.raise_for_status()
        if r.status_code!=206 or len(r.content)!=end-start+1: raise RuntimeError('Invalid HTTP range response')
        header=struct.unpack('<4s5H3I2H',r.content[:30])
        if header[0]!=b'PK\x03\x04': raise RuntimeError('Bad ZIP local header')
        offset=30+header[-2]+header[-1]
        payload=r.content[offset:offset+i.compress_size]
        if len(payload)!=i.compress_size: raise RuntimeError('Unexpected large local extra field')
        data=zlib.decompress(payload,-15) if i.compress_type==zipfile.ZIP_DEFLATED else payload
        if len(data)!=i.file_size or zlib.crc32(data)!=i.CRC: raise RuntimeError(f'CRC/length failure {name}')
        dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix(dest.suffix+'.part')
        tmp.write_bytes(data); tmp.replace(dest)
        return {'path':name,'bytes':i.file_size,'crc32':i.CRC,'reused':False}
    receipt=a.root/'full_download_manifest.json'
    with ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(fetch,n):n for n in names}
        for future in as_completed(futures):
            try: completed.append(future.result())
            except Exception as e: errors.append({'path':futures[future],'error':str(e)})
            n=len(completed)+len(errors)
            if n%100==0 or n==len(names):
                status={'source':url,'total':len(names),'completed':len(completed),'failed':len(errors),
                    'elapsed_seconds':round(time.time()-started,1),'errors':errors}
                receipt.with_suffix('.progress.json').write_text(json.dumps(status,indent=2))
                print(json.dumps({k:v for k,v in status.items() if k!='errors'}),flush=True)
    receipt.write_text(json.dumps({'source':url,'files':completed,'errors':errors},indent=2))
    if errors: raise RuntimeError(f'{len(errors)} files failed; rerun to retry only missing/corrupt files')
    print('FULL RAW DATA READY',len(meta),'scenes',flush=True)


if __name__=='__main__': main()
