"""Resumable coalesced block download of required official ZIP data (few requests)."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import io
import json
from pathlib import Path
import threading
import time
import zipfile
import zlib
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.io import loadmat
from download_sunrgbd import RemoteZip


class BlockReader(io.RawIOBase):
    def __init__(self,url,root,size,block_size=8*1024**2):
        self.url=url; self.root=root; root.mkdir(exist_ok=True); self.size=size; self.block_size=block_size; self.pos=0
        self.local=threading.local()
    def get(self,i):
        start=i*self.block_size; end=min(start+self.block_size,self.size)-1
        path=self.root/f'{i:06d}.bin'
        if path.exists() and path.stat().st_size==end-start+1: return path
        if not hasattr(self.local,'session'):
            self.local.session=requests.Session()
            self.local.session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=2,status_forcelist=[429,500,502,503,504])))
        tmp=path.with_suffix('.part')
        result=subprocess.run(['curl','--fail','--silent','--show-error','--location','--retry','5',
            '--connect-timeout','20','--max-time','180','--range',f'{start}-{end}',
            '--output',str(tmp),'--write-out','%{http_code}',self.url],capture_output=True,text=True)
        if result.returncode or result.stdout.strip()!='206' or not tmp.exists() or tmp.stat().st_size!=end-start+1:
            raise RuntimeError(f'Block {i}: {result.stderr[-300:]} HTTP {result.stdout}')
        tmp.replace(path); return path
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset+(0 if whence==0 else self.pos if whence==1 else self.size); return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size,self.size-self.pos); chunks=[]
        while n>0:
            i,offset=divmod(self.pos,self.block_size); count=min(n,self.block_size-offset)
            with self.get(i).open('rb') as f: f.seek(offset); chunks.append(f.read(count))
            self.pos+=count; n-=count
        return b''.join(chunks)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--workers',type=int,default=4); a=p.parse_args()
    url='https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
    # Values verified by the official HTTP HEAD in this audit; cached central directory
    # uses this ETag. Every extracted member is still checked against its ZIP CRC.
    remote=RemoteZip(url,a.root/'range_cache',known_size=6885481608,known_etag='"19a681c88-51e45f39c4b5d"')
    with zipfile.ZipFile(remote) as z: entries={i.filename:i for i in z.infolist()}
    meta=loadmat(a.root/'SUNRGBDMeta3DBB_v2.mat',simplify_cells=True)['SUNRGBDMeta']
    names=sorted({Path('SUNRGBD/'+m[k].split('/SUNRGBD/')[1]).as_posix() for m in meta for k in ('depthpath','rgbpath')})
    reader=BlockReader(url,a.root/'zip_blocks_8MiB',remote.size); needed=set(); pending=[]
    for name in names:
        info=entries[name]; path=a.root/name
        if path.exists() and path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())==info.CRC: continue
        pending.append(name)
        start=info.header_offset; end=min(remote.size-1,start+info.compress_size+len(name.encode())+65536)
        needed.update(range(start//reader.block_size,end//reader.block_size+1))
    started=time.time(); errors=[]
    print('Existing files reused',len(names)-len(pending),'pending',len(pending),'blocks',len(needed),flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(reader.get,i):i for i in sorted(needed)}
        for n,f in enumerate(as_completed(futures),1):
            try: f.result()
            except Exception as e: errors.append({'block':futures[f],'error':str(e)})
            if n%10==0 or n==len(needed):
                status={'completed_blocks':n-len(errors),'total_blocks':len(needed),'errors':errors,'seconds':round(time.time()-started,1)}
                (a.root/'block_download_progress.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status),flush=True)
    if errors: raise RuntimeError('Incomplete blocks; rerun resumes existing blocks')
    with zipfile.ZipFile(reader) as z:
        for n,name in enumerate(pending,1):
            content=z.read(name); path=a.root/name; path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(path.suffix+'.part'); tmp.write_bytes(content); tmp.replace(path)
            if n%1000==0: print('EXTRACTED',n,'/',len(pending),flush=True)
    receipt={'source':url,'scenes':len(meta),'files':len(names),'reused_files':len(names)-len(pending),'CRC_verified':True}
    (a.root/'full_data_ready.json').write_text(json.dumps(receipt,indent=2)); print('FULL RAW DATA READY',receipt,flush=True)


if __name__=='__main__': main()
