"""Fetch only required RGB/depth payloads using official HTTP multipart byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import re
import struct
import subprocess
import time
import zipfile
import zlib
from scipy.io import loadmat
from download_sunrgbd import RemoteZip


def unpack_member(info,segment,start):
    offset=info.header_offset-start
    header=struct.unpack('<4s5H3I2H',segment[offset:offset+30])
    if header[0]!=b'PK\x03\x04': raise RuntimeError('Bad local header')
    offset+=30+header[-2]+header[-1]
    payload=segment[offset:offset+info.compress_size]
    if len(payload)!=info.compress_size: raise RuntimeError('Truncated compressed data')
    data=zlib.decompress(payload,-15) if info.compress_type==zipfile.ZIP_DEFLATED else payload
    if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC: raise RuntimeError('Bad member CRC')
    return data


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path.home()/'dabaset/SUNRGBD')
    p.add_argument('--workers',type=int,default=24); a=p.parse_args()
    url='https://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
    remote=RemoteZip(url,a.root/'range_cache',6885481608,'"19a681c88-51e45f39c4b5d"')
    with zipfile.ZipFile(remote) as z: entries={i.filename:i for i in z.infolist()}
    meta=loadmat(a.root/'SUNRGBDMeta3DBB_v2.mat',simplify_cells=True)['SUNRGBDMeta']
    names={Path('SUNRGBD/'+m[k].split('/SUNRGBD/')[1]).as_posix() for m in meta for k in ('depthpath','rgbpath')}
    cache=a.root/'multipart_cache'; cache.mkdir(exist_ok=True)
    def commit(info,data):
        dest=a.root/info.filename; dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+'.part'); temp.write_bytes(data); temp.replace(dest)
    pending=[]; reused=0; block_size=8*1024**2
    for info in sorted((entries[n] for n in names),key=lambda i:i.header_offset):
        path=a.root/info.filename
        if path.exists() and path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())==info.CRC:
            reused+=1; continue
        start=info.header_offset; end=min(remote.size-1,start+info.compress_size+30+len(info.filename.encode())+len(info.extra)+255)
        blocks=range(start//block_size,end//block_size+1)
        paths=[a.root/'zip_blocks_8MiB'/f'{i:06d}.bin' for i in blocks]
        if all(p.exists() for p in paths):
            data=b''.join(p.read_bytes() for p in paths)
            commit(info,unpack_member(info,data,(start//block_size)*block_size)); reused+=1; continue
        pending.append((info,start,end))
    groups=[]; group=[]; size=0
    for item in pending:
        if len(group)>=64 or size+item[2]-item[1]+1>8*1024**2:
            groups.append(group); group=[]; size=0
        group.append(item); size+=item[2]-item[1]+1
    if group: groups.append(group)
    print('Reused',reused,'pending files',len(pending),'groups',len(groups),'workers',a.workers,flush=True)
    def fetch(item):
        index,group=item; body=cache/f'{index:05d}.body'; headers=cache/f'{index:05d}.headers'
        # Merge overlapping requested ranges so Apache cannot reject duplicate spans.
        spans=[]
        for _,lo,hi in group:
            if spans and lo<=spans[-1][1]+1: spans[-1][1]=max(spans[-1][1],hi)
            else: spans.append([lo,hi])
        ranges=','.join(f'{lo}-{hi}' for lo,hi in spans)
        error=''
        for attempt in range(5):
            r=subprocess.run(['/usr/bin/curl','-fLSs','--noproxy','*','--connect-timeout','20','--max-time','240',
                '--range',ranges,url,'--dump-header',str(headers),'--output',str(body),'--write-out','%{http_code}'],capture_output=True,text=True)
            try:
                if r.returncode or r.stdout.strip()!='206': raise RuntimeError(r.stderr[-300:])
                h=headers.read_text(); content=body.read_bytes(); segments=[]
                boundary=re.findall(r'boundary=([^\s;]+)',h,re.I)
                if boundary:
                    for part in content.split(b'--'+boundary[-1].strip('"').encode())[1:]:
                        if b'\r\n\r\n' not in part: continue
                        header,payload=part.split(b'\r\n\r\n',1)
                        match=re.search(rb'Content-Range:\s*bytes\s+(\d+)-(\d+)/',header,re.I)
                        if match:
                            lo,hi=map(int,match.groups()); segments.append((lo,hi,payload[:hi-lo+1]))
                else:
                    match=re.findall(r'Content-Range:\s*bytes\s+(\d+)-(\d+)/',h,re.I)
                    if match:
                        lo,hi=map(int,match[-1]); segments=[(lo,hi,content)]
                decoded=[]
                for info,start,end in group:
                    segment=next((s for s in segments if s[0]<=start and s[1]>=end),None)
                    if segment is None: raise RuntimeError('Missing multipart range')
                    decoded.append((info,unpack_member(info,segment[2],segment[0])))
                for info,data in decoded: commit(info,data)
                body.unlink(missing_ok=True); headers.unlink(missing_ok=True)
                return len(group)
            except Exception as e:
                error=str(e); time.sleep(min(2**attempt,16))
        raise RuntimeError(f'group {index}: {error}')
    completed=reused; errors=[]; started=time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        futures={pool.submit(fetch,(i,g)):i for i,g in enumerate(groups)}
        for future in as_completed(futures):
            try: completed+=future.result()
            except Exception as e: errors.append(str(e))
            status={'completed_files':completed,'total_files':len(names),'elapsed_seconds':round(time.time()-started,1),'errors':errors}
            (a.root/'multipart_progress.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status),flush=True)
    if errors: raise RuntimeError('Some groups incomplete; rerun reuses every verified file')
    receipt={'source':url,'scenes':len(meta),'files':completed,'CRC_verified':True,'download_strategy':'official multipart byte ranges'}
    (a.root/'full_data_ready.json').write_text(json.dumps(receipt,indent=2)); print('FULL RAW DATA READY',flush=True)


if __name__=='__main__': main()
