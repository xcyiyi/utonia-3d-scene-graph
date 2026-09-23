"""Selective, cached HTTP-range access to official SUN RGB-D ZIP archives.

Only fetch files needed by the requested sample IDs; ZIP CRCs are verified.
Completed byte ranges are reusable after interruption. No third-party data mirror.
"""
import argparse
import io
import json
from pathlib import Path
import hashlib
import zipfile
import zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RemoteZip(io.RawIOBase):
    def __init__(self, url, cache, known_size=None, known_etag=None):
        self.url, self.cache, self.pos = url, Path(cache), 0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        if known_size is not None:
            self.size,self.etag=known_size,known_etag
        else:
            r = self.session.head(url, timeout=60)
            r.raise_for_status()
            self.size = int(r.headers['Content-Length'])
            self.etag = r.headers.get('ETag', '')

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size)
        if n <= 0: return b''
        start, end = self.pos, self.pos + n - 1
        key = hashlib.sha256(f'{self.url}:{self.etag}:{start}:{end}'.encode()).hexdigest()
        path = self.cache / key
        if not path.exists():
            r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'}, timeout=120)
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != n:
                raise RuntimeError(f'Server did not honor range: {r.status_code}, {len(r.content)} != {n}')
            tmp = path.with_suffix('.part')
            tmp.write_bytes(r.content)
            tmp.replace(path)
        data = path.read_bytes()
        if len(data) != n: raise RuntimeError(f'Bad cached range: {path}')
        self.pos += n
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.home() / 'dabaset/SUNRGBD')
    parser.add_argument('--toolbox', action='store_true')
    parser.add_argument('--ids', nargs='+', type=int)
    parser.add_argument('--ids-file',type=Path,help='JSON with train/val sample ID lists')
    args = parser.parse_args()
    if args.ids_file:
        splits=json.loads(args.ids_file.read_text())
        args.ids=[sid for ids in splits.values() for sid in ids]
    if not args.toolbox and not args.ids: parser.error('Provide --ids or --ids-file')
    archive = 'SUNRGBDtoolbox.zip' if args.toolbox else 'SUNRGBD.zip'
    url = 'https://rgbd.cs.princeton.edu/data/' + archive
    with zipfile.ZipFile(RemoteZip(url, args.root / 'range_cache')) as z:
        names = z.namelist()
        if args.toolbox:
            wanted = [n for n in names if not n.startswith('__MACOSX/') and n.endswith(('/allsplit.mat', '/read3dPoints.m', '/read_3d_pts_general.m', '/SUNRGBDMeta.mat'))]
            print('Toolbox candidates:', [n for n in names if 'read3d' in n or 'allsplit' in n])
        else:
            from scipy.io import loadmat
            meta = loadmat(args.root / 'SUNRGBDMeta3DBB_v2.mat', simplify_cells=True)['SUNRGBDMeta']
            wanted = []
            for sid in args.ids:
                m = meta[sid - 1]
                for key in ('depthpath', 'rgbpath'):
                    name = 'SUNRGBD/' + m[key].split('/SUNRGBD/')[1]
                    # Follow the depthpath used by the official read3dPoints.m.
                    if name not in names: raise FileNotFoundError(name)
                    wanted.append(name)
        manifest = []
        for name in wanted:
            path = args.root / name
            path.resolve().relative_to(args.root.resolve())
            info = z.getinfo(name)
            if not path.exists() or path.stat().st_size != info.file_size or zlib.crc32(path.read_bytes()) != info.CRC:
                data = z.read(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary=path.with_suffix(path.suffix+'.part')
                temporary.write_bytes(data)
                temporary.replace(path)
            manifest.append({'path': name, 'source': url, 'crc32': info.CRC,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print('Ready:', name, path.stat().st_size, flush=True)
        dest = args.root / ('toolbox_manifest.json' if args.toolbox else 'samples_manifest.json')
        previous = json.loads(dest.read_text()) if dest.exists() else []
        by_path = {m['path']: m for m in previous + manifest}
        dest.write_text(json.dumps(list(by_path.values()), indent=2))


if __name__ == '__main__': main()
