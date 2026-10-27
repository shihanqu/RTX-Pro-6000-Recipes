#!/usr/bin/env python3
"""Fetch from the original publisher; verify every byte against a public manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

from config import ROOT


def manifest(kind):
    data=json.loads((ROOT/'manifests'/f'{kind}.json').read_text())
    for item in data['files']:
        path=PurePosixPath(item['file'])
        if path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe manifest path')
    if len(data['files'])!=data['file_count'] or sum(item['bytes'] for item in data['files'])!=data['total_bytes']:
        raise RuntimeError('Manifest count/size mismatch')
    return data


def verify_file(directory,item):
    path=directory/item['file']
    if not path.is_file() or path.stat().st_size!=item['bytes']:
        raise RuntimeError(f'Missing/wrong-size file: {item["file"]}')
    if item['algorithm']=='sha256':result=hashlib.sha256()
    elif item['algorithm']=='git-blob-sha1':
        result=hashlib.sha1()
        result.update(f'blob {item["bytes"]}\0'.encode())
    else:raise RuntimeError('Unsupported manifest hash')
    with path.open('rb') as stream:
        while block:=stream.read(8*1024*1024):result.update(block)
    if result.hexdigest()!=item['hash']:raise RuntimeError(f'Hash mismatch: {item["file"]}')
    print('Verified',item['file'],flush=True)
    return item


def fetch(kind,directory,verify_only=False):
    data=manifest(kind)
    if not verify_only:
        from huggingface_hub import snapshot_download
        directory.mkdir(parents=True,exist_ok=True)
        snapshot_download(data['repo'],revision=data['revision'],local_dir=str(directory),
                          allow_patterns=[item['file'] for item in data['files']],max_workers=4,
                          token=False if kind=='source' else None)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item:verify_file(directory,item),data['files']))
    receipt={key:data[key] for key in ('repo','revision','file_count','total_bytes','files')}
    receipt['verified_at_unix']=time.time()
    path=directory/'recipe-verification.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt,indent=2)+'\n')
    temp.replace(path)
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=('model','source'))
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.kind=='model':
        if not os.environ.get('MODEL_DIR'):parser.error('Set MODEL_DIR in .env')
        directory=Path(os.environ['MODEL_DIR']).expanduser().resolve()
    else:directory=ROOT/'.build/upstream'
    data=fetch(args.kind,directory,args.verify)
    print(f'Complete: {data["file_count"]} verified files, {data["total_bytes"]:,} bytes')


if __name__=='__main__':main()
