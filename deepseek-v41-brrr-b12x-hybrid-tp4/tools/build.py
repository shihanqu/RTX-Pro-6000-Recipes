#!/usr/bin/env python3
"""Build pinned BRRR with bounded compilation, then add the measured runtime changes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from config import ROOT,PINS,STATE,RUNTIME_FILE_SHA256,GIB,available_memory,inspect,run,digest
from fetch import fetch


def serialize_prebuild(dockerfile):
    before='RUN ["/opt/venv/bin/lil-runtime-bootstrap", "/opt/venv/bin/python", "-m", "brrr.prebuild"]'
    if dockerfile.count(before)!=1:
        raise RuntimeError('Unexpected BRRR Dockerfile prebuild command')
    # Keep the source and flags intact; only serialize compilation on cache misses.
    after='RUN --network=none '+json.dumps(['/usr/bin/env','MAX_JOBS=1',
        '/opt/venv/bin/lil-runtime-bootstrap','/opt/venv/bin/python','-c',
        'from brrr import cuext; tuple(map(cuext.extension, cuext.SOURCES))'])
    return dockerfile.replace(before,after,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-source',action='store_true',help='Use an already downloaded source tree')
    args=parser.parse_args()
    if available_memory()<8*GIB:
        raise RuntimeError('Build requires at least 8 GiB available RAM; build before starting the RAM model')
    source=ROOT/'.build/upstream'
    manifest=fetch('source',source,args.verify_source)
    environment={**os.environ,'DOCKER_BUILDKIT':'1'}
    with tempfile.TemporaryDirectory(prefix='verified-context-',dir=ROOT/'.build') as directory:
        context=Path(directory)
        # A fresh allowlisted context excludes stale, unverified build files.
        for item in manifest['files']:
            target=context/item['file']
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/item['file'],target)
        dockerfile=context/'Dockerfile'
        dockerfile.write_text(serialize_prebuild(dockerfile.read_text()))
        run(['docker','build','--build-arg','BASE_IMAGE='+PINS['base_image'],
            '--build-arg','UV_IMAGE='+PINS['uv_image'],
            '-t',PINS['upstream_tag'],str(context)],env=environment)
    upstream=inspect('image',PINS['upstream_tag'])
    if not upstream:raise RuntimeError('Upstream build missing')
    run(['docker','build','--build-arg','BRRR_IMAGE='+PINS['upstream_tag'],
        '-f',str(ROOT/'Dockerfile'),'-t',PINS['hybrid_tag'],str(ROOT)],env=environment)
    image=inspect('image',PINS['hybrid_tag'])
    if not image:raise RuntimeError('Hybrid build missing')
    layers=upstream['RootFS']['Layers']
    if image['RootFS']['Layers'][:len(layers)]!=layers:
        raise RuntimeError('Hybrid image is not based on the verified upstream build')
    sums=subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint',
        'sha256sum',image['Id'],*RUNTIME_FILE_SHA256],text=True)
    actual={path:checksum for checksum,path in (line.split(maxsplit=1) for line in sums.splitlines())}
    if actual!=RUNTIME_FILE_SHA256:raise RuntimeError('Image runtime source verification failed')
    STATE.mkdir(exist_ok=True)
    receipt={'image_id':image['Id'],'image_tag':PINS['hybrid_tag'],'upstream_image_id':upstream['Id'],
        'base_image':PINS['base_image'],'brrr_revision':PINS['brrr_revision'],
        'runtime_file_sha256':actual,'patch_sha256':{
            path.name:digest(path) for path in sorted((ROOT/'patches').glob('*.patch'))}}
    (STATE/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Built and verified the RAM-Engram hybrid. Local build receipt:',STATE/'build.json')


if __name__=='__main__':main()
