#!/usr/bin/env python3
"""Build pinned BRRR unchanged, then add only the validated dispatch patch."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from config import ROOT,PINS,STATE,inspect,run,digest
from fetch import fetch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-source',action='store_true',help='Use an already downloaded source tree')
    args=parser.parse_args()
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
    actual=subprocess.check_output(['docker','run','--rm','--entrypoint','sha256sum',image['Id'],
        '/opt/venv/lib/python3.12/site-packages/brrr/decoder.py'],text=True).split()[0]
    if actual!=PINS['hybrid_decoder_sha256']:raise RuntimeError('Image decoder verification failed')
    STATE.mkdir(exist_ok=True)
    receipt={'image_id':image['Id'],'image_tag':PINS['hybrid_tag'],'upstream_image_id':upstream['Id'],
        'base_image':PINS['base_image'],'brrr_revision':PINS['brrr_revision'],
        'decoder_sha256':actual,'patch_sha256':digest(ROOT/'patches/hybrid-dispatch.patch')}
    (STATE/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Built and verified the hybrid. Local build receipt:',STATE/'build.json')


if __name__=='__main__':main()
