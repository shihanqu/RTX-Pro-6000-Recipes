#!/usr/bin/env python3
"""Read-only host checks and an idempotent launcher; never kill another service."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from config import ROOT,PINS,STATE,PROJECT,CONTAINER,MODELS,settings,inspect,run
from fetch import manifest


def gpu_records():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version',
                                  '--format=csv,noheader,nounits'],text=True)
    return {row[0].strip():[part.strip() for part in row] for row in csv.reader(text.splitlines())}


def select_gpus(config):
    records=gpu_records()
    selected=[]
    for index in config['gpus']:
        if index not in records:raise RuntimeError('Selected GPU index not found')
        _,uuid,name,memory,driver=records[index]
        if 'RTX PRO 6000' not in name or 'Blackwell' not in name or int(memory)<95000:
            raise RuntimeError('This preset requires four RTX PRO 6000 Blackwell 96 GB GPUs')
        if tuple(int(part) for part in driver.split('.'))<(615,65):
            raise RuntimeError('Driver 615.65 or newer is required by the pinned CUDA 13.4 image')
        selected.append(uuid)
    return selected


def host_checks():
    groups=Path('/sys/kernel/iommu_groups')
    if not groups.is_dir() or any(groups.iterdir()):
        raise RuntimeError('The measured profile requires IOMMU disabled; see host setup and security warnings')
    hmm=Path('/sys/module/nvidia_uvm/parameters/uvm_disable_hmm')
    if not hmm.is_file() or hmm.read_text().strip()!='Y':
        raise RuntimeError('Expected uvm_disable_hmm=1; load nvidia_uvm after the documented host setup/reboot')
    params=Path('/proc/driver/nvidia/params').read_text()
    for value in ('ForceP2P=0x11','RMForceP2PType=1','RMPcieP2PType=2',
                  'GrdmaPciTopoCheckOverride=1','EnableResizableBar=1'):
        if value not in params:raise RuntimeError('Measured driver option missing: '+value)


def checkpoint_checks(config):
    expected=manifest('model')
    receipt_path=config['model']/'recipe-verification.json'
    if not receipt_path.is_file():raise RuntimeError('Run download_checkpoint.sh (or --verify) first')
    receipt=json.loads(receipt_path.read_text())
    for key in ('repo','revision','file_count','total_bytes','files'):
        if receipt.get(key)!=expected[key]:raise RuntimeError('Checkpoint verification receipt does not match the pinned manifest')
    for item in expected['files']:
        path=config['model']/item['file']
        if not path.is_file() or path.stat().st_size!=item['bytes']:
            raise RuntimeError('Missing/wrong-size checkpoint file: '+item['file'])
        if not path.resolve().is_relative_to(config['model']):
            raise RuntimeError('External checkpoint symlinks would break the /model bind; use a real-file local-dir download')
    if any(config['model'].rglob('*.incomplete')):raise RuntimeError('Checkpoint still has incomplete downloads')
    for number in (47,48):
        path=config['model']/f'model-{number:05d}-of-00048.safetensors'
        kind=subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(path.resolve())],text=True).strip()
        if kind not in ('ext4','xfs'):raise RuntimeError('Disk Engram requires a local ext4/XFS filesystem')


def image_checks():
    path=STATE/'build.json'
    if not path.is_file():raise RuntimeError('Run build.sh first; a verified local image receipt is required')
    receipt=json.loads(path.read_text())
    image=inspect('image',PINS['hybrid_tag'])
    if not image or image['Id']!=receipt.get('image_id'):raise RuntimeError('Image tag changed since build verification')
    for key in ('base_image','brrr_revision'):
        if receipt.get(key)!=PINS[key]:raise RuntimeError('Build receipt provenance mismatch')
    if receipt.get('decoder_sha256')!=PINS['hybrid_decoder_sha256']:raise RuntimeError('Wrong dispatch patch')
    return image


def compose(config,image,uuids,*args,capture=False):
    environment={**os.environ,'MODEL_DIR':str(config['model']),'RUNTIME_CACHE_DIR':str(config['cache']),
        'HYBRID_IMAGE':image['Id'],'GPU_UUIDS':','.join(uuids)}
    return run(['docker','compose','-p',PROJECT,'-f',str(ROOT/'compose.yaml'),
                '--env-file',str(ROOT/'.env'),*args],env=environment,
                **({'capture_output':True,'text':True} if capture else {}))


def assert_idle(uuids,port):
    if subprocess.check_output(['ss','-H','-ltn',f'sport = :{port}'],text=True).strip():
        raise RuntimeError('Host port is occupied; no service will be stopped')
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader,nounits'],text=True)
    if any(row and row[0].strip() in uuids for row in csv.reader(text.splitlines())):
        raise RuntimeError('A selected GPU has compute processes; no process will be killed')


def verify_running(info,config,image,uuids):
    if info['Image']!=image['Id'] or not info['State']['Running'] or info['State']['OOMKilled']:
        raise RuntimeError('Existing container is not the expected running image')
    if info['Config']['Labels'].get('com.docker.compose.project')!=PROJECT:
        raise RuntimeError('Container name belongs to another project')
    resolved=json.loads(compose(config,image,uuids,'config','--format','json',capture=True).stdout)['services']['deepseek']
    if info['Config']['Cmd']!=resolved['command']:raise RuntimeError('Running arguments differ from this preset')
    env=dict(item.split('=',1) for item in info['Config']['Env'] if '=' in item)
    expected=dict(item.split('=',1) for item in image['Config']['Env'] if '=' in item)
    expected.update({key:str(value) for key,value in resolved['environment'].items()})
    if env!=expected:raise RuntimeError('Running environment differs from the pinned preset')
    binds=info['HostConfig']['PortBindings'].get('8000/tcp',[])
    if not binds or binds[0]['HostPort']!=str(config['port']) or binds[0]['HostIp']!=config['bind']:
        raise RuntimeError('Running port/bind address differs')
    mounts={row['Destination']:row for row in info['Mounts']}
    if set(mounts)!={'/model','/cache/jit'} or mounts['/model']['Source']!=str(config['model']) or mounts['/model']['RW']:
        raise RuntimeError('Unexpected model/cache mount')
    if mounts['/cache/jit']['Source']!=str(config['cache']):raise RuntimeError('Runtime cache path changed')
    if info['HostConfig']['RestartPolicy']['Name']!='unless-stopped':raise RuntimeError('Unexpected restart policy')


def wait_ready(config,image,uuids):
    host='127.0.0.1' if config['bind'] in ('0.0.0.0','127.0.0.1') else config['bind']
    deadline=time.monotonic()+int(os.environ.get('READY_TIMEOUT_S','5400'))
    last=0
    while time.monotonic()<deadline:
        info=inspect('container',CONTAINER)
        if not info or not info['State']['Running']:raise RuntimeError('Server stopped; inspect launch.sh logs')
        try:
            with urllib.request.urlopen(f'http://{host}:{config["port"]}/health',timeout=5) as response:
                if response.status!=200:raise RuntimeError('Not ready')
            with urllib.request.urlopen(f'http://{host}:{config["port"]}/v1/models',timeout=5) as response:
                models=json.load(response)
            if [row['id'] for row in models['data']]!=MODELS:raise RuntimeError('API aliases differ')
            verify_running(info,config,image,uuids)
            logs=subprocess.run(['docker','logs',CONTAINER],capture_output=True,text=True,check=True)
            text=logs.stdout+logs.stderr
            (STATE/'startup.log').write_text(text)
            if 'Overriding to FULL_DECODE_ONLY' in text:raise RuntimeError('Unexpected graph fallback')
            if not all(marker in text for marker in ('Breakable CUDA graph enabled',
                    'Capturing CUDA graphs (PIECEWISE)','Capturing CUDA graphs (FULL)')):
                raise RuntimeError('Expected full/piecewise graph capture not found')
            pool=re.findall(r'GPU KV cache size: ([\d,]+) tokens',text)
            print(f'Ready: http://127.0.0.1:{config["port"]}/v1; both model aliases verified')
            if pool:print('GPU KV cache pool:',pool[-1],'tokens (shared, not per stream)')
            return
        except (OSError,ValueError):pass
        if time.monotonic()-last>30:
            print('Waiting for model load, kernel preparation and CUDA graphs...',flush=True)
            last=time.monotonic()
        time.sleep(5)
    raise RuntimeError('Readiness timed out; container retained for diagnosis, no other service changed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('start','check','config','logs','stop'),default='start',nargs='?')
    args=parser.parse_args()
    STATE.mkdir(exist_ok=True)
    with (STATE/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action in ('stop','logs'):
            info=inspect('container',CONTAINER)
            if not info or info['Config']['Labels'].get('com.docker.compose.project')!=PROJECT:
                raise RuntimeError('No owned recipe container; refusing operation')
            run(['docker','stop','--timeout','120',CONTAINER] if args.action=='stop' else ['docker','logs','-f',CONTAINER])
            return
        config=settings()
        image=image_checks()
        if args.action=='config':
            compose(config,image,config['gpus'],'config')
            return
        uuids=select_gpus(config)
        host_checks()
        checkpoint_checks(config)
        info=inspect('container',CONTAINER)
        if info and info['State']['Running']:
            verify_running(info,config,image,uuids)
            if args.action=='check':print('Matching recipe container is already running')
            else:wait_ready(config,image,uuids)
            return
        if info and info['Config']['Labels'].get('com.docker.compose.project')!=PROJECT:
            raise RuntimeError('Container name belongs to another project; refusing replacement')
        assert_idle(uuids,config['port'])
        probe='import torch; assert torch.cuda.device_count()==4; assert all(torch.cuda.can_device_access_peer(i,j) for i in range(4) for j in range(4) if i!=j); print("All twelve directed CUDA peer-access pairs available")'
        run(['docker','run','--rm','--gpus','all','-e','CUDA_VISIBLE_DEVICES='+','.join(uuids),
             '--entrypoint','/opt/venv/bin/python',image['Id'],'-c',probe])
        assert_idle(uuids,config['port'])
        if args.action=='check':
            print('Read-only host/image/checkpoint/device checks passed; power limits unchanged')
            return
        config['cache'].mkdir(parents=True,exist_ok=True)
        compose(config,image,uuids,'up','-d','--no-build')
        wait_ready(config,image,uuids)


if __name__=='__main__':main()
