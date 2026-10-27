"""Public recipe constants; machine-specific values come only from local .env."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PINS=json.loads((ROOT/'pins.json').read_text())
PROJECT='deepseek-v41-brrr-b12x-hybrid'
CONTAINER=PROJECT
MODELS=['DeepSeek-4.1-Flash-1017-BRRR','deepseek-v4.1-flash']
STATE=ROOT/'.state'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024*1024):result.update(block)
    return result.hexdigest()


def run(command,**kwargs):
    return subprocess.run(command,check=True,**kwargs)


def inspect(kind,name):
    result=subprocess.run(['docker',kind,'inspect',name],capture_output=True,text=True)
    return json.loads(result.stdout)[0] if result.returncode==0 else None


def settings():
    for name in ('MODEL_DIR','RUNTIME_CACHE_DIR'):
        if not os.environ.get(name):raise RuntimeError(f'Set {name} in .env first')
    port=int(os.environ.get('HOST_PORT','8000'))
    if not 1<=port<=65535:raise RuntimeError('HOST_PORT must be between 1 and 65535')
    gpus=[part.strip() for part in os.environ.get('GPU_IDS','0,1,2,3').split(',')]
    if len(gpus)!=4 or len(set(gpus))!=4 or not all(part.isdigit() for part in gpus):
        raise RuntimeError('GPU_IDS must contain four distinct nvidia-smi indices')
    model=Path(os.environ['MODEL_DIR']).expanduser()
    cache=Path(os.environ['RUNTIME_CACHE_DIR']).expanduser()
    if not model.is_absolute() or not cache.is_absolute():raise RuntimeError('Model/cache paths must be absolute')
    if model.resolve().is_relative_to(cache.resolve()) or cache.resolve().is_relative_to(model.resolve()):
        raise RuntimeError('Model and writable runtime-cache directories must not overlap')
    return dict(port=port,bind=os.environ.get('BIND_ADDRESS','127.0.0.1'),gpus=gpus,
                model=model.resolve(),cache=cache.resolve())
