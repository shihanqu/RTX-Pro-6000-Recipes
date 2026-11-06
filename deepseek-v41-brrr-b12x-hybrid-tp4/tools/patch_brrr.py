#!/usr/bin/env python3
"""Apply the measured BRRR changes with exact before/after hash gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def apply(directory,patch,filename,before,after):
    path=directory/filename
    if hashlib.sha256(path.read_bytes()).hexdigest()!=before:
        raise RuntimeError(f'Unexpected upstream {filename}; refusing to patch')
    subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-d',str(directory),'-i',str(patch)],check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=after:
        raise RuntimeError(f'Patched {filename} does not match the measured implementation')


def apply_recipe(directory,patches,pins):
    apply(directory,patches/'hybrid-dispatch.patch','decoder.py',
          pins['original_decoder_sha256'],pins['hybrid_decoder_sha256'])
    apply(directory,patches/'ram-engram.patch','engram_nvme.py',
          pins['original_engram_sha256'],pins['ram_engram_sha256'])


if __name__=='__main__':
    apply_recipe(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
