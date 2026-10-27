#!/usr/bin/env python3
"""Apply the exact three-line dispatch change, with before/after hash gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def apply(directory,patch,pins):
    path=directory/'decoder.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pins['original_decoder_sha256']:
        raise RuntimeError('Unexpected upstream decoder; refusing to patch')
    subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-d',str(directory),'-i',str(patch)],check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pins['hybrid_decoder_sha256']:
        raise RuntimeError('Patched decoder does not match the validated implementation')


if __name__=='__main__':
    apply(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
