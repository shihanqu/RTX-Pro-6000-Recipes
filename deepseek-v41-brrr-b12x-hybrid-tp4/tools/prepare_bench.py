#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

from config import ROOT,PINS,digest,run


def location():
    path=Path(os.environ.get('BENCH_DIR',str(ROOT/'.bench'))).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def verify():
    directory=location()
    source=directory/'llm_decode_bench.py'
    if not source.is_file() or digest(source)!=PINS['benchmark_sha256']:
        raise RuntimeError('Benchmark bytes differ; run validate.sh prepare with a clean BENCH_DIR')
    commit=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if commit!=PINS['benchmark_commit']:raise RuntimeError('Benchmark checkout is not the pinned commit')
    return source


def main():
    directory=location()
    if directory.exists() and any(directory.iterdir()):
        verify()
        print('Existing pinned benchmark verified; no checkout or edits performed')
        return
    directory.mkdir(parents=True,exist_ok=True)
    run(['git','init',str(directory)])
    run(['git','-C',str(directory),'remote','add','origin',PINS['benchmark_repo']])
    run(['git','-C',str(directory),'fetch','--depth','1','origin',PINS['benchmark_commit']])
    run(['git','-C',str(directory),'checkout','--detach','FETCH_HEAD'])
    verify()
    print('Exact benchmark commit and source checksum verified')


if __name__=='__main__':main()
