#!/usr/bin/env python3
"""Pinned benchmark runners; raw output remains private in ignored results/."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from config import ROOT,PINS,CONTAINER,MODELS,settings
import prepare_bench
import smoke


def decode_summary(data,strict_no_renewal=False):
    rows=[]
    for cell in data.get('results',[]):
        c=cell['concurrency']
        issues=[]
        for flag in ('underfilled','warmup_timed_out','capacity_limited','loop_detected'):
            if cell.get(flag) is not False:issues.append(flag)
        if cell.get('num_errors')!=0:issues.append('request_errors')
        if cell.get('aggregate_source')!='openai_continuous_usage':issues.append('missing_continuous_usage')
        for field in ('avg_running_reqs','max_running_reqs','effective_concurrency'):
            if cell.get(field)!=c:issues.append(field)
        for field in ('avg_queue_reqs','max_queue_reqs','queue_fraction'):
            if cell.get(field)!=0:issues.append(field)
        rate=cell.get('aggregate_tps',0)
        server=cell.get('server_gen_throughput',0)
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (rate,server)):
            issues.append('invalid_rate')
        elif abs(rate-server)/server>=0.05:issues.append('client_server_disagreement')
        renewals=cell.get('request_count',0)>c
        if strict_no_renewal and renewals:issues.append('request_renewal')
        rows.append({'context_tokens':cell['context_tokens'],'concurrency':c,'valid':not issues,
            'aggregate_tok_s':None if issues else rate,'issues':sorted(set(issues)),
            'renewing_streams':renewals})
    if not rows:raise RuntimeError('No benchmark cells')
    return {'all_cells_valid':all(row['valid'] for row in rows),'cells':rows}


def pool():
    if os.environ.get('KV_BUDGET'):return int(os.environ['KV_BUDGET'])
    logs=subprocess.run(['docker','logs',CONTAINER],capture_output=True,text=True,check=True)
    matches=re.findall(r'GPU KV cache size: ([\d,]+) tokens',logs.stdout+logs.stderr)
    if not matches:raise RuntimeError('Cannot read the current KV pool; set KV_BUDGET to the actual startup value')
    return int(matches[-1].replace(',',''))


def quality_summary(data,expected):
    summary=data['all_summary']
    counts=('attempted','completed','correct','errors','truncated','stalled','timeout','incomplete','hit_max_tokens')
    result={name:summary[name] for name in counts}
    result['passed']=all(summary[name]==expected for name in ('attempted','completed','correct')) and all(
        summary[name]==0 for name in counts[3:])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','smoke','quick','quality','decode','confirm','sweep'))
    args=parser.parse_args()
    if args.mode=='prepare':prepare_bench.main();return
    prepare_bench.verify()
    config=settings()
    if [row['id'] for row in smoke.request('/v1/models')['data']]!=MODELS:
        raise RuntimeError('Expected model aliases are not served')
    output=ROOT/'results'/(args.mode+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    def bench(name,options):
        command=[sys.executable,'-u',str(ROOT/'tools/bench_wrapper.py'),
            '--host',os.environ.get('HOST','127.0.0.1'),'--port',str(config['port']),
            '--model',MODELS[1],'--display-mode','plain','--no-resume',*options,
            '--output',str(output/(name+'.json'))]
        with (output/(name+'.log')).open('x') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        prepare_bench.verify()
        return json.loads((output/(name+'.json')).read_text())

    results={'benchmark_version':PINS['benchmark_version'],'benchmark_sha256':PINS['benchmark_sha256']}
    if args.mode in ('smoke','quick'):
        results['smoke']=smoke.run()
    if args.mode in ('quality','quick'):
        if args.mode=='quality':
            common=['--profile-concurrency','30','--profile-runs','30','--reasoning-effort','max',
                '--completion-stats-save-text','--completion-stats-seed','20261026',
                '--completion-stats-request-timeout','900','--completion-stats-stall-timeout','600']
            cases=[('lavd30','lavd-test','100000'),('estonia30','estonia','40000')]
            expected=30
        else:
            common=['--profile-concurrency','10','--profile-runs','10',
                '--completion-stats-temperature','1','--completion-stats-top-p','1',
                '--completion-stats-seed','20261018']
            cases=[('estonia10','estonia','40000')]
            expected=10
        print('Raw quality reports may contain benchmark personal details. Keep results/ private.',flush=True)
        for name,profile,maximum in cases:
            data=bench(name,['--test-profile',profile,'--max-tokens',maximum,*common])
            results[name]=quality_summary(data,expected)
            if not results[name]['passed']:
                (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
                raise RuntimeError('Quality gate failed; raw report retained privately')
    if args.mode in ('decode','confirm','sweep'):
        if args.mode=='sweep':
            options=['--concurrency','1,2,4,8,16,32','--contexts','0,16k,32k,64k,128k',
                '--duration','30','--max-tokens','8192','--decode-warmup-seconds','3','--kv-budget',str(pool())]
        else:
            options=['--concurrency','1,4,16,32','--contexts','0','--duration','60' if args.mode=='confirm' else '30',
                '--max-tokens','65536' if args.mode=='confirm' else '32768','--decode-warmup-seconds','3',
                '--skip-prefill','--no-hw-monitor']
        data=bench(args.mode,options)
        results['decode']=decode_summary(data,strict_no_renewal=args.mode!='sweep')
    (output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    print('Private reports:',output)
    if 'decode' in results and not results['decode']['all_cells_valid']:
        raise SystemExit('At least one cell was rejected; no rate is claimed for invalid cells')


if __name__=='__main__':main()
