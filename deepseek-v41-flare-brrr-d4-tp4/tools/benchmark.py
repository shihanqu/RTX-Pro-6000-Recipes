#!/usr/bin/env python3
"""Run the pinned sustained-decode matrix plus ten clean C1 windows per context."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

REV = '52216d70865a015b34689626d04a80b89c9dc45f'
SHA = 'd15646fcf09993597b75887c134d812721f4ccad6cdc846bea299cceae1f8a90'
CLIENT_URL = f'https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/{REV}/llm_decode_bench.py'
MODEL = 'DeepSeek-4.1-Flash-1017-BRRR'
CONTEXTS = (0, 16384, 32768, 65536, 131072, 262144)
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
IDLE = ('flare_active_requests', 'flare_pending_requests', 'flare_retiring_request_slots',
        'flare_http_inflight_requests', 'flare_frontend_active_workers')
COUNTERS = ('flare_generated_tokens_total', 'flare_requests_admitted_total', 'flare_requests_completed_total')


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def metrics(base):
    selected = (*IDLE, *COUNTERS, 'flare_total_kv_tokens')
    result = {}
    for line in get(base + '/metrics').decode().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in selected:
            result[parts[0]] = float(parts[1])
    if not all(key in result for key in (*IDLE[:2], *COUNTERS, 'flare_total_kv_tokens')):
        raise RuntimeError('Expected native Flare metrics on the selected endpoint.')
    return result


def wait_idle(base, quiet_seconds):
    start = time.monotonic()
    since = previous = None
    while time.monotonic() - start < 900:
        values = metrics(base)
        tick = time.monotonic()
        counters = tuple(values[key] for key in COUNTERS)
        if all(values.get(key, 0) == 0 for key in IDLE) and (previous is None or previous == counters):
            since = tick if since is None else since
            if tick - since >= quiet_seconds:
                return
        else:
            since = None
        previous = counters
        time.sleep(2)
    raise RuntimeError('Endpoint did not become idle. No other requests were stopped.')


def valid(row):
    if not row or any(row.get(k) for k in ('num_errors', 'loop_detected', 'capacity_limited',
                                            'underfilled', 'warmup_timed_out')):
        return False
    c = row['concurrency']
    client, server = row['aggregate_tps'], row['server_gen_throughput']
    return (row['aggregate_source'] == 'openai_continuous_usage'
            and row['avg_running_reqs'] == row['max_running_reqs'] == c
            and row['max_queue_reqs'] == 0 and 29.5 <= row['measurement_wall_seconds'] <= 30.5
            and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (client, server))
            and abs(client - server) / server <= .05)


def summary(full, samples, rejected):
    by_cell = {(r['context_tokens'], r['concurrency']): r for r in full}
    rows = []
    for ctx in CONTEXTS:
        values = samples[ctx][:10]
        other = {str(c): (r['aggregate_tps'] if valid(r := by_cell.get((ctx, c))) else None)
                 for c in CONCURRENCIES[1:]}
        rows.append({'context_tokens': ctx, 'c1': {'n': len(values),
                    'mean': statistics.mean(values) if values else None,
                    'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
                    'samples': values}, 'c2_to_c32_tok_s': other})
    return {'benchmark_revision': REV, 'rows': rows, 'rejected_c1_windows': rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=1230)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in 1..65535')
    base = f'http://127.0.0.1:{args.port}'
    def models():
        return json.loads(get(base + '/v1/models'))['data']

    initial = models()
    model = next((m for m in initial if m['id'] == MODEL), None)
    if model is None or model.get('root') != 'deepseek-ai/DeepSeek-V4.1-Flash-1017':
        raise RuntimeError('Expected the recipe’s stock DeepSeek 1017 Flare endpoint.')
    if model['max_model_len'] < max(CONTEXTS) + 8192:
        raise RuntimeError('Model limit does not fit the requested 256K prompt and output budget.')
    kv = int(metrics(base)['flare_total_kv_tokens'])
    if kv < max(CONCURRENCIES) * (max(CONTEXTS) + 8192):
        raise RuntimeError('Reported KV capacity does not fit the full matrix.')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = get(CLIENT_URL)
    if hashlib.sha256(source).hexdigest() != SHA:
        raise RuntimeError('Pinned benchmark checksum did not match.')
    client = out / 'llm_decode_bench.py'
    client.write_bytes(source)
    env = dict(os.environ, LLM_BENCH_NO_UPDATE_CHECK='1')
    env.pop('LLM_BENCH_API_KEY', None)
    samples = {ctx: [] for ctx in CONTEXTS}
    rejected = []
    full = []
    for pass_number in range(1, 14):
        wait_idle(base, 15 if pass_number == 1 else 6)
        if models() != initial:
            raise RuntimeError('Serving model discovery changed during benchmarking.')
        name = 'full-matrix' if pass_number == 1 else f'c1-repeat-{pass_number:02d}'
        path = out / (name + '.json')
        cs = CONCURRENCIES if pass_number == 1 else (1,)
        cmd = [sys.executable, '-u', str(client), '--host', '127.0.0.1', '--port', str(args.port),
               '--model', MODEL, '--display-mode', 'plain', '--no-resume',
               '--concurrency', ','.join(map(str, cs)), '--contexts', '0,16k,32k,64k,128k,256k',
               '--duration', '30', '--max-tokens', '8192', '--decode-warmup-seconds', '3',
               '--prefill-contexts', '8k', '--kv-budget', str(kv), '--output', str(path)]
        print(f'Starting {name}', flush=True)
        with path.with_suffix('.log').open('x') as log:
            subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        if models() != initial:
            raise RuntimeError('Serving model discovery changed during benchmarking.')
        report = json.loads(path.read_text())
        meta = report['metadata']
        if meta['engine'] != 'flare' or not meta['metrics_available'] or meta['version'] != '0.8.0':
            raise RuntimeError('Benchmark did not use the pinned Flare-aware measurement path.')
        results = report['results']
        if pass_number == 1:
            full = results
        for row in results:
            if row['concurrency'] == 1:
                if valid(row):
                    samples[row['context_tokens']].append(row['aggregate_tps'])
                else:
                    rejected.append({'report': path.name, 'context_tokens': row['context_tokens'],
                                     'loop_detected': row.get('loop_detected', False)})
        (out / 'summary.json').write_text(json.dumps(summary(full, samples, rejected), indent=2) + '\n')
        print('Clean C1 counts:', {ctx // 1024: min(len(v), 10) for ctx, v in samples.items()}, flush=True)
        if all(len(v) >= 10 for v in samples.values()):
            break
    result = summary(full, samples, rejected)
    print('| Context | C1 mean +/- SD | C2 | C4 | C8 | C16 | C32 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for row in result['rows']:
        stats = row['c1']
        c1 = f'{stats["mean"]:.2f} +/- {stats["sample_sd"]:.2f}' if stats['n'] == 10 else 'incomplete'
        other = ' | '.join(f'{v:.2f}' if v is not None else 'flagged' for v in row['c2_to_c32_tok_s'].values())
        print(f'| {row["context_tokens"] // 1024}K | {c1} | {other} |')
    print(f'Raw logs and measurements: {out}')
    if any(row['c1']['n'] != 10 or None in row['c2_to_c32_tok_s'].values() for row in result['rows']):
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
