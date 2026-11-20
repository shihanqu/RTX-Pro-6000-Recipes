"""Reproduce the accepted 0K/32K sustained-decode matrix on a local WARMIND endpoint."""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from benchlib import MODEL, REV, client, env, snapshot, target, wait_idle

CS = (1, 2, 4, 8, 16, 32)
CONTEXTS = (0, 32768)


def issues(row, audit):
    bad = [
        k
        for k in (
            "num_errors",
            "loop_detected",
            "underfilled",
            "warmup_timed_out",
            "capacity_limited",
            "failure_reason",
        )
        if row.get(k)
    ]
    c = row["concurrency"]
    rate = row["aggregate_tps"]
    server = row["server_gen_throughput"]
    if row["aggregate_source"] != "openai_continuous_usage" or not all(
        isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in (rate, server)
    ):
        bad.append("unusable rate")
    elif abs(rate - server) / server > 0.05:
        bad.append("client/server disagreement")
    if row["avg_running_reqs"] != c or row["max_running_reqs"] != c or row["max_queue_reqs"] != 0:
        bad.append("occupancy")
    if not 29.5 <= row["measurement_wall_seconds"] <= 30.5:
        bad.append("window duration")
    if (
        not audit
        or not audit["quiet_at_entry"]
        or audit["server_requests"] != audit["client_requests"]
        or sum(audit["client_requests"].values()) < c
    ):
        bad.append("HTTP audit")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=1230)
    p.add_argument("--profile", choices=("d4", "d7"), default="d4")
    p.add_argument("--container")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--client", type=Path, help="optional local copy of the pinned client")
    p.add_argument(
        "--probe",
        action="store_true",
        help="One audited 0K/C1 window to check the setup, not the full matrix",
    )
    a = p.parse_args()
    depth = 4 if a.profile == "d4" else 7
    if not 1 <= a.port <= 65535 or (depth == 7 and not a.container):
        p.error("Use a valid port and specify --container for the D7 reference")
    os.umask(0o077)
    base = f"http://localhost:{a.port}"
    identity = target(base, a.port, depth, a.container)
    budget = int(snapshot(base)[0]["flare_total_kv_tokens"])
    required = max(CS) * (max(CONTEXTS) + 8192)
    if budget < required:
        raise ValueError("Reported shared KV is too small for the matrix")
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = client(out, a.client)
    contexts, concurrencies = ((0,), (1,)) if a.probe else (CONTEXTS, CS)
    goals = {
        (ctx, c): (1 if a.probe else 10 if c == 1 or (depth == 4 and c == 2) else 3 if depth == 4 else 1)
        for ctx in contexts
        for c in concurrencies
    }
    samples = {k: [] for k in goals}
    excluded = []

    def save():
        rows = []
        for (ctx, c), values in samples.items():
            rows.append(
                dict(
                    context_tokens=ctx,
                    concurrency=c,
                    n=len(values),
                    target=goals[ctx, c],
                    windows_tok_s=values,
                    mean=statistics.mean(values) if values else None,
                    sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                )
            )
        tmp = out / "summary.tmp"
        tmp.write_text(
            json.dumps(
                dict(benchmark_revision=REV, profile=a.profile, rows=rows, rejected=excluded), indent=2
            )
            + "\n"
        )
        tmp.replace(out / "summary.json")

    def collect(name, cs, contexts, first=False):
        wait_idle(base)
        if target(base, a.port, depth, a.container) != identity:
            raise RuntimeError("Deployment changed")
        path = out / (name + ".json")
        audit = out / (name + ".audit.jsonl")
        args = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("observe.py")),
            str(source),
            "--host",
            "localhost",
            "--port",
            str(a.port),
            "--model",
            MODEL,
            "--display-mode",
            "plain",
            "--no-resume",
            "--concurrency",
            ",".join(map(str, cs)),
            "--contexts",
            ",".join(f"{v // 1024}k" if v else "0" for v in contexts),
            "--duration",
            "30",
            "--max-tokens",
            "8192",
            "--decode-warmup-seconds",
            "3",
            "--prefill-contexts",
            "8k",
            "--kv-budget",
            str(budget),
            "--output",
            str(path),
        ]
        if not first:
            args.append("--skip-prefill")
        with (out / (name + ".log")).open("x") as log:
            subprocess.run(
                args,
                env={**env(), "WARMIND_CELL_AUDIT": str(audit)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=4500,
            )
        if target(base, a.port, depth, a.container) != identity:
            raise RuntimeError("Deployment changed during pass")
        report = json.loads(path.read_text())
        meta = report["metadata"]
        if not (
            meta["version"] == "0.8.0"
            and meta["engine"] == "flare"
            and meta["metrics_available"]
            and meta["ignore_eos"]
            and meta["loop_detection"]["enabled"]
            and meta["duration_per_test"] == 30
            and meta["max_tokens"] == 8192
            and meta["temperature"] is None
        ):
            raise ValueError("Unexpected measurement settings")
        audits = {
            (v["context"], v["concurrency"]): v
            for line in audit.read_text().splitlines()
            if (v := json.loads(line))["duration"] == 30
        }
        expected = {(ctx, c) for ctx in contexts for c in cs}
        rows = report["results"]
        if len(rows) != len(expected) or {(r["context_tokens"], r["concurrency"]) for r in rows} != expected:
            raise ValueError("Incomplete pass")
        for r in rows:
            key = r["context_tokens"], r["concurrency"]
            bad = issues(r, audits.get(key))
            if bad:
                excluded.append({"pass": name, "cell": key, "reasons": bad})
            elif len(samples[key]) < goals[key]:
                samples[key].append(r["aggregate_tps"])
        save()
        print(
            f"{name}: {sum(map(len, samples.values()))}/{sum(goals.values())} valid; {len(excluded)} rejected",
            flush=True,
        )

    collect("matrix", concurrencies, contexts, True)
    for i in range(1, 36):
        groups = defaultdict(list)
        for c in concurrencies:
            missing = tuple(ctx for ctx in contexts if len(samples[ctx, c]) < goals[ctx, c])
            if missing:
                groups[missing].append(c)
        if not groups:
            break
        for j, (needed, cs) in enumerate(groups.items()):
            collect(f"repeat-{i:02d}-{j}", cs, needed)
    if any(len(samples[k]) != n for k, n in goals.items()):
        raise SystemExit("Some cells did not reach their clean-window target")


if __name__ == "__main__":
    main()
