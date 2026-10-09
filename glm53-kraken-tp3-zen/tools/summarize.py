#!/usr/bin/env python3
"""Print a short summary of llm-inference-bench v0.6.2 JSON results (used by validate.sh)."""

import json
import sys


def label(ctx: int) -> str:
    return "0" if ctx == 0 else f"{ctx // 1024}K"


def summarize(path: str) -> None:
    with open(path) as stream:
        data = json.load(stream)
    print(f"== {path}")
    if "selected_summary" in data:  # test profile (estonia, lavd)
        s = data["selected_summary"]
        tokens = s.get("completion_tokens", {})
        print(f"   correct {s['correct']}/{s['attempted']}  wrong {s['wrong']}  errors {s['errors']}  "
              f"truncated {s['truncated']}  scores {s.get('score_counts')}  "
              f"completion tokens avg {tokens.get('avg', 0):,.0f}  "
              f"concurrency {data.get('selected_concurrency')}")
        return
    results = [r for r in data.get("results", [])]
    concs = sorted({r["concurrency"] for r in results})
    ctxs = sorted({r["context_tokens"] for r in results})
    cells = {(r["context_tokens"], r["concurrency"]): r for r in results}
    print("   aggregate decode tok/s [per-user tok/s]")
    print("   ctx    " + "".join(f"{'C' + str(c):>16}" for c in concs))
    for ctx in ctxs:
        row = []
        for c in concs:
            r = cells.get((ctx, c))
            if not r or r.get("failure_reason") or r.get("aggregate_tps", 0) <= 0:
                row.append(f"{'-':>16}")
            else:
                row.append(f"{r['aggregate_tps']:>9.0f} [{r.get('output_tps_per_user_avg', 0):>4.0f}]")
        print(f"   {label(ctx):<7}" + "".join(row))
    prefill = data.get("prefill") or {}
    if prefill:
        print("   prefill tok/s: " + ", ".join(
            f"{label(int(k))} {v.get('tok_per_sec', 0):,.0f}" for k, v in prefill.items()))


for name in sys.argv[1:]:
    summarize(name)
