# BRRR D7 Experiment At 450 W

**Decision: keep D4 as the general-serving default.** In the November 10, 2026
comparison, D7 was slower in 28 of 36 sustained-decode cells, including every
C32 cell (**0.6% to 4.8% slower**). It also reduced shared GPU KV capacity from
**16,464,000 to 15,437,312 tokens**, a loss of 1,026,688 tokens (**6.2%**).
D7 was faster on the Tetris coding prompt and passed both C30 correctness tests.
The decision was about general-serving throughput and KV capacity, not a
correctness failure.

## Matched Setup

The [D4 reference](results/2026-11-10-450w.json) and D7 trial used four 96 GB
RTX PRO 6000 Blackwell Workstation Edition GPUs on one PCIe Gen5 x16 TP4 host,
with a 24-core Threadripper PRO and about 251 GiB of Linux-visible RAM.
Both used **450 W configured per GPU**, not measured draw. All 208 power checks
in the D7 suite confirmed that cap on each serving card.

D7 was a separate Ignition export of the same DeepSeek V4.1 Flash 1017
checkpoint for Flare `972e28e`, with seven learned draft tokens instead of
four. It was **not** the stock Flare D7 package or a flag change on the
captured D4 graphs. The BRRR fusion gate stayed at 5-8 physical verification
rows. Both exports used native expert scales, BF16 MLA compute, an NVFP4 draft
head, and an FP8 Markov head. Serving kept 32 request slots, a 1,048,576-token
per-request limit, adaptive verification at alpha 0.6 with a 128-row cap, balanced
scheduling, RAM Engrams, no host prefix-cache tier, and KV sized from available
GPU memory. Adaptive verification can verify fewer than seven drafts per request.

## Sustained Decode

Aggregate tok/s from [LLM Inference Bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/52216d70865a015b34689626d04a80b89c9dc45f).
Both runs used 30-second windows, an 8,192-token output cap, the three-second
pre-matrix decode warmup, default sampling and reasoning, and the repetition
guard. C1 is **mean ± sample SD over ten valid windows per context**. C2-C32
are single valid windows.

| Context | D7 C1 mean ± SD | C2 | C4 | C8 | C16 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 349.74 ± 12.27 | 572.07 | 820.10 | 1,117.16 | 1,621.96 | 2,185.99 |
| 16K | 355.10 ± 22.28 | 576.49 | 845.86 | 1,128.30 | 1,528.69 | 2,142.40 |
| 32K | 357.11 ± 21.79 | 574.36 | 817.31 | 1,105.43 | 1,577.88 | 2,161.73 |
| 64K | 346.45 ± 12.86 | 557.43 | 811.54 | 1,089.81 | 1,585.73 | 2,153.84 |
| 128K | 336.73 ± 8.04 | 557.40 | 806.48 | 1,078.17 | 1,563.24 | 2,123.38 |
| 256K | 328.13 ± 9.45 | 530.63 | 805.08 | 1,053.10 | 1,510.33 | 2,019.17 |

Compared with D4, C1 means ranged from **4.0% lower to 0.7% higher**.
C32 dropped from 2,266.83 to 2,185.99 tok/s at 0K and from 2,092.67 to
2,019.17 tok/s at 256K. Prefill was mixed: the single 128K scout was
19,960 tok/s at 6.44 s TTFT on D7 versus 20,587 tok/s at 6.24 s on D4.
These are observed differences, not a general speed ranking for every prompt.

All 36 cells validated after targeted reruns. The HTTP request audit detected
one extra request in each of two 256K/C1 windows; both windows were discarded
and replaced at the end. Two C2 windows (128K and 256K) underfilled and the
initial 256K/C4 window tripped the repetition guard. All three also passed on
retry. There was no speed-based trimming.

## Tetris Generation

The same [Tetris runner](https://www.voipmonitor.org/tetristest.py) made three
isolated single requests per profile. End-to-end throughput includes reasoning
and time to first token. Peaks are the script's highest sampled interval in
each run, not sustained decode speeds.

| Run | D7 completion tokens | D7 elapsed | D4 end-to-end tok/s | D7 end-to-end tok/s | D4 peak tok/s | D7 peak tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44,068 | 77.4 s | 543.39 | 569.43 | 661.19 | 838.77 |
| 2 | 44,638 | 77.0 s | 537.56 | 579.86 | 664.48 | 868.12 |
| 3 | 50,177 | 86.5 s | 539.16 | 580.02 | 657.83 | 845.27 |

Mean end-to-end speed rose from **540.04 ± 3.01 to 576.44 ± 6.07 tok/s**
(**+6.7%**), and the mean of the three interval peaks rose from **661.17 to
850.72 tok/s (+28.7%)**. All three unmodified D7 games passed desktop start,
scoring, pause/resume, and audio-startup checks. The generations differed in
content and length; the result shows a gain on this prompt, not on all coding
workloads.

## Correctness At C30

Each D7 test ran one wave of 30 measured requests with maximum reasoning.
Server metrics confirmed all 30 active together, with no queueing or extra
requests. No measured requests errored, timed out, or truncated.

| Profile | D7 score | Aggregate decode at C30 | Slowest request |
| --- | ---: | ---: | ---: |
| Estonia 30 | **30/30 passed** | 2,710 tok/s | 28.8 s |
| LAVD 30 | **30/30 exact** | 3,207 tok/s | 279.4 s |

The C30 rates use native token counters after prefill while all 30 requests
were active. Estonia also ran one unscored prefill scout. These tests are
workload-specific checks, not evidence of general model-quality equivalence.
