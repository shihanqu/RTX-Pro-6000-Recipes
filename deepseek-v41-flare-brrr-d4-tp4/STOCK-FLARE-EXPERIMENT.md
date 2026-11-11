# Stock Flare And BRRR At 450 W

Clean measurements from November 10-11, 2026. **Fused D4 led C32 at every
tested context.** Stock D7 had the largest shared KV pool, while fused D7 had
the highest mean Tetris throughput and interval peaks. These results compare
prepared deployments, not BRRR fusion or draft depth in isolation.

## Test Setup

All three profiles served DeepSeek V4.1 Flash 1017 with Flare `972e28e` on the
same TP4 host: four 96 GB RTX PRO 6000 Blackwell Workstation Edition GPUs,
PCIe Gen5 x16 peer access, a 24-core Threadripper PRO, and about 251 GiB of
Linux-visible RAM. **450 W** is the configured per-GPU cap, not measured draw.
All 1,194 sampled power checks across the two stock runs matched
that cap on each serving card.

| Profile | Prepared package | Drafts | Shared GPU KV tokens |
| --- | --- | ---: | ---: |
| Stock D7 | [Published Flare package at `6d51b29`](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE/tree/6d51b296f32f24967225d71b4b14eb8ffaa68e15), without BRRR fusion | 7 | 16,962,176 |
| Fused D7 | [Locally exported BRRR D7 experiment](D7-EXPERIMENT.md) | 7 | 15,437,312 |
| Fused D4 | [BRRR D4 recipe, 450 W reference](results/2026-11-10-450w.json) | 4 | 16,464,000 |

All used 32 request slots, a 1,048,576-token per-request limit, adaptive
verification at alpha 0.6 with a 128-row cap, balanced scheduling, RAM Engrams,
and no host prefix-cache tier. The two BRRR packages fuse eligible graphs with
5-8 physical verification rows. The stock package does not use those kernels;
its compatibility API alias is not evidence of fusion. The prepared graph catalogs
and memory layouts can differ even between the two D7 profiles.

## Sustained Decode

Aggregate decode tok/s from [LLM Inference Bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/52216d70865a015b34689626d04a80b89c9dc45f).
All profiles used 30-second windows, an 8,192-token output cap, a three-second
pre-matrix decode warmup, default sampling and reasoning, and the repetition
guard. **C1 is mean ± sample SD over the first ten valid windows per context**;
C2-C32 are single valid windows. Contexts are nominal per-stream targets.

| Context | Profile | C1 mean ± SD | C2 | C4 | C8 | C16 | C32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | Stock D7 | 334.62 ± 16.21 | 537.05 | 816.96 | 1,135.69 | 1,409.42 | 2,189.97 |
| 0K | Fused D7 | 349.74 ± 12.27 | 572.07 | 820.10 | 1,117.16 | 1,621.96 | 2,185.99 |
| 0K | Fused D4 | 364.45 ± 21.30 | 632.32 | 842.44 | 1,145.03 | 1,680.14 | 2,266.83 |
| 16K | Stock D7 | 326.90 ± 7.20 | 543.57 | 868.15 | 930.78 | 1,615.53 | 2,155.10 |
| 16K | Fused D7 | 355.10 ± 22.28 | 576.49 | 845.86 | 1,128.30 | 1,528.69 | 2,142.40 |
| 16K | Fused D4 | 358.31 ± 16.10 | 620.17 | 856.66 | 1,146.24 | 1,616.45 | 2,250.69 |
| 32K | Stock D7 | 338.84 ± 23.81 | 547.13 | 817.76 | 1,028.95 | 1,621.59 | 2,178.81 |
| 32K | Fused D7 | 357.11 ± 21.79 | 574.36 | 817.31 | 1,105.43 | 1,577.88 | 2,161.73 |
| 32K | Fused D4 | 354.68 ± 14.35 | 569.32 | 815.44 | 1,118.26 | 1,572.19 | 2,239.50 |
| 64K | Stock D7 | 321.19 ± 5.50 | 524.76 | 822.27 | 1,136.13 | 1,570.82 | 2,122.94 |
| 64K | Fused D7 | 346.45 ± 12.86 | 557.43 | 811.54 | 1,089.81 | 1,585.73 | 2,153.84 |
| 64K | Fused D4 | 351.33 ± 6.96 | 595.65 | 830.11 | 1,128.23 | 1,583.15 | 2,252.78 |
| 128K | Stock D7 | 320.71 ± 16.90 | 507.98 | 824.09 | 1,039.80 | 1,554.84 | 2,079.92 |
| 128K | Fused D7 | 336.73 ± 8.04 | 557.40 | 806.48 | 1,078.17 | 1,563.24 | 2,123.38 |
| 128K | Fused D4 | 344.80 ± 19.02 | 563.50 | 811.13 | 1,090.78 | 1,562.54 | 2,136.42 |
| 256K | Stock D7 | 314.53 ± 18.62 | 531.49 | 760.37 | 1,043.95 | 1,523.53 | 1,998.00 |
| 256K | Fused D7 | 328.13 ± 9.45 | 530.63 | 805.08 | 1,053.10 | 1,510.33 | 2,019.17 |
| 256K | Fused D4 | 334.91 ± 6.37 | 523.24 | 775.70 | 1,098.84 | 1,545.01 | 2,092.67 |

Stock D7 had **3.0% more KV** than fused D4 and **9.9% more** than fused D7.
Against fused D4, stock D7 was **4.5% to 8.8% slower at C1** and **2.6% to 5.8%
slower at C32** across the six contexts. Against fused D7, stock D7 was **4.1%
to 7.9% slower at C1**; C32 ranged from **2.0% lower to 0.8% higher**.
The single C2-C32 windows are observed rates, not estimates of run-to-run
variation.

The stock matrix retains 45 valid C1 windows and 28 higher-concurrency cells
from the initial suite. Recovery supplied the remaining 15 C1 windows and
two cells on the unchanged container. Across both phases, 160 decode windows
were excluded, 156 with extra HTTP requests observed in the audit. Only
windows with matching request counts, validated occupancy, and no other
benchmark flags were kept. There was no speed-based trimming. Fused D7 also
used the HTTP audit; fused D4 used the occupancy and token-counter checks
recorded in its reference run.

## Prefill Scouts

Each entry is **prefill tok/s / time to first token in seconds**, one scout per
profile and context. The stock 8K scout passed its own HTTP audit; the larger
stock scouts come from the first valid C1 cell at each matching context. These
are single scouts, not repeated prefill means.

| Context | Stock D7 | Fused D7 | Fused D4 |
| --- | ---: | ---: | ---: |
| 8K | 18,122 / 0.452 | 17,927 / 0.457 | 17,935 / 0.457 |
| 16K | 20,766 / 0.781 | 19,556 / 0.829 | 19,035 / 0.852 |
| 32K | 21,214 / 1.520 | 20,229 / 1.594 | 19,765 / 1.632 |
| 64K | 21,577 / 2.982 | 20,873 / 3.083 | 20,356 / 3.161 |
| 128K | 20,319 / 6.325 | 19,960 / 6.439 | 20,587 / 6.243 |

## Tetris Generation

Three isolated requests per profile with the same
[Tetris runner](https://www.voipmonitor.org/tetristest.py). End-to-end tok/s
includes reasoning tokens and time to first token. Peak tok/s is the highest
sampled interval within that run, not a sustained decode rate.

| Run | Stock D7 end-to-end | Stock D7 peak | Fused D7 end-to-end | Fused D7 peak | Fused D4 end-to-end | Fused D4 peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 587.84 | 813.82 | 569.43 | 838.77 | 543.39 | 661.19 |
| 2 | 561.68 | 783.58 | 579.86 | 868.12 | 537.56 | 664.48 |
| 3 | 544.23 | 805.64 | 580.02 | 845.27 | 539.16 | 657.83 |
| Mean | 564.58 | 801.01 | 576.44 | 850.72 | 540.04 | 661.17 |

Stock D7 averaged **564.58 ± 21.95 tok/s** end to end, **4.5% faster** than
fused D4 and **2.1% slower** than fused D7 on this prompt. Its mean interval
peak was **21.2% above** fused D4 and **5.8% below** fused D7. The highest
observed peaks were **813.82** (stock D7), **868.12** (fused D7), and **664.48**
(fused D4). The generated outputs differed in length and content.

All three unmodified stock D7 games and all three fused D7 games passed the
desktop smoke check. Two of the three fused D4 games passed. No generated
HTML or raw response text is published here.

## Correctness At C30

For each measured cell, all 30 requests ran together with maximum reasoning,
no competing requests or queueing, and no errors or truncation. The C30
decode rates below are aggregate tok/s after prefill while all 30 requests
were active. Estonia used one additional unscored prefill scout.

| 450 W profile | Estonia 30 score | Estonia C30 tok/s | LAVD 30 score | LAVD C30 tok/s |
| --- | --- | ---: | --- | ---: |
| Stock D7 | **30/30 passed** | 2,747 | **30/30 exact** | 3,158 |
| Fused D7 | **30/30 passed** | 2,710 | **30/30 exact** | 3,207 |
| Fused D4 | Not measured | N/A | Not measured | N/A |

The stock LAVD result is from the isolated recovery wave, not the earlier
traffic-affected runs. Fused D4's [November 9 qualification](README.md#correctness)
recorded Estonia 10/10 separately at C1, C2 and C10. An
[older November 7 D4 build](https://github.com/shihanqu/RTX-Pro-6000-Recipes/blob/1d9e854980e02ca07116b6632b869df9bc702410/deepseek-v41-flare-brrr-d4-tp4/README.md#correctness-at-c30)
scored 30/30 on both tests at C30. Neither is a matched 450 W C30 result for
the fused D4 deployment in this table.
