# RTX Pro 6000 Recipes

A catalog of independent serving recipes for NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs.
Each recipe folder is a standalone project with its own setup guide, configuration,
validation results, licenses and third-party notices. Open a recipe below to get started.

| Recipe | Model | GPUs | C1 speed (tok/s) | C32 throughput (tok/s) | KV capacity (tokens) |
|---|---|---|---:|---:|---:|
| [glm53-kraken-tp3-dcp3-zen](glm53-kraken-tp3-dcp3-zen/) | GLM-5.3-Flash (NVFP4/MXFP8 CSF-QAD) | 3x RTX PRO 6000 96 GB | ~227 | ~1,484 | ~9.5M |
| [deepseek-v41-brrr-b12x-hybrid-tp4](deepseek-v41-brrr-b12x-hybrid-tp4/) | DeepSeek-V4.1-Flash-1017 (RAM Engrams) | 4x RTX PRO 6000 96 GB | ~400 | ~1,824 | ~7.83M |
| [deepseek-v41-flare-brrr-d4-tp4](deepseek-v41-flare-brrr-d4-tp4/) | DeepSeek-V4.1-Flash-1017 (Flare, BRRR D4, RAM Engrams) | 4x RTX PRO 6000 96 GB | 357.33 ± 17.03 (395 W)<br>364.45 ± 21.30 (450 W) | ~2,223 (395 W)<br>~2,267 (450 W) | ~16.46M |
| [WARMIND v1.1 FLARE-Fused D4](warmind-v1.1-flare-fused-d4-tp4/) | WARMIND-1017 v1.1 (Flare, BRRR D4, RAM Engrams) | 4x RTX PRO 6000 96 GB | 381.93 ± 10.31 (450 W) | 2,462.5 (450 W) | 16,143,616 |

Decode figures use the 0K rows of each recipe's published sweep; C32 is aggregate throughput,
and KV capacity is shared across streams. Benchmark conditions and validation details are in each recipe's README.
The Flare recipes report C1 as mean ± sample SD over ten clean windows at each listed power limit.

Model weights and container images are not included; each recipe links to its original publishers.
