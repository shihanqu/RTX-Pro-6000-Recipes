# RTX Pro 6000 Recipes

A catalog of independent serving recipes for NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs.
Each recipe folder is a standalone project with its own setup guide, configuration,
validation results, licenses and third-party notices. Open a recipe below to get started.

| Recipe | Model | GPUs | C1 speed (tok/s) | C32 throughput (tok/s) | KV capacity (tokens) |
|---|---|---|---:|---:|---:|
| [glm53-kraken-tp3-dcp3-zen](glm53-kraken-tp3-dcp3-zen/) | GLM-5.3-Flash (NVFP4/MXFP8 CSF-QAD) | 3x RTX PRO 6000 96 GB | ~227 | ~1,484 | ~9.5M |
| [deepseek-v41-brrr-b12x-hybrid-tp4](deepseek-v41-brrr-b12x-hybrid-tp4/) | DeepSeek-V4.1-Flash-1017 (BRRR/B12X hybrid) | 4x RTX PRO 6000 96 GB | ~342 | ~1,755 | ~8.1M |

Decode figures use the 0K rows of each recipe's published sweep; C32 is aggregate throughput,
and KV capacity is shared across streams. Benchmark conditions and validation details are in each recipe's README.

Model weights and container images are not included; each recipe links to its original publishers.
