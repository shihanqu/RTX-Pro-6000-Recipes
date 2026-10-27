# RTX Pro 6000 Recipes

A catalog of independent serving recipes for NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs.
Each recipe folder is a standalone project with its own setup guide, configuration,
validation results, licenses and third-party notices. Open a recipe below to get started.

| Recipe | Model | GPUs | Headline numbers | Validated |
|---|---|---|---|---|
| [glm53-kraken-tp3-dcp3-zen](glm53-kraken-tp3-dcp3-zen/) | GLM-5.3-Flash (NVFP4/MXFP8 CSF-QAD), 1M context, MTP3 | 3x RTX PRO 6000 96 GB | ~227 tok/s at 1 user, ~896 at 12, ~1,484 aggregate at 32; 9.5M-token KV cache; prefill ~10.8K tok/s | estonia 30/30, LAVD 30/30 at 30 concurrent; full 1-32 x 0-512K sweep |
| [deepseek-v41-brrr-b12x-hybrid-tp4](deepseek-v41-brrr-b12x-hybrid-tp4/) | DeepSeek-V4.1-Flash-1017, original BRRR/B12X hybrid, 1M configured context, DSpark2, NVMe Engram | 4x RTX PRO 6000 96 GB | ~350-400 tok/s at C1, ~1,250 at C16, ~1,750 aggregate at C32; ~8.1M shared KV tokens | Estonia 30/30 and LAVD 30/30 at C30/max reasoning; C4 vision and actual C32; 29/30 usable cells in 0-128K sweep (one repetition failure retained) |

Model weights and container images are not included; each recipe links to its original publishers.
