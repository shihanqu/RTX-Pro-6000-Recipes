> This project is based on GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

# RTX Pro 6000 Recipes

Tested serving recipes for machines with NVIDIA RTX PRO 6000 Blackwell 96 GB GPUs. Each recipe is a self-contained
folder with a pinned container image and checkpoint revision, a Docker Compose file, a launcher with safety checks,
and the benchmark and correctness results it was validated with. These are configurations only: weights and images
come from their original publishers, and each recipe credits them.

| Recipe | Model | GPUs | Default port | Headline numbers | Validated |
|---|---|---|---|---|---|
| [glm53-kraken-tp3-dcp3-zen](glm53-kraken-tp3-dcp3-zen/) | GLM-5.3-Flash (NVFP4/MXFP8 CSF-QAD), 1M context, MTP3 | 3x RTX PRO 6000 96 GB | 8000 | ~227 tok/s at 1 user, ~896 at 12, ~1,484 aggregate at 32; 9.5M-token KV cache; prefill ~10.8K tok/s | estonia 30/30, LAVD 30/30 at 30 concurrent; full 1-32 x 0-512K sweep |

```bash
git clone https://github.com/shihanqu/RTX-Pro-6000-Recipes
cd RTX-Pro-6000-Recipes/glm53-kraken-tp3-dcp3-zen
```

Then follow that folder's README.

## Licence

The scripts and documentation here are MIT licensed (see [LICENSE](LICENSE)). Model weights and container images are not included and stay under their publishers' own licences; each recipe's README and NOTICE.md give the details and required attributions.
