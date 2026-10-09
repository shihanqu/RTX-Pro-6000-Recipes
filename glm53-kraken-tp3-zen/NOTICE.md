# Third-party notices

This recipe contains only configuration and helper scripts. It does not include or redistribute any model weights,
container images or third-party source code. The components it downloads or runs belong to their authors and
stay under their own terms. Read them before you use them.

## GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD (model checkpoint)

> This project is based on GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

- Original location: <https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD>. The pinned
  revision is `dec48abd33efa73c3bb7c95b74eee10cad34f9be`.
- Licence: Local Inference Lab License, Version 1.0 (SPDX `LicenseRef-LIL-1.0`). It is the Apache License 2.0 terms
  plus additional restricting conditions, and it is not the Apache License. The binding text is the `LICENSE` file
  in the model repository. Among its conditions:
  - Re-uploading or mirroring the checkpoint, or any substantially similar copy, is not permitted. Link to the
    original location instead.
  - Every README, model card or landing page of a project that runs the checkpoint must begin with the attribution
    notice above.
- Upstream material: [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), MIT licence.

## GLM-5.3-Flash-NVFP4 (optional fallback checkpoint, `CHECKPOINT=default`)

- [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4), pinned
  revision `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49`, by Local Inference Lab. Its model card lists the MIT licence.
  Check the repository's licence files for the revision you download.

## Container image `azallaza/glm53-kraken-tp3:20261008`

- Docker Hub: <https://hub.docker.com/r/azallaza/glm53-kraken-tp3>, pinned by digest
  `sha256:44c2d27ec0269e4620895a3ce21f639ee93f9e4d9b935e896ed52a3a6900c486`.
- Published by azallaza. Build notes and release history:
  [BoxTops4Coochie/GLM5.3-Flash-TP3-Testing](https://github.com/BoxTops4Coochie/GLM5.3-Flash-TP3-Testing).
- The image contains a GLM-5.3 tensor-parallel-3 port of vLLM. It is built on Local Inference Lab's `karmic-kraken`
  vLLM build, with the B12X kernels, FlashInfer, LMCache, PyTorch and the NVIDIA CUDA stack, each under its own
  licence. The serve script used by this recipe (`/usr/local/bin/serve-glm53-flash-tp3-kraken.py`) is the one baked
  into the image. It is not copied here.
- Use of the NVIDIA base image is subject to NVIDIA's licence terms for its containers and CUDA components.

## llm-inference-bench (validation only)

- <https://github.com/local-inference-lab/llm-inference-bench>, tag `v0.6.2`, by Local Inference Lab. `validate.sh`
  calls a checkout of it that you clone yourself. It is not included here.

## vLLM

- <https://github.com/vllm-project/vllm>, Apache License 2.0. The fork inside the image is based on it.
