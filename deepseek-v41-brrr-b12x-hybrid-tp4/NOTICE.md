# Third-party notices

The original recipe wrappers and documentation are MIT licensed under this recipe's
[LICENSE](LICENSE). The small `patches/hybrid-dispatch.patch` is a derivative of
BRRR's Apache-2.0 decoder source and remains Apache-2.0. The unmodified upstream
[BRRR notice](licenses/BRRR-NOTICE) and [Apache-2.0 text](licenses/Apache-2.0.txt)
accompany it. This does not relicense any third-party component.

## Direct parent: BRRR

- [fraserprice/DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)
- Pinned revision: `0bc4dbd0a15458e0957b899d72c7a5089e3335a5`.
- Copyright 2026 Fraser Price; Apache License 2.0.
- The build downloads the original source, LICENSE and NOTICE from its publisher,
  verifies the public checksums, builds it, and adds the dispatch guard. No model
  weights or container images are redistributed here.

## BRRR's foundations

BRRR's own notice credits these Apache-2.0 projects and their respective authors:

- [vLLM](https://github.com/vllm-project/vllm).
- [Local Inference Lab's vLLM fork](https://github.com/local-inference-lab/vllm),
  including the DeepSeek-V4.1 integration patched by BRRR.
- [b12x](https://github.com/local-inference-lab/b12x), copyright the b12x authors.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), copyright NVIDIA and
  the FlashInfer community. b12x development moved into FlashInfer.

PyTorch, NVIDIA CUDA/CUTLASS/CuTe DSL and Linux/liburing are also used by the
runtime. They retain their own licences and notices. NVIDIA container/CUDA use is
subject to NVIDIA's terms; a public recipe is not a grant of those rights.

## Model checkpoint

- [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- The pinned model card declares MIT and its `LICENSE` contains the MIT text,
  copyright 2023 DeepSeek. Download it directly from the original location and
  retain that licence. Its text/vision/tokenizer and draft-head work belongs to DeepSeek.
- This recipe does not change, mirror or re-upload the checkpoint.

## Validation tooling

[Local Inference Lab's llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
is fetched from its original repository at a pinned commit with the exact recorded
0.6.2 script checksum. Its code and datasets are not included here. Consult its
own licence and notices before use. Public results here are numeric summaries,
not copies of prompts, saved completions, host telemetry or private run reports.
