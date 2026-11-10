# Third-party notices

The recipe scripts and documentation are distributed under the [MIT license](LICENSE).
The preparation patches contain modifications to Flare and FlashInfer and a
port of BRRR's CUDA decoder code; those portions retain the
[Apache-2.0 license](licenses/Apache-2.0.txt) and their upstream notices.
This recipe does not relicense other components.

## Flare and BRRR

- [NVIDIA Hyperflare Inference](https://github.com/NVIDIA/hyperflare-inference)
  at `972e28e44781d8d0c84d70ae434db92c36076a43`, Apache-2.0.
  See [Flare NOTICE](licenses/Flare-NOTICE). The patch is applied to Ignition
  preparation code. The unmodified serving image is obtained from Local
  Inference Lab's registry at the digest in `compose.yaml`.
- [Fraser Price's DeepSeek-4.1-Flash-1017-BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR),
  Apache-2.0. The small verification port retains the original
  [BRRR NOTICE](licenses/BRRR-NOTICE) and derives from revision
  `0bc4dbd0a15458e0957b899d72c7a5089e3335a5`. See the notice for vLLM,
  Local Inference Lab's vLLM fork, b12x, and FlashInfer attribution.
- [Local Inference Lab's FlashInfer fork](https://github.com/local-inference-lab/flashinfer)
  at `1f8ed8a3c46e27d03cff54e08f23e9bc50476d57` supplies the pinned
  preparation kernels and b12x. The expert-reuse workload patch comes from
  [FlashInfer PR #78](https://github.com/local-inference-lab/flashinfer/pull/78),
  revision `e16f949554f4efe8ee4154165231633e1cad8c05`. Only the workload module
  is patched; FlashInfer's original notices and licenses remain in force.

## Weights and tooling

- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, is downloaded from
  DeepSeek. Its pinned license is MIT, copyright 2023 DeepSeek.
- CUDA, PyTorch, the Rust and uv images, and container runtime components
  retain their respective terms. This recipe does not redistribute their images.
- [LLM Inference Bench](https://github.com/local-inference-lab/llm-inference-bench)
  is fetched at the pinned revision and verified before execution. Only
  numeric summaries are published here, not its code, prompts, or completions.

No checkpoint shards, locally exported packages, publisher keys, container
images, or private benchmark logs are included.
