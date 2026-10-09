> This project is based on GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

# glm53-kraken-tp3-dcp3-zen

A tested recipe for serving **GLM-5.3-Flash** with a **1M-token context** on **three RTX PRO 6000 Blackwell 96 GB**
GPUs, through an OpenAI-compatible API. It runs at **~230 tok/s for a single user** and **~1,480 tok/s aggregate at 32
concurrent streams**, with a 9.5M-token KV cache.

This folder does not contain the model or the container image. It holds a Docker Compose file, a launcher with safety
checks, a checkpoint downloader that verifies every file, and a validation script. Everything heavy comes from its
original publisher:

- **Container image:** [`azallaza/glm53-kraken-tp3:20261008`](https://hub.docker.com/r/azallaza/glm53-kraken-tp3), a vLLM
  fork with the B12X kernels and a GLM-5.3 tensor-parallel-3 port. It is published by azallaza
  ([BoxTops4Coochie/GLM5.3-Flash-TP3-Testing](https://github.com/BoxTops4Coochie/GLM5.3-Flash-TP3-Testing)). The image
  is built on Local Inference Lab's `karmic-kraken` vLLM.
- **Checkpoint:** [`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
  by [Local Inference Lab](https://huggingface.co/local-inference-lab), a QAD-distilled NVFP4/MXFP8 quantization of
  [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16).
- **Benchmark:** [`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench) v0.6.2.

**Nothing here redistributes the weights or the image.** `download_checkpoint.sh` fetches the checkpoint from its
original Hugging Face location and `docker pull` fetches the image from Docker Hub. The checkpoint's licence forbids
re-uploading or mirroring it, so link to the original instead. See [NOTICE.md](NOTICE.md) for all third-party terms.

What this recipe adds to the publisher's image is the configuration. The image defaults to 8 sequences and CUDA graphs up to
32 rows. This recipe serves 32 sequences, captures graphs up to 128 rows and uses a tuned NCCL setting. That removes a
3x throughput collapse above 8 concurrent users (details in [Settings and why](#settings-and-why)).

## Requirements

| | Tested with | Notes |
|---|---|---|
| GPUs | 3x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB | PCIe Gen5 x16, no NVLink, 395 W power limit. All three must be free: the server uses ~95% of their memory. |
| NVIDIA driver | 595.91.07 | Check with `nvidia-smi`. The image ships CUDA 13.4. |
| Docker | Docker Engine 29.6.1, Compose v5.3.1 | Compose v2 syntax (`docker compose`). |
| NVIDIA Container Toolkit | 1.19.1 | `docker run --rm --gpus all ubuntu nvidia-smi` must work. |
| OS | Ubuntu 24.04 (kernel 7.0) | Any Linux with the above should work. |
| CPU / RAM | 24-core Threadripper PRO 7965WX, 256 GB RAM | The container uses `ipc: host` and a 64 GB `/dev/shm`. |
| Disk | ~178 GB for the checkpoint + 12.9 GB for the image | Fast NVMe recommended; loading takes a few minutes. |
| Host tools | `bash`, `curl`, `jq`, `ss` (iproute2), `python3` | `python3` with `httpx rich psutil` for validation. |

No driver tweaks (P2P override, modprobe options) and no host tuning are needed. The numbers below come from a box
whose other GPUs were serving other models at the same time, sharing the PCIe fabric.

## Quick start

```bash
git clone https://github.com/shihanqu/RTX-Pro-6000-Recipes
cd RTX-Pro-6000-Recipes/glm53-kraken-tp3-dcp3-zen

cp .env.example .env
$EDITOR .env                   # GPU_IDS, HOST_PORT, CPUSET, HF_CACHE_DIR, RUNTIME_CACHE_DIR

docker pull azallaza/glm53-kraken-tp3@sha256:44c2d27ec0269e4620895a3ce21f639ee93f9e4d9b935e896ed52a3a6900c486
./download_checkpoint.sh       # ~178 GB; shows the licence and asks you to accept it
./launch.sh                    # checks, starts, waits for /health (6-12 min), prints the KV size
```

When it is ready, the launcher prints something like:

```text
Ready after 699 s: http://127.0.0.1:8000/v1  (model: GLM-5.3-Flash-Kraken-TP3)
CUDA graph sizes: [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
GPU KV cache size: 9,505,147 tokens, Maximum concurrency for 1,048,576 tokens per request: 9.06x
```

Try it:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-Kraken-TP3",
  "messages": [{"role": "user", "content": "Say hello in five languages."}],
  "max_tokens": 2000}' | jq -r '.choices[0].message.content'
```

The model thinks first (reasoning effort `max` by default). The reasoning comes back in `message.reasoning`, and the
answer in `content`. Allow enough `max_tokens` for both.

Other launcher commands: `./launch.sh check` (pre-start checks only), `./launch.sh config` (resolved compose file),
`./launch.sh logs`, `./launch.sh stop`. To stop and remove the container: `docker compose -p glm53-kraken-tp3-dcp3-zen down`.

### Files

| File | What it does |
|---|---|
| `compose.yaml` | The service: image pinned by digest, every serving setting in the `environment:` block, GPUs/port/CPUs/paths from `.env`. It uses the serve script baked into the image (no bind-mounted scripts). |
| `.env.example` | All machine-specific settings, documented. Copy to `.env`. |
| `launch.sh` | Refuses to start if the port is taken, a GPU is busy, the checkpoint receipt is missing or wrong, or the image is not pulled. Then starts the container, waits for `/health`, checks the startup log (PyNCCL all-reduce, FP8 decode path, CUDA graph ladder) and prints the KV size. Startup logs go to `logs/`. |
| `download_checkpoint.sh` | Downloads the pinned revision (`dec48abd`), verifies the size and SHA-256 of every file against the Hub manifest and writes `checkpoint-verification.json`. It uses the host's `huggingface_hub` if installed, otherwise runs inside the image (CPU only). `--verify` re-checks files already on disk. |
| `validate.sh` | Correctness and speed checks with llm-inference-bench v0.6.2 (see [Validation](#validation)). |
| `tools/` | Helpers: `.env` loader and checkpoint table, downloader, result summary. |

## Settings and why

All of this is in `compose.yaml`. The serve script baked into the image turns it into the vLLM command line.

| Setting | Value | Why |
|---|---|---|
| Checkpoint | `CHECKPOINT=csf-qad-2` (`dec48abd`), `MOE_TP=1` | CSF-QAD keeps the routed experts in NVFP4 with losslessly compressed scales, and stores attention and shared experts in MXFP8. The experts run W4A4 on B12X kernels, sharded over the 3 GPUs. Versus the original checkpoint: +3-5% decode at 12-32 streams, +3% KV, -12% prefill. |
| Parallelism | TP 3, `DCP=3` | Decode context parallelism splits the KV cache across the GPUs instead of replicating it. That gives 9.5M KV tokens, enough for nine 1M-token requests at once. |
| Speculative decoding | `MODE=mtp`, 3 draft tokens | The model's MTP head proposes 3 tokens per step. On real traffic about 2.9 tokens are accepted per step (2.6 on the synthetic benchmark). 2 draft tokens were 3-10% faster at 12-32 streams on the benchmark but slower for a single user, so 3 is kept. |
| Sequences | `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=1048576`, `MAX_NUM_BATCHED_TOKENS=4096`, `GPU_MEMORY_UTILIZATION=0.95` | 32 concurrent streams, full 1M context. |
| **CUDA graphs** | `MAX_CUDAGRAPH_CAPTURE_SIZE=128`, `CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16 24 32 48 64 96 128"` | **This is the main fix.** With MTP3, every stream puts 4 rows into each decode step. The default graph cap of 32 rows therefore covers only 8 streams. From 9 streams on, every step ran without CUDA graphs at a fixed ~125 ms, so each user got ~22 tok/s no matter how many there were (C12: 263 tok/s aggregate, below C8's 667). Capturing up to 128 rows (32 streams x 4) gives 896 tok/s at C12 and 1,484 at C32. The sparse ladder is as fast as a 15-size ladder and leaves ~157K more tokens of KV. |
| NCCL | `NCCL_PROTO=LL,LL128`, `NCCL_MIN_NCHANNELS=8`, `NCCL_MAX_NCHANNELS=8` | The low-latency protocols suit the small all-reduces of decode over PCIe. Measured: +5% at C1, +0-2% at C12-C32, ~200K more KV tokens (smaller NCCL buffers), ~2.5% slower prefill. `LL` with 4 channels was 9% slower at C24/C32. |
| All-reduce | `VLLM_ENABLE_PCIE_ALLREDUCE=0` (PyNCCL) | The configuration that was validated. The b12x PCIe one-shot all-reduce was not A/B tested on this image. |
| FP8 paths, L2 prefetch, KDA backends | `VLLM_GLM53_FP8_*`, `VLLM_GLM53_L2_PREFETCH_BUDGET_*`, `ADDITIONAL_CONFIG` | Carried over unchanged from the earlier validated deployment. |
| Reasoning | `REASONING_EFFORT=max`, `CLEAR_THINKING=true` | Server-side chat-template defaults. Clients can override them per request. |

Tried and not adopted: batch-size-dependent MTP depth (fastest at an empty context, but the engine crashed at C12
with 128K contexts), 1 draft token (C1 -7%), `VLLM_TP3_PCIE_DCP=1` (no gain), `VLLM_GLM53_FP8_DENSE_MAX_M=128`
(neutral).

## Results

Measured on 2026-10-08 with llm-inference-bench v0.6.2 (`./validate.sh sweep`: 30 s per cell, exact token targeting,
5 s decode warm-up) on the same machine. "Before" is the previous deployment: the publisher's earlier image
(`20260923`), the original NVFP4 checkpoint and the default 32-row graph cap, with otherwise the same settings.
"After" is this recipe.

### Aggregate decode tok/s, before → after

| context | C1 | C2 | C4 | C8 | C12 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 205 → **227** | 321 → **319** | 469 → **486** | 667 → **695** | 263 → **896** | 347 → **1031** | 691 → **1484** |
| 8K | 203 → **210** | 328 → **355** | 485 → **498** | 686 → **703** | 264 → **886** | 354 → **1033** | 705 → **1442** |
| 16K | 208 → **224** | 345 → **338** | 473 → **486** | 688 → **718** | 266 → **911** | 337 → **1037** | 681 → **1448** |
| 32K | 202 → **217** | 333 → **348** | 463 → **489** | 675 → **707** | 253 → **872** | 346 → **1032** | 694 → **1454** |
| 64K | 210 → **213** | 304 → **336** | 474 → **496** | 693 → **717** | 256 → **891** | 354 → **1027** | 685 → **1465** |
| 128K | 196 → **221** | 321 → **344** | 475 → **477** | 673 → **728** | 264 → **877** | 353 → **1038** | 695 → **1451** |
| 256K | 196 → **209** | 314 → **325** | 472 → **498** | 712 → **696** | 272 → **886** | 335 → **1001** | n/a → **1402** |
| 512K | 215 → **212** | 305 → **323** | 467 → **483** | 655 → **656** | 257 → **842** | n/a → **1008** | n/a |

"n/a" means the cell did not fit in the KV cache (before: 8.16M tokens; after: 9.51M). C32 at 512K needs 16.8M tokens.

### Per-user decode tok/s, before → after

| context | C1 | C2 | C4 | C8 | C12 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 208 → **229** | 163 → **163** | 119 → **124** | 85 → **89** | 22 → **75** | 22 → **65** | 22 → **47** |
| 128K | 199 → **225** | 163 → **174** | 121 → **122** | 85 → **91** | 23 → **74** | 23 → **65** | 22 → **45** |
| 512K | 215 → **215** | 155 → **164** | 120 → **122** | 84 → **84** | 27 → **72** | n/a → **65** | n/a |

At C1-C8, launch-to-launch noise on this box is about 2% (C1) to 6% (C8). The differences there are within noise.
The C12-C32 gains are not noise.

### Prefill (single request, tok/s)

| prompt | before | after | time to first token, after |
|---|---:|---:|---:|
| 8K | 12,746 | 11,034 | 0.74 s |
| 32K | 12,805 | 10,894 | 3.01 s |
| 128K | 12,410 | 10,601 | 12.37 s |

### Correctness (`./validate.sh full`)

| test | runs | concurrency | result | avg completion tokens | per-request tok/s |
|---|---:|---:|---|---:|---:|
| estonia (long-context reasoning, ~133K-token prompt) | 30 | 30 | **30/30 pass**, 0 truncated | 4,536 | 60.7 |
| LAVD (long structured-context consistency, ~20K prompt) | 30 | 30 | **30/30 exact** | 12,672 | 62.3 |

## Trade-off: prefill, and the `CHECKPOINT=default` fallback

Prefill is **~15% slower** than with the original checkpoint (12.7K → 10.8K tok/s). A 128K-token prompt takes ~12.4 s
instead of ~10.6 s. About 12 points of that come from the CSF-QAD checkpoint and ~2.5 from the 8 NCCL channels.

If your workload is mostly long prompts with short answers, use the original checkpoint on the same image:

```bash
# in .env: a separate cache dir for the other checkpoint (~199 GB, MIT licence per its model card)
CHECKPOINT=default
HF_CACHE_DIR="/srv/models/glm53-nvfp4/huggingface"
```

Then run `./download_checkpoint.sh` and `./launch.sh` again. The launcher switches to `MOE_TP=0` (expert parallel) and
checks the matching startup log line. Measured: prefill ~12.3K tok/s, KV 9.24M tokens, decode within noise at C12-C32
and 2-3% lower at C4/C8. This fallback has had less validation: estonia 10/10 and an A/B decode matrix, not the
30-run tests.

## Validation

```bash
git clone --branch v0.6.2 --depth 1 https://github.com/local-inference-lab/llm-inference-bench.git
python3 -m venv .venv && .venv/bin/pip install httpx rich psutil
export PYTHON=.venv/bin/python        # or set BENCH_DIR / PYTHON in .env

./validate.sh quick     # estonia, 10 runs at C8 (~5 min); expect 10/10
./validate.sh decode    # C1 and C12 at empty context (~3 min); expect ~220 and ~850-900 tok/s
./validate.sh full      # estonia 30 + LAVD 30 at C30 (~10 min); expect 30/30 and 30/30
./validate.sh sweep     # the full table above (~40 min)
```

A fresh copy of this folder was re-tested on 2026-10-09 with a cold runtime cache: startup took 699 s, KV was
9,505,147 tokens, `quick` gave 10/10, and `decode` gave 215 tok/s at C1 and 846 tok/s at C12.

Results go to `results/<mode>-<time>/`. The benchmark offers to upgrade itself at startup. `validate.sh` always runs it
with `</dev/null`, so that prompt is never answered, and it checks that the version is still 0.6.2 afterwards. Do the
same if you run it by hand.

## Troubleshooting

- **`Port ... is already in use` / `GPU n has compute processes`:** the launcher never stops anything for you. Stop
  the other server, or pick other GPUs or another port in `.env`.
- **`Checkpoint not verified`:** run `./download_checkpoint.sh`. If the files are already there, run
  `./download_checkpoint.sh --verify`. If you keep the receipt elsewhere, set `CHECKPOINT_RECEIPT`.
- **Startup takes long the first time:** kernels are JIT-compiled and autotuned into `RUNTIME_CACHE_DIR`. Keep that
  directory; later starts are faster. Raise `READY_TIMEOUT_S` (default 1800) on slow disks.
- **Container exits or runs out of memory during startup:** read `logs/startup-*.log`. The usual cause is another
  process holding VRAM on one of the GPUs (the server takes 95% of each). Also check `nvidia-smi` for ECC or Xid
  errors.
- **Throughput drops sharply above 8 concurrent users:** the CUDA graph ladder did not apply. The launcher checks for
  `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]` in the log. If you changed `MAX_NUM_SEQS` or
  the draft depth, keep the largest graph size at least `streams x (draft tokens + 1)`.
- **Requests run out of tokens without an answer:** the model reasons at length (`REASONING_EFFORT=max`). Raise
  `max_tokens`, or send `"chat_template_kwargs": {"reasoning_effort": "high"}` per request.
- **The benchmark asked to upgrade:** it was not run with `</dev/null`. Restore it with
  `git -C llm-inference-bench checkout v0.6.2 -- .`.
- **`docker compose` complains about `GPU_ID_1`:** start through `./launch.sh`, which derives `GPU_ID_1..3` from
  `GPU_IDS`.
- **Security:** the API has no authentication. Set `BIND_ADDRESS=127.0.0.1` unless you want it reachable from your
  network, and never expose it to the internet as is.

## Licence and attribution

- The scripts and documentation in this repository are released under the [MIT License](../LICENSE). That licence covers only this repository's own files, not the checkpoint or the container image.
- The checkpoint is licensed under the Local Inference Lab License, Version 1.0: **do not re-upload or mirror it**. Any
  project, model card or service that runs it must begin its README with the notice at the top of this file.
- The container image and the vLLM/B12X code in it belong to their authors, under their own licences.
- See [NOTICE.md](NOTICE.md) for all third-party attributions.
