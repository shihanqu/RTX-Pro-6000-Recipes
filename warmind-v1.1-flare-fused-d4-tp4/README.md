# WARMIND v1.1 FLARE-Fused D4

Serve [WARMIND-1017 v1.1](https://huggingface.co/jcartu/WARMIND-1017/tree/a6382f222da49b7109586eecbe1e53a10a4e0637) on four RTX PRO 6000 Blackwell Workstation 96 GB GPUs. This recipe starts with the **native WARMIND checkpoint**, exports new TP4 graphs with four learned draft tokens and BRRR's small-batch verification fusion, then serves the signed package with an unmodified, pinned Flare runtime. Vision and 32 simultaneous text streams are enabled. **Changing a runtime flag on the publisher's D7 package does not produce this D4 export.**

## Measured performance

Aggregate sustained decode tokens/s, not prompt processing or end-to-end response speed. Both deployments used the same four cards, 450 W per card, clock offsets, runtime, RAM Engrams and serving policy. The reference is the [publisher's WARMIND D7 package](https://huggingface.co/jcartu/WARMIND-1017-FLARE-gb202/tree/3c8e7b1a0f3551fb00afdc84a9ae96765f64d80d), **not stock DeepSeek**. 32K means 32,768 prompt tokens **per stream**.

| Streams | 0K D7 | 0K fused D4 | Change | 32K D7 | 32K fused D4 | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 337.8 | **381.9** | +13.1% | 342.0 | **381.6** | +11.6% |
| 2 | 576.9 | **628.7** | +9.0% | 567.3 | **623.2** | +9.9% |
| 4 | 898.0 | 892.0 | -0.7% | 856.0 | 900.5 | +5.2% |
| 8 | 1,225.3 | 1,214.6 | -0.9% | 1,211.3 | 1,189.0 | -1.8% |
| 16 | 1,744.9 | 1,758.3 | +0.8% | 1,745.1 | 1,770.7 | +1.5% |
| 32 | 2,422.0 | **2,462.5** | +1.7% | 2,344.6 | **2,473.1** | +5.5% |

At C1, mean (sample SD) was 381.93 (10.31) versus 337.78 (6.90) at 0K and 381.59 (7.21) versus 342.01 (19.06) at 32K. D4 has ten windows at C1/C2 and three at C4 and above, per context: **64 clean D4 windows**. D7 has ten at C1 and **one** at each higher concurrency per context. Small differences at C4 and above are not strong evidence of a repeatable gain. D4's shared GPU KV pool was **16,143,616 tokens**, down 5.5% from D7's 17,083,008. The per-stream context ceiling remains 1,048,576, not 32 independent million-token windows. [Individual samples, sample SDs, verification time and acceptance](results/d4-vs-d7.json).

**Why small batches are faster:** at 0K, verifier GPU time went from 7.29 to 6.47 ms at C1 and 8.79 to 7.64 ms at C2. Average accepted length stayed near 2.47 at C1 and actually *fell* at C2 (2.55 to 2.41). The improvement is consistent with cheaper verification, not more accepted draft tokens. This comparison varies the draft depth **and** the compiled verification path; it is not an ablation of either change alone.

## Correctness

| Gate | Qualified result |
|---|---|
| Estonia, 10 simultaneous runs | 10/10 correct; no errors, truncations or stalls |
| Four concurrent images with independent color and OCR targets | 4/4 correct for both fields; separate solid-color smoke test also 4/4 |
| Concurrent arithmetic | 32/32 exact responses; a separate structured-output check also passed |
| Native checkpoint integrity | All 88 files checked against the pinned upstream revision, including the index for all 48 weight shards |
| Exported weights | 12/12 sampled files (layers 4, 20, 36 on every TP rank) matched the publisher's WARMIND package byte for byte. This is a sample, not a whole-export equivalence claim. |

## Requirements

- Linux x86-64, Docker Engine with Compose v2, NVIDIA Container Toolkit, Git, Python 3.12, Hugging Face CLI (`hf`; tested with `huggingface_hub==1.6.0`), and four idle RTX PRO 6000 Blackwell **Workstation** 96 GB cards. Reference: driver 615.71.09, PCIe Gen5 x16 with peer access. Use `nvidia-smi topo -m` to inspect your own layout; rank order matters.
- Host memory: the reference had about 251 GiB usable. The launcher requires **200 GiB available** before export or startup. The embedding table uses about 189 GiB of RAM; it is not served from SSD. Container memory limit: 216 GiB, no additional swap.
- Storage: the native snapshot is **510,318,572,683 bytes** (~510 GB). Allow **1,200 GiB free on the export workspace after staging the input** for the package, compilation and scratch files. Use local NVMe; source and work can be on different filesystems. First export took **82 min 20 s** on the reference host, excluding download and builder construction. Timing and disk needs vary.
- The published speed run used **450 W** on each selected GPU and driver-reported clock offsets of **+60 MHz core / +6000 MHz memory**. They are measurement conditions, not instructions to overclock. Stability is hardware-specific. The wrapper checks the configured power limit; it does not set power, clocks, ACS, IOMMU, boot behavior or other services.

## Reproduce the export

Run these commands from this recipe folder. Use separate dedicated paths, keep the work directory private, and supply **your own four physical GPU UUIDs in TP rank order**. The numbers in the container arguments are its logical ranks, not host GPU indices.

```bash
nvidia-smi --query-gpu=index,uuid,name,power.limit --format=csv
export MODEL_DIR=/path/on/nvme/warmind-native
export WORK_DIR=/path/on/nvme/warmind-d4-work
export GPU_IDS='<rank-0-uuid>,<rank-1-uuid>,<rank-2-uuid>,<rank-3-uuid>'
export EXPECTED_WATTS=450
export HOST_PORT=1230
```

1. **Stage the native WARMIND weights.** Use `python3 recipe.py download` for a full pinned download. It verifies all 88 files. If the exact native snapshot is already present, point `MODEL_DIR` at it and use `python3 recipe.py verify`. Do not use a Flare package, a stock DeepSeek checkpoint, or an HF cache directory full of symlinks as the native input. The wrapper expects regular files. The [complete file manifest](manifests/native.json) includes content hashes and byte counts.
2. **Build the exporter.** The build uses pinned **public** sources; no private builder image or preexisting caches are required. `build --source-only` is an optional source/patch check. The unmodified runtime is fetched separately by digest.

```bash
python3 recipe.py build
docker pull ghcr.io/local-inference-lab/flare@sha256:08742b9e683425a5f2e86c8c4668ea9bac0c9cf041aaa1f3df9a1916de638b52
python3 recipe.py plan
```

3. **Export once, then inspect.** Plan checks the real Ignition arguments and kernel sources inside a CPU-only container. Export needs the four selected cards exclusively. Use a new package cache and signer; never use a stock package or `--reuse-weights`. WARMIND and stock DeepSeek are shape-compatible; a successful compile by itself does not establish which weights were exported.

```bash
python3 recipe.py export
python3 recipe.py inspect
python3 recipe.py check-layers
```

The last command is optional and reads the [12 reference samples](manifests/edited-layer-samples.json); run it outside timed benchmarks. Export produces a locally signed package and a private receipt in `WORK_DIR`. Regenerating a signer changes the package identity. The server verifies all package weights at startup and receives only the public signer profile; **never publish or mount the private signing key in the server**.

### Optional low-bandwidth staging

If you already have a distinct regular-file checkout of [stock native DeepSeek at the exact base revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), the alternative below uses the publisher's [pinned derivation manifest](https://huggingface.co/jcartu/WARMIND-1017/blob/a6382f222da49b7109586eecbe1e53a10a4e0637/DERIVATIVE.json):

```bash
python3 recipe.py download --stock-dir /path/to/pristine-native-deepseek --link-identical
```

It downloads **1.78 GB of replacement tensor data** plus metadata, checks 132 edits, and creates 33 new shards totaling **243.93 GB on disk**. With `--link-identical`, the other 15 shards are hash-checked and hard-linked; this requires both checkpoints on the same filesystem. Omit the flag to make independent copies. The tool never modifies the source weights; do not edit hard-linked files in place afterwards. The finished output must pass the **same full WARMIND manifest** as a fresh download.

## Start, check, stop

```bash
./launch.sh
python3 recipe.py status
curl --fail "http://localhost:${HOST_PORT:-1230}/health"
curl --fail "http://localhost:${HOST_PORT:-1230}/v1/models"
curl --fail "http://localhost:${HOST_PORT:-1230}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

Clients request **`deepseek-v4.1-flash`**. The model card's `root` and chat response's `model` identify **`DeepSeek-V4.1-Flash-1017-warmind-orth4c`**. The alias alone is not proof that WARMIND is loaded. The example disables thinking only for the quick arithmetic check; the decode benchmarks use defaults. Host publishing is loopback-only. For rollback, drain requests, run `python3 recipe.py stop`, and start your separately retained D7 launcher. This recipe does not evict a port owner, replace another launcher or change a boot default.

## Configuration

| Setting | Value |
|---|---|
| Export / runtime | TP4, SM120, 32 streams, 1,048,576 tokens/stream, 4 image feature slots; prefill chunks of 4,096 |
| Speculation | 4 learned draft tokens; packed adaptive verification; token capacity/budget 128; export graph budget 64; runtime alpha 0.6 |
| Preparation math | Native expert scales, BF16 MLA compute, NVFP4 draft head, FP8 Markov head, vocabulary-parallel sampling |
| Collectives and build | Runtime collective tuning, `gb202_pcie_tp4_clustered_full` ladder, four compile workers, default autotune effort |
| Serving memory | RAM embedding table; host prefix-cache memory 0; GPU KV limit `memory` (not a fixed token promise) |
| Scheduling | `balanced`; this is also the policy used for the D7 reference |

The BRRR port uses the compact native verifier for small physical verification batches. A batch row here is a **token to verify**, not a client request. The qualified C1 graph had **925 nodes** per rank; C2 with 5/6/7/8 packed rows had **916/910/910/908**. The wider 10-row C2 graph stayed distinct; larger workloads retain the ordinary path. The export produced 149 serving graph variants. [The catalog check](tools/catalog.py) inspects every rank and rejects a D7 package, a wrong model root, missing vision slots or a missing compact path. See [export Compose](compose.export.yaml), [runtime Compose](compose.yaml), and [preparation configuration](prepare.toml) for the complete settings.

## Repeat the validation

Install the [pinned client dependencies](requirements-bench.txt) in a private environment. The scripts fetch the [pinned LLM Inference Bench client](https://github.com/local-inference-lab/llm-inference-bench/blob/52216d70865a015b34689626d04a80b89c9dc45f/llm_decode_bench.py), verify its SHA-256 before execution, or accept an identical offline copy via `--client PATH`. Choose a **new** output directory per invocation. Reports and any client-generated text stay local.

```bash
python3 -m venv "$WORK_DIR/bench-venv"
"$WORK_DIR/bench-venv/bin/python" -m pip install -r requirements-bench.txt
"$WORK_DIR/bench-venv/bin/python" tools/validate.py --port "$HOST_PORT" --output-dir "$WORK_DIR/runs/quality"
"$WORK_DIR/bench-venv/bin/python" tools/benchmark.py --port "$HOST_PORT" --output-dir "$WORK_DIR/runs/d4"
```

Use `--smoke-only` on the validator for structured arithmetic and four concurrent image requests, or `--probe` on the benchmark for one audited 0K/C1 window. The full benchmark uses C1/C2/C4/C8/C16/C32 at 0K and 32K, a 3-second decode warmup, 30-second measurement windows, an 8,192 output-token cap, continuous usage reporting, normal server reasoning/sampling defaults and an initial 8K prefill scout. The scout is not counted as decode throughput. The runner rejects windows with wrong occupancy, queued/foreign traffic, loops, underfill, errors or client/server rate disagreement over 5%; it retries only missing cells. On a shared server, give it a quiet interval.

To reproduce the reference column, start the separately downloaded publisher D7 with the **same** GPU order, limits, clocks, alias, RAM embedding table, balanced scheduler, adaptive-verification alpha 0.6, verification capacity 128, disabled host prefix cache and memory-sized KV. Run the same benchmark with `--profile d7 --container YOUR_D7_CONTAINER` and a new output directory. Merely using the publisher's default runtime policy does not recreate the comparison. The runner checks the target's live image, model root, draft depth and runtime settings before and after each pass.

## Pins and attribution

| Component | Reproducible source |
|---|---|
| WARMIND | [jcartu's native v1.1](https://huggingface.co/jcartu/WARMIND-1017/tree/a6382f222da49b7109586eecbe1e53a10a4e0637); [publisher's GB202 D7 reference](https://huggingface.co/jcartu/WARMIND-1017-FLARE-gb202/tree/3c8e7b1a0f3551fb00afdc84a9ae96765f64d80d) |
| Base model | [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |
| Ignition exporter | [NVIDIA / Hyperflare Inference, `972e28e`](https://github.com/NVIDIA/hyperflare-inference/tree/972e28e44781d8d0c84d70ae434db92c36076a43) |
| Native verifier | [Fraser Price / BRRR recipe](https://github.com/fraserprice/DeepSeek-4.1-Flash-1017-BRRR); [pinned BRRR source package](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR/tree/0bc4dbd0a15458e0957b899d72c7a5089e3335a5) |
| Public fusion build kit | [Pinned Dockerfile, fusion patch and workload patch](https://github.com/shihanqu/RTX-Pro-6000-Recipes/tree/03e7a8a7ff48607d8fc7c155c1e965948cab5c54/deepseek-v41-flare-brrr-d4-tp4); [source and artifact hashes](tools/sources.py) |
| FlashInfer / B12X | [Local Inference Lab fork, `1f8ed8a`](https://github.com/local-inference-lab/flashinfer/tree/1f8ed8a3c46e27d03cff54e08f23e9bc50476d57) plus the [expert-reuse workload change from PR 78](https://github.com/local-inference-lab/flashinfer/pull/78) at `e16f949554f4efe8ee4154165231633e1cad8c05` |
| Runtime | Local Inference Lab's unmodified Flare OCI image; exact digest is in [Compose](compose.yaml). Only the **exporter** is patched. There is no public prebuilt fused WARMIND D4 package assumed by this recipe. |

## Troubleshooting

- **Looks healthy but unexpectedly slow at C1:** check the model root, signed catalog, four physical-rank order and `check-layers` first. The compatibility alias is intentionally identical to other deployments. Compare sustained decode, not end-to-end time; also check power, clocks under load and competing requests.
- **Package unreadable:** the runtime user is UID 10001. Grant read/traverse access only within the **exported package** if needed. Do not expose the signing directory, relax permissions on the entire workspace, or disable signature verification.
- **Busy GPUs, occupied port or low RAM:** the launcher reports the condition and does not stop unrelated services. Plan an idle maintenance window. To retain rollback, keep the D7 package, launcher and signing configuration separate.

Run the portable regression checks with `python3 -m unittest discover -s tests`. See [license and third-party notices](NOTICE.md). No model weights, container images, locally generated signatures or raw benchmark logs are redistributed here.
