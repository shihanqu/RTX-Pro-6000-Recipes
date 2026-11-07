# DeepSeek V4.1 Flash 1017: Flare BRRR D4 (TP4)

A separate Flare recipe for stock DeepSeek-V4.1-Flash-1017 on four RTX PRO 6000
Blackwell Workstation Edition 96 GB cards. It prepares a local package with
BRRR small-batch fusion, four draft tokens and RAM-resident Engrams. The
[vLLM BRRR/B12X recipe](../deepseek-v41-brrr-b12x-hybrid-tp4/) remains independent.

## Requirements

- Linux x86-64, Docker with BuildKit and Compose v2, NVIDIA Container Toolkit
  and a driver compatible with the pinned CUDA images. Use four idle RTX PRO
  6000 Blackwell Workstation Edition cards on a working peer-access topology.
- Bash 4.4+, Git, Python 3, `nvidia-smi`, `ss` and the Hugging Face `hf` CLI.
- A dedicated NVMe work directory. The native checkpoint is **510.3 GB**; the
  export guard requires **1,200 GiB free after the download** for cache, staging
  and package output.
- At least **200 GiB available host RAM** at export and launch. The containers
  are limited to 216 GiB of host memory with no additional swap; RAM Engrams
  account for roughly 189 GiB before runtime overhead.

The scripts refuse busy GPUs and a listening launch port. They do not stop
services, modify host boot settings, or change GPU power limits or clocks.

## Build and export

Run from this recipe directory on the GPU host. Use actual physical UUIDs in
the desired rank order:

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
export WORK_DIR=/path/on/nvme/flare-brrr-d4-work
export GPU_IDS='GPU-first,GPU-second,GPU-third,GPU-fourth'

./download_checkpoint.sh
./build.sh
./export.sh
```

The download is pinned to the stock
[DeepSeek 1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
It verifies all 83 files using [the manifest](manifests/model.json). To reuse
a local copy, set `MODEL_DIR` to its absolute path and run
`python3 tools/checkpoint.py "$MODEL_DIR"` instead of downloading. Export
checks sizes again and mounts the checkpoint read-only into the offline cache.

The build fetches [Flare at `18b16e1`](https://github.com/NVIDIA/spark-local-inference/commit/18b16e170bb9eab03e2b1800107c72966c56594f),
applies [the preparation patch](patches/flare-brrr-d4.patch), and uses its
locked preparation dependencies, including FlashInfer `e977747a`, Torch
`2.14.1+cu132` and the pinned CUDA 13.2.1 development image. Export uses native
expert scales, BF16 MLA compute, the NVFP4 draft head, FP8 Markov head, 4,096-token
prefill chunks, 32 request slots and 128 packed verification rows. Its runtime
collective ladder comes from the pinned Flare tree.

Ignition writes the package below `$WORK_DIR/packages/models/` and generates
a local development signer in `$WORK_DIR/settings/`. Keep the private
`development-signing-key.pem` private. The recipe does not upload packages,
weights, images, or keys. Export can take a while and must have exclusive
use of the selected GPUs.

## Serve

```bash
export HOST_PORT=1230
./launch.sh
docker logs -f deepseek-v41-flare-brrr-d4-serve
```

Set `PACKAGE_DIR` to a specific absolute export directory when the work directory
contains more than one package. Use the same work directory and local publisher
profile that signed that export. The launch script approves only the exact
fingerprint of that local profile. Do not use it to approve an unrelated key.

The pinned Flare serving image uses D4 adaptive verification with alpha 0.6,
a 128-row verification cap, the balanced scheduler, RAM Engrams, GPU KV sized
from available memory, and no host prefix-cache tier. Weight verification
can delay readiness. The unauthenticated API is published to **loopback only**:

```bash
curl --fail "http://127.0.0.1:$HOST_PORT/health"
curl --fail "http://127.0.0.1:$HOST_PORT/v1/models"
```

The served ID is `DeepSeek-4.1-Flash-1017-BRRR`. To stop this recipe's container,
use `docker stop deepseek-v41-flare-brrr-d4-serve`.

## Comparison With Flare And BRRR

### Kernels And Deployment

**Versus stock Flare:** The [published gb202 package](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE/blob/32385269008081f7870cb04291d269d1bbee0788/README.md)
ships with native Flare/b12x graphs prepared at `78b18b1c` and seven learned
draft tokens (D7). This recipe starts from the same native checkpoint but
exports and signs its own four-draft (D4) package. The packed stock package
supports its captured depth or no drafting; setting `--mtp-draft-tokens 4`
on that package is not a substitute for exporting D4. The locally exported
unfused D4 control used the same head precisions and serving settings as this
recipe, but without the BRRR kernel port.

**What the port fuses:** Only eligible target-model verification graphs with
**5-8 physical token rows** use BRRR. The gate requires TP4, SM120, BF16
activations and native expert scales, not compressed CSF scales. The persistent
suffix combines the grouped attention output projection, TP reductions,
hyperconnections, router, and shared and routed expert work. On compatible
layers with carried state and no Engram lookup, the fused prologue also handles
the input hyperconnection and Q/KV projections. Flare still handles rotary
transforms, indexed attention and KV cache updates. Prefill, draft computation,
and verification shapes outside the gate stay on Flare/b12x. Isolated 2- and
4-row trials regressed with fusion, so those shapes were left alone. A physical
row is a row of the captured verification tensor, which can include pending
inputs, draft candidates and padding; **5-8 rows does not mean 5-8 requests**.

**Versus vLLM BRRR:** The [original BRRR launcher](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR)
mounts the native checkpoint directly and adds vLLM-side decode hooks, including
its own sparse attention and indexer kernels. It defaults to SSD Engrams,
16 sequences, DSpark depth 7 and a CUDA graph capture ceiling of 128. The
original fused suffix supports up to 128 eligible decode rows, with a fused
prologue for up to eight. The [vLLM BRRR/B12X hybrid](../deepseek-v41-brrr-b12x-hybrid-tp4/)
keeps that integration but sets `BRRR_FUSED_MAX_ROWS=8`, so larger scheduled
decode batches use the inherited B12X layer path instead of the fused decoder.
This recipe borrows the BRRR prologue and suffix, not the vLLM
hooks, BRRR attention kernels or DSpark scheduler. The choice is made during
Ignition export and saved in the prepared graphs. The Flare serving binary
and checkpoint files are unchanged; no Python callbacks or recompilation are
needed for requests. The fused activation arithmetic is not bit-identical to
Flare's unfused path.

### Launch Settings

| Setting | Published Flare D7 example | Unfused Flare D4 control and this recipe | vLLM BRRR/B12X hybrid |
| --- | --- | --- | --- |
| Drafting and verification | D7, adaptive, alpha 0.6, up to 128 packed rows | D4, adaptive, alpha 0.6, up to 128 packed rows | DSpark **4 / 3 / 2** at active decode batches **1 / 2-4 / 5-32**; probabilistic drafting, block rejection, adaptive verification off |
| Scheduling | Balanced, 32 active requests, 4,096-token prefill chunks | Balanced, 32 active requests, 4,096-token prefill chunks | Async, 32 sequences, 4,096 batched tokens per step |
| CUDA graphs | Prebuilt packed catalog | Locally exported packed catalog, speculative graph budget 64, four image slots | Breakable full and piecewise graphs, capture ceiling 256 |
| Engrams and prefix cache | Disk-backed Engrams, 64 GiB host prefix cache in the example | `--embedding-table-storage ram`, `--prefix-cache-memory 0`; GPU prefix cache remains enabled | Locked host RAM via the hybrid's row reader, inherited vLLM prefix caching |
| Shared GPU KV | `--kv-cache-limit memory` | `--kv-cache-limit memory`; **16,485,888 unfused** vs **14,957,568 fused** tokens observed | FP8 GPU-only KV, `--gpu-memory-utilization 0.96`; **7,832,679** tokens in the linked recipe's run |

The Flare launcher's `--mtp-max-verification-token-budget 128` limits packed
rows across requests, including pending inputs. Adaptive verification chooses
which draft prefixes to verify within that limit; it is not the hybrid's
batch-size-based draft schedule. The Flare export retains BF16 MLA compute, an
NVFP4 draft head and an FP8 Markov head. These are not runtime overrides for a
prebuilt package. See the [Flare option reference](https://docker.local-inference-lab.ai/flare/options)
for what can be changed at launch.

The fused export's additional packed layouts and scratch consume GPU memory.
Its observed KV pool is about **9.3% smaller** than the unfused D4 control's
and about **1.9x** the vLLM hybrid's published pool. All three figures are
shared token capacities, not per-request limits or throughput comparisons.

## Measured Results

November 7, 2026, using the fused Flare D4 profile. The runtime reported
**14,957,568 shared KV tokens** and a 1,048,576-token per-request context limit.
The shared pool is not a million tokens for every active stream.

| Component | Benchmark configuration |
| --- | --- |
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each, TP4 on one host |
| Power | 395 W configured limit per GPU, not measured draw; benchmark did not record clock offsets |
| Interconnect | PCIe 5.0 x16, no NVLink; all four GPUs on one NUMA node. Driver reported read/write peer access for all 12 directed GPU pairs |
| CPU and RAM | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, about 251 GiB Linux-visible RAM |
| Storage | Local NVMe RAID 0 (ext4) for checkpoint and package; live Engram lookups use host RAM |
| System | Ubuntu 24.04 LTS, NVIDIA driver 615.71.09 |

Aggregate decode tok/s, measured in 30-second sustained windows. **C1 is
mean ± sample SD over ten clean windows per context**; other columns are
single measurements.

| Context | C1 mean ± SD | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 363.67 ± 11.16 | 571.27 | 851.07 | 1,143.46 | 1,472.03 | 2,262.96 |
| 16K | 354.91 ± 9.69 | 573.72 | 820.78 | 1,124.26 | 1,495.02 | 2,242.34 |
| 32K | 354.13 ± 16.58 | 573.05 | 835.47 | 1,114.92 | 1,443.53 | 2,238.38 |
| 64K | 348.22 ± 7.94 | 560.14 | 797.61 | 1,114.54 | 1,445.17 | 2,181.72 |
| 128K | 345.34 ± 9.60 | 554.95 | 777.37 | 1,089.92 | 1,430.73 | 2,115.71 |
| 256K | 329.78 ± 6.93 | 513.36 | 759.15 | 1,076.40 | 1,342.51 | 2,000.30 |

All 36 full-matrix cells sustained the requested active request count with
no queueing. At nominal 256K/C32, measured input averaged 256,882 tokens per
stream. One additional C1 pass replaced a 64K repetition-guard failure. The
first ten valid windows per context are included without speed-based trimming.
The [numeric results](results/2026-11-07.json) include the individual C1 rates
and prefill scouts. Prefill scouts stop at 128K in this client; the 256K decode
cells were measured.

These measurements are from the original local export of this profile. They do
not establish model-quality equivalence or validate million-token requests.
The publication scripts reproduce the pinned settings, but a rebuild may
select different collective profiles or have different performance.

## Reproduce the benchmark

Use a quiet endpoint with no competing traffic. This takes about 70 minutes
on the tested setup. The runner verifies and uses the pinned
[LLM Inference Bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/52216d70865a015b34689626d04a80b89c9dc45f)
client with native Flare metrics. It runs the full matrix once, then C1-only
repeats, using the 8,192-token output cap, a three-second decode warmup,
default sampling and the repetition guard. Invalid windows remain in the raw
logs; only clean windows count toward the C1 statistics.

```bash
python3 -m venv "$WORK_DIR/bench-venv"
"$WORK_DIR/bench-venv/bin/pip" install httpx rich psutil nvidia-ml-py
"$WORK_DIR/bench-venv/bin/python" tools/benchmark.py \
  --port "${HOST_PORT:-1230}" --output-dir "$WORK_DIR/bench-results/$(date +%Y%m%d-%H%M%S)"
```

Keep raw outputs local. See [NOTICE.md](NOTICE.md) for provenance and licensing.
