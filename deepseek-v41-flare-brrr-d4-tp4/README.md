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

The build fetches [Flare at `972e28e`](https://github.com/NVIDIA/hyperflare-inference/commit/972e28e44781d8d0c84d70ae434db92c36076a43),
applies [the BRRR preparation patch](patches/flare-brrr-d4.patch), and uses its
locked dependencies, including Torch `2.14.1+cu132`, FlashInfer `1f8ed8a3`, and
the pinned CUDA 13.2.1 development image. The FlashInfer pin masks unused KV
operands. The [expert-reuse workload patch](patches/flashinfer-route-reuse.patch)
adds whole-batch shared-expert cases to b12x autotuning without changing
inference routing or the checkpoint. The build verifies the workload module's
Git blob before and after patching. Flare shares startup-tuning storage across
the prepared graphs. The preparation image applies Flare's bundled PyTorch
and Triton fixes during the build so export can run as the host user.

Export uses native expert scales, BF16 MLA compute, the NVFP4 draft head, FP8
Markov head, 4,096-token prefill chunks, 32 request slots and 128 packed
verification rows. Its runtime collective ladder comes from the pinned Flare
tree. The attention and tuning changes are captured at export, so changing the
serving image alone does not reproduce the results below. Use a separate
`WORK_DIR` when upgrading so the previous package and runtime remain available.

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
Ignition export and saved in the prepared graphs. The recipe does not modify
the pinned serving binary or checkpoint files. Requests require no Python
callbacks or recompilation. The fused activation arithmetic is not bit-identical
to Flare's unfused path.

### Launch Settings

| Setting | Published Flare D7 example | Unfused Flare D4 control and this recipe | vLLM BRRR/B12X hybrid |
| --- | --- | --- | --- |
| Drafting and verification | D7, adaptive, alpha 0.6, up to 128 packed rows | D4, adaptive, alpha 0.6, up to 128 packed rows | DSpark **4 / 3 / 2** at active decode batches **1 / 2-4 / 5-32**; probabilistic drafting, block rejection, adaptive verification off |
| Scheduling | Balanced, 32 active requests, 4,096-token prefill chunks | Balanced, 32 active requests, 4,096-token prefill chunks | Async, 32 sequences, 4,096 batched tokens per step |
| CUDA graphs | Prebuilt packed catalog | Locally exported packed catalog, speculative graph budget 64, four image slots | Breakable full and piecewise graphs, capture ceiling 256 |
| Engrams and prefix cache | Disk-backed Engrams, 64 GiB host prefix cache in the example | `--embedding-table-storage ram`, `--prefix-cache-memory 0`; GPU prefix cache remains enabled | Locked host RAM via the hybrid's row reader, inherited vLLM prefix caching |
| Shared GPU KV | `--kv-cache-limit memory` | `--kv-cache-limit memory`; **16,464,000 fused** tokens observed | FP8 GPU-only KV, `--gpu-memory-utilization 0.96`; **7,832,679** tokens in the linked recipe's run |

The Flare launcher's `--mtp-max-verification-token-budget 128` limits packed
rows across requests, including pending inputs. Adaptive verification chooses
which draft prefixes to verify within that limit; it is not the hybrid's
batch-size-based draft schedule. The Flare export retains BF16 MLA compute, an
NVFP4 draft head and an FP8 Markov head. These are not runtime overrides for a
prebuilt package. See the [Flare option reference](https://docker.local-inference-lab.ai/flare/options)
for what can be changed at launch.

The November 9 fused export had **10.1% more shared KV** than the November 7
fused export (14,957,568 tokens). It nearly matches the earlier unfused D4
control's 16,485,888 tokens and is about **2.1x** the vLLM hybrid's published
pool. These are shared capacities, not per-request limits or throughput
comparisons. The comparison does not isolate a single preparation change.

See the [BRRR D7 experiment](D7-EXPERIMENT.md) for the same-power comparison
with a separate seven-draft export and why D4 stayed the default.
The [stock Flare experiment](STOCK-FLARE-EXPERIMENT.md) compares stock D7,
fused D7, and fused D4 at 450 W per GPU.

## Measured Results

The November 9 (395 W) and November 10 (450 W) tests used the same running
`972e28e` Flare BRRR D4 deployment and launch settings. Both runs reported
**16,464,000 shared KV tokens** and a 1,048,576-token per-request limit. The
shared pool is not a million tokens for every active stream.

| Component | Benchmark configuration |
| --- | --- |
| GPUs | 4x NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB each, TP4 on one host |
| Power | 395 W or 450 W configured limit per GPU, not measured draw. All 188 power checks in the 450 W run matched that limit on all four serving cards. |
| Interconnect | PCIe 5.0 x16, no NVLink; all four GPUs on one NUMA node. Driver reported read/write peer access for all 12 directed GPU pairs |
| CPU and RAM | AMD Ryzen Threadripper PRO 7965WX, 24 cores / 48 threads, about 251 GiB Linux-visible RAM |
| Storage | Local NVMe RAID 0 (ext4) for checkpoint and package; live Engram lookups use host RAM |
| System | Ubuntu 24.04 LTS, NVIDIA driver 615.71.09 |

Aggregate decode tok/s, measured in 30-second sustained windows. **C1 is
mean ± sample SD over the first ten valid windows per context**, selected in
chronological order without speed-based trimming. C2 through C32 are single
valid windows.

### 395 W Per GPU

| Context | C1 mean ± SD | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 357.33 ± 17.03 | 604.46 | 823.84 | 1,119.47 | 1,635.63 | 2,222.72 |
| 16K | 352.65 ± 13.85 | 575.01 | 794.09 | 1,112.62 | 1,617.16 | 2,222.51 |
| 32K | 354.65 ± 13.79 | 594.95 | 786.61 | 1,110.62 | 1,637.72 | 2,194.78 |
| 64K | 351.33 ± 14.55 | 581.17 | 814.40 | 1,106.96 | 1,544.00 | 2,192.39 |
| 128K | 337.50 ± 8.47 | 560.36 | 774.85 | 1,080.25 | 1,561.74 | 2,093.67 |
| 256K | 330.14 ± 10.41 | 530.17 | 778.83 | 1,053.76 | 1,518.02 | 2,088.08 |

At 395 W, compared with the [November 7 results](results/2026-11-07.json), C16 was
**6.8% to 13.5% faster** across these contexts, C1 means stayed within 2.3%,
and 256K/C32 was 4.4% faster. This is a package comparison on the tested host,
not a claim of higher throughput at every concurrency.

All 36 cells in the [395 W results](results/2026-11-09.json) validated after
rerunning 128K/C8, whose initial window had an extra active request. Four C1
windows were rejected: two for extra active requests and two for repetition.

### 450 W Per GPU

| Context | C1 mean ± SD | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 364.45 ± 21.30 | 632.32 | 842.44 | 1,145.03 | 1,680.14 | 2,266.83 |
| 16K | 358.31 ± 16.10 | 620.17 | 856.66 | 1,146.24 | 1,616.45 | 2,250.69 |
| 32K | 354.68 ± 14.35 | 569.32 | 815.44 | 1,118.26 | 1,572.19 | 2,239.50 |
| 64K | 351.33 ± 6.96 | 595.65 | 830.11 | 1,128.23 | 1,583.15 | 2,252.78 |
| 128K | 344.80 ± 19.02 | 563.50 | 811.13 | 1,090.78 | 1,562.54 | 2,136.42 |
| 256K | 334.91 ± 6.37 | 523.24 | 775.70 | 1,098.84 | 1,545.01 | 2,092.67 |

All 36 cells in the [450 W results](results/2026-11-10-450w.json) validated.
The initial 0K/C1 window was rejected and replaced because two requests were
briefly active. The endpoint remained healthy and the deployment did not change.

Compared with 395 W, **C32 was 0.2% to 2.8% faster** across the six contexts and
C1 means stayed within 2.2%. The largest observed prefill difference was the
single 128K scout: **20.6K versus 18.9K tok/s (+8.8%)**. These are observed
differences, not evidence of a gain at every concurrency or a power-efficiency
comparison.

Both numeric reports include individual C1 rates, per-cell native token rates,
rejection reasons and prefill scouts through 128K. At nominal 256K/C32, the
395 W input averaged 256,882 tokens per stream. The measurements do not establish
model-quality equivalence or validate million-token requests. Rebuilding the
pinned recipe may select different kernel or collective profiles and yield
different rates.

### Tetris Generation At 450 W

The [Tetris runner](https://www.voipmonitor.org/tetristest.py) made three
isolated requests. End-to-end throughput includes reasoning and averaged
**540.04 ± 3.01 tok/s** (mean ± sample SD across all three requests). This is a
separate workload, not a sustained-decode rate. No matched 395 W Tetris result
is reported.

| Run | Completion tokens | Elapsed | End-to-end tok/s | Desktop check |
| --- | ---: | ---: | ---: | --- |
| 1 | 46,168 | 85.0 s | 543.39 | Passed |
| 2 | 40,628 | 75.6 s | 537.56 | Passed |
| 3 | 45,639 | 84.6 s | 539.16 | Canvas color error after starting |

The games were checked without modifying the generated HTML. Two of three
passed the desktop browser check. The generated HTML is not published here.

### Correctness

The November 9 package passed Estonia 10/10 separately at C1, C2 and C10,
four concurrent vision cases, JSON schema, tool-call round trips and request
transitions. The [Estonia 30 and LAVD 30 record](https://github.com/shihanqu/RTX-Pro-6000-Recipes/blob/1d9e854980e02ca07116b6632b869df9bc702410/deepseek-v41-flare-brrr-d4-tp4/README.md#correctness-at-c30)
applies to the November 7 package; C30 correctness was not rerun for this build.

## Reproduce the benchmark

Use a quiet endpoint with no competing traffic. Run each power setting
separately and verify the configured limits; this recipe does not set power
limits. Allow about 80 minutes per run on the tested setup. The runner verifies
and uses the pinned
[LLM Inference Bench v0.8.0](https://github.com/local-inference-lab/llm-inference-bench/tree/52216d70865a015b34689626d04a80b89c9dc45f)
client with native Flare metrics. It runs the full matrix once, then C1-only
repeats, using the 8,192-token output cap, a three-second decode warmup,
default sampling and the repetition guard. Invalid windows remain in the raw
logs; only clean windows count toward the C1 statistics. A flagged higher-
concurrency cell makes the runner exit with status 2. Rerun that cell on the
same quiet endpoint with the pinned client and unchanged measurement settings.

```bash
python3 -m venv "$WORK_DIR/bench-venv"
"$WORK_DIR/bench-venv/bin/pip" install httpx rich psutil nvidia-ml-py
"$WORK_DIR/bench-venv/bin/python" tools/benchmark.py \
  --port "${HOST_PORT:-1230}" --output-dir "$WORK_DIR/bench-results/$(date +%Y%m%d-%H%M%S)"
```

Keep raw outputs local. See [NOTICE.md](NOTICE.md) for provenance and licensing.
