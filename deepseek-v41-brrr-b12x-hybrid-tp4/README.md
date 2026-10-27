# DeepSeek-V4.1 BRRR/B12X Hybrid, TP4

The original, validated **BRRR/B12X hybrid** for **DeepSeek-V4.1-Flash-1017** on
**four RTX PRO 6000 Blackwell 96 GB** cards: **32 sequences**, **1M configured
context**, **DSpark 2**, **FP8 GPU KV**, and **NVMe Engram offload**.

Observed default-reasoning decode: **about 350-400 tok/s at C1, 1,250 at C16 and
1,750 aggregate at C32**. These are not BRRR's reasoning-off code-generation
numbers, nor a promise for every prompt. Speculative acceptance changes token rates.

## Results First

Latest completed full sweep, 2026-10-27, llm-inference-bench **0.6.2**. Aggregate
sustained decode tok/s, 30-second cells, 3-second warmup, request cap 8192,
server-default sampling/reasoning. Context labels are nominal, estimated targets.

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 341.9 | 558.9 | 634.0 | 915.4 | 1244.2 | 1755.0 |
| 16K | 331.6 | 540.4 | 605.4 | 901.9 | 1237.7 | 1732.0 |
| 32K | 324.7 | **Invalid** | 594.7 | 901.0 | 1204.9 | 1713.6 |
| 64K | 315.3 | 511.2 | 589.3 | 892.6 | 1211.0 | 1694.1 |
| 128K | 291.2 | 486.8 | 553.2 | 849.3 | 1160.9 | 1604.5 |

**29/30 usable cells.** C2/32K triggered exact repetition and is not assigned a
throughput. Five later standalone repeats were clean at 543.4, 527.3, 543.4,
531.9 and 540.1 tok/s. Their different warmup/order does **not** establish a fix;
they do not replace the failed matrix cell.

Valid cells maintained the requested sampled concurrency, no queue/underfill or
errors, and client/server token counters agreed within 0.146%. C1 renewed requests
at the 8192 cap: these are sustained renewing-stream measurements, not strict
single-request/no-renewal tests. Sampling between polls cannot rule out a brief overlap.

Separate strict 0K checks used a 32768 cap at 30 seconds, and 65536 at 60 seconds,
to avoid renewals. A valid independent 60-second confirmation gave:

| C1 | C4 | C16 | C32 |
| ---: | ---: | ---: | ---: |
| 386.1 | 635.5 | 1260.4 | 1754.7 |

One other forced ignore-EOS 60-second C1 run repeated after about 21.6K tokens.
The guard rejected it; it is excluded. The unchanged retry passed. Ordinary
EOS-stopped inference finished normally. No repetition guard was disabled.

### Correctness And Prefill

| Test | Conditions | Result |
| --- | --- | --- |
| Estonia | 30 requests, C30, explicit maximum reasoning | **30/30 pass**, no truncation/error/stall/timeout |
| LAVD | 30 requests, C30, explicit maximum reasoning | **30/30 exact**, no truncation/error/stall/timeout |
| Vision | Four concurrent image OCR/color requests | **4/4** |
| Capacity | 32 real active streams, plus arithmetic requests | **32 active**, **32/32 arithmetic** |
| API features | Both model aliases, JSON schema, tool round trip | Passed |

Quality profiles used seed base 20261026 (plus run index), normal EOS, default
temperature/top_p, output caps 40000 (Estonia) and 100000 (LAVD). The decode
matrix did **not** request maximum reasoning. These are synthetic checks, not a
general model-quality guarantee; the million-token configured limit was not fully tested.

Prefill scouts from that same sweep, one sample per context, **client prompt tokens / TTFT**:

| Nominal | Actual tokens | TTFT (s) | tok/s |
| --- | ---: | ---: | ---: |
| 8K | 8189 | 0.434 | 18864 |
| 16K | 16211 | 0.878 | 18474 |
| 32K | 32254 | 1.757 | 18357 |
| 64K | 64353 | 3.653 | 17616 |
| 128K | 128520 | 7.928 | 16211 |

These are not repeated standalone cold-prefill trials. Prefix caching can be
reused, and the estimate-mode calibration cache was not retained. Numeric-only
records are in [validation/](validation/); no raw saved completions or machine telemetry are published.

## Requirements

| Component | Tested / required |
| --- | --- |
| GPUs | 4x RTX PRO 6000 Blackwell Workstation Edition, 96 GB, PCIe Gen5 x16, no NVLink |
| Fabric | All twelve directed GPU peer-access pairs available; selected GPUs on one NUMA node, mixed NODE/PHB relationships |
| Power | Measured at 395 W per selected card; scripts never change power limits |
| CPU/RAM | 24-core Threadripper PRO, 256 GB RAM; not a validated minimum-RAM claim |
| Storage | Complete 510,311,045,295-byte checkpoint (~475 GiB); fast local ext4/XFS NVMe for the two ~101 GB Engram files; measured on 2x PCIe Gen5 NVMe striped RAID0 |
| Runtime | NVIDIA driver **615.65+** (tested 615.71.09), Docker Engine 29.6.1, Compose 5.3.1, NVIDIA Container Toolkit 1.19.1 |
| OS | x86_64 Ubuntu 24.04, kernel 7.0.0-38; required host configuration below |
| Disk overhead | Reference image ~12.9 GB, plus build layers and persistent JIT/autotune cache; allow tens of GB extra |
| Host tools | Python 3.12 recommended, venv/pip, git, Docker, nvidia-smi, ss, findmnt |

The Engram table stays on disk; this does not allocate its full ~203 GB in host
RAM. CPU scale/row/control buffers are pinned. KV is **GPU-only**, not an SSD/host
KV connector. All four selected GPUs must be free. Other services were present
on the test host, but their identities and telemetry are intentionally omitted.

## Quick Start

```bash
git clone https://github.com/shihanqu/RTX-Pro-6000-Recipes
cd RTX-Pro-6000-Recipes/deepseek-v41-brrr-b12x-hybrid-tp4
cp .env.example .env
$EDITOR .env
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

./build.sh                    # pinned upstream source + base image, then the tiny guard patch
./download_checkpoint.sh      # ~510 GB from the original publisher; verifies every byte
./launch.sh check             # read-only host/fabric/idle-device/image/checkpoint checks
./launch.sh                   # starts only this recipe, waits for readiness and graph verification
```

**Before launching, perform the deliberate host setup below.** The scripts do
not edit boot/module settings, reboot, change power, or stop another model.
The default API bind is loopback on port 8000, without authentication. Secure it
separately before choosing a LAN/public bind.

Download authentication, if required, is the reader's own Hugging Face login.
No credentials are shipped or mounted into the serving container. Files already
present can be checked with `./download_checkpoint.sh --verify`. Use a complete
real-file directory, not an HF cache snapshot whose symlinks point outside the
read-only `/model` bind. Hardlinks within a filesystem are fine; never modify shared weights in place.

The build downloads and verifies 35 BRRR source files. It builds BRRR unchanged
from the pinned LIL base image, then applies the dispatch-only patch with exact
before/after SHA-256 gates. It does not rely on an unpublished local image. Local
image IDs can differ across rebuilds; the runtime/code/checkpoint pins are authoritative.

The public package was built successfully with cached upstream compilation. Its
2,890 BRRR/vLLM source files, effective arguments and runtime environment match
the measured hybrid (apart from generic device selection and API port). Compose,
host prerequisites, pinned benchmark CLI and 15 unit tests passed. This is a
packaging/equivalence check, not a second performance run on a freshly launched
container; see [validation/packaging.json](validation/packaging.json).

First build/start compiles kernels and can take considerably longer than a warm
start. Keep the runtime cache. The launcher prints the actual KV pool, usually
around **8.1M shared logical tokens** on the reference setup. It is not 8.1M per
stream: 32 concurrent slots do not imply 32 simultaneous full-million-token requests.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4.1-flash",
  "messages":[{"role":"user","content":"Prove that the first n positive odd integers sum to n squared."}],
  "max_tokens":2000
}'
```

Both `DeepSeek-4.1-Flash-1017-BRRR` and `deepseek-v4.1-flash` are served. Reasoning
is enabled by default and consumes completion tokens. Use `./launch.sh logs`,
`./launch.sh stop`, or `./launch.sh config`. Matching healthy instances are reused;
occupied ports/GPUs or foreign containers are refused, never automatically killed.

## Host Setup And Security

**Host-wide, driver-specific configuration; maintenance and reboot required.**
Disabling IOMMU removes DMA isolation and can break passthrough/virtualization.
The NVIDIA registry overrides are experimental, not a general NVIDIA support
recommendation. Disabling HMM affects other CUDA workloads. Assess your platform,
keep backups and revalidate other services; do not apply blindly to a production host.
The container also inherits host IPC, unlimited memory locking and an unconfined
seccomp profile from the serving recipe; run it only on a trusted inference host.

The following was loaded for the reported results. BRRR upstream documents
IOMMU-off/P2P prerequisites; the extra registry/UVM settings are a host-specific
part of this tested deployment, **not** a claim that BRRR itself introduced them.

```text
# Kernel command line, preserving all other existing options:
iommu=off amd_iommu=off intel_iommu=off

# /etc/modprobe.d/rtx-pro-p2p.conf
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"

# /etc/modprobe.d/rtx-pro-uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

On GRUB/initramfs-based distributions, edit the existing configuration deliberately,
then run the distribution's `update-grub` and `update-initramfs -u`, and reboot.
Enable the board's appropriate Above-4G/Resizable BAR settings as necessary.
Load UVM (`sudo modprobe nvidia_uvm`) before checking its parameter if it is not yet loaded.

Verify the **loaded** settings, not just files on disk:

```bash
ls /sys/kernel/iommu_groups | wc -l       # reference: 0
cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm  # reference: Y
grep -E 'RegistryDwords|EnableResizableBar' /proc/driver/nvidia/params
nvidia-smi topo -m
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

The launcher also checks CUDA peer access. It does not prove PCIe link bandwidth;
confirm Gen5 x16 under load and measure the fabric/storage if throughput is low.
Reference BAR1 was 128 GiB per card. Other pre-existing boot options were
`pcie_aspm=off transparent_hugepage=madvise numa_balancing=disable idle=nomwait`;
they were not isolated in an A/B test and are not asserted necessary.

In a matched stock-image check, applying the registry/UVM changes together moved
default-reasoning C1/C16 from about 153/455 to 305/1064 tok/s. That is an observed
combined host effect, not attribution to any individual option. A normal-looking
GPU utilization number is not proof of correct peer access.

## Exact Serving Settings

| Setting | This recipe |
| --- | --- |
| Checkpoint | Native `deepseek-ai/DeepSeek-V4.1-Flash-1017` revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, 83 verified files |
| Base runtime | LIL vLLM image digest in `pins.json`; CUDA 13.4, reference vLLM `0.1.dev21836+g6d66eed35`, Torch `2.14.0a0` NVIDIA 26.08 |
| Kernel stack | Pinned image's FlashInfer `0.7.1+lil.cu134.sm120.g79819b112f8b`, B12X, Triton 3.8.0, CuTe DSL 4.7.1; do not casually upgrade |
| Parallelism | TP4, inherited expert layout, DCP1 |
| Hybrid dispatch | `BRRR_FUSED_MAX_ROWS=8`; above it, inherited native B12X path |
| Scheduling | 32 sequences; 1,048,576-token per-request cap; 4096 batch-token limit; async scheduling |
| Memory/KV | GPU utilization 0.96, FP8 KV, GPU-only KV; block size 256, SWA block size 128 |
| Graphs | `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, FULL_AND_PIECEWISE, capture ceiling 256; no eager or language-model-only flag |
| Draft | DSpark 2; probabilistic draft, block rejection, adaptive verification off, B12X draft attention/MoE |
| Sampling | Inherited temperature 1.0/top_p 1.0; default reasoning retained; normal EOS for ordinary requests |
| Features | Vision, auto tool choice, DeepSeek-V4.1 reasoning/tool parsers, prefix cache behavior inherited |
| NCCL/PCIe | Inherited SYS peer level/read enable; LL/LL128/Simple; 16 channels; 2 MiB buffers; B12X + BRRR phased collectives |

The entrypoint and low-level environment come from checksum-verified BRRR source,
not hidden host scripts. `compose.yaml` contains the changed flags. At DSpark 2,
four users already schedule at least twelve verify-token rows before padding,
so this is **not** a threshold of eight users. Keep graph capacity sufficient
for sequence count, speculative expansion and padding; changing the depth or graph cap requires rebenchmarking.

## Validation Commands

```bash
./validate.sh prepare  # fetch exact public commit whose benchmark bytes match the recorded SHA
./validate.sh smoke    # aliases, JSON, tools, C4 OCR/color, C32 streams and arithmetic
./validate.sh quick    # smoke + Estonia 10, default reasoning
./validate.sh decode   # strict no-renewal 0K C1/C4/C16/C32, 30 s, cap 32768
./validate.sh confirm  # independent 60 s windows, cap 65536; guards remain enabled
./validate.sh sweep    # reproduce the full nominal 0-128K x C1/C2/C4/C8/C16/C32 sweep
./validate.sh quality  # LAVD30 then Estonia30, both C30 and maximum reasoning
```

The full sweep uses an 8192 cap and integrated prefill scouts, matching the
recorded method. Do not substitute standalone-prefill or exact token targeting
and call it the same measurement. Output goes to private, ignored `results/`.
The wrapper disables only self-update prompts and verifies source bytes/commit;
it does not change generation, timing, loop guards or scoring. Quality reports
save text for reproducibility and can repeat personal details in the public
benchmark inputs: **do not publish raw reports**. Only allowlisted numeric summaries belong in a public recipe.

## Origins

### What Changed Versus BRRR

Direct parent: [BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR),
revision `0bc4dbd0a15458e0957b899d72c7a5089e3335a5`.

| Area | Pinned BRRR recipe | This fork |
| --- | --- | --- |
| Decoder eligibility | Fused decoder up to its compiled 128-token limit | Environment-controlled ceiling, set to **8 scheduled tokens**; native B12X above it |
| CUDA graphs | Breakable off, capture ceiling 128; runtime could resolve to FULL_DECODE_ONLY | Breakable on, ceiling **256**, full and piecewise capture verified |
| Sequences | 16 | **32**, actual simultaneous execution verified |
| Draft depth | DSpark 7 | **DSpark 2**, less unproductive verify work on this reasoning workload |
| API naming | Official BRRR model ID | Preserve it and add `deepseek-v4.1-flash` compatibility alias |
| Deployment | Publisher's serve script | Generic paths/devices, safe launcher, verified downloads/builds and repeatable validation |
| Host | Upstream IOMMU/P2P guidance | Explicit loaded host settings above, with risks and no automatic mutation |

**No new checkpoint, quantization, training, CUDA kernel arithmetic or tokenizer
was introduced.** The only Python decoder change is the tiny guard patch. Major
fused kernels, attention, communication, sharded sampling, prompt fixes and SSD
Engram reader are **BRRR's work**, not new inventions of this configuration fork.
The original hybrid excludes later Sonar/Aphrodite, row-cache, tensor-core Markov,
and dense split-K flag experiments; they did not show a reliable gain in the matching tests.

### The Giants Under BRRR

- **DeepSeek:** native model, tokenizer, vision and draft-head work; original checkpoint above.
- **Fraser Price / BRRR:** persistent fused decode/prologue kernels, SSD Engram
  pipelining, specialized attention/top-k/projections, phased PCIe communication,
  vocabulary-sharded drafting/verification and tool-result prompt fixes.
- **[Local Inference Lab's vLLM fork](https://github.com/local-inference-lab/vllm):**
  the pinned base runtime, model-specific integration and DSpark/B12X deployment support.
- **[vLLM](https://github.com/vllm-project/vllm):** scheduler, API serving, KV and graph infrastructure.
- **[b12x authors](https://github.com/local-inference-lab/b12x):** SM120 attention,
  MoE, linear, normalization, Engram and collective kernels with preparation/autotuning.
- **[FlashInfer](https://github.com/flashinfer-ai/flashinfer):** the home of b12x
  development and broader CUDA inference infrastructure, by its community and NVIDIA.
- **[PyTorch](https://github.com/pytorch/pytorch), [NVIDIA CUTLASS/CuTe](https://github.com/NVIDIA/cutlass),
  CUDA and [Linux/liburing](https://github.com/axboe/liburing):** tensor/compiler,
  GPU programming and asynchronous storage foundations.

All retain their authorship and own terms. See [NOTICE.md](NOTICE.md), including
the required original BRRR notice. We link to publishers rather than mirror weights/images.

## Privacy And Troubleshooting

Examples use generic device indices and `/srv` paths. `.env`, build/state/cache
artifacts, raw results and logs are ignored. No hostnames, private network
addresses, GPU UUIDs/serials, account home paths or credentials are in the public
results. Runtime tools necessarily inspect local device IDs; keep their output private.
Public author/project attribution and mandatory copyright notices are retained.

- Busy GPU/port or wrong container: choose free resources yourself; no automatic stop/kill occurs.
- Slow first start: preserve JIT caches and allow the readiness timeout; do not claim a stalled engine healthy.
- Missing host setting: follow the deliberate maintenance procedure; the script will not apply it for you.
- No full/piecewise graph capture: inspect private startup logs; do not silently substitute eager mode.
- Unexpected low speed: verify loaded P2P settings, all twelve peer pairs, Gen5 x16 under load and actual NVMe latency.
- Repetition during forced long decode: retain the failed cell and rerun unchanged if desired. Never disable the guard or replace an invalid matrix cell with a convenient repeat.
