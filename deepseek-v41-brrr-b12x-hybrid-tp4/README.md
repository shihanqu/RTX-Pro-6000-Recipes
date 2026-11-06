# DeepSeek-V4.1 BRRR/B12X Hybrid, TP4

The **BRRR/B12X hybrid** for **DeepSeek-V4.1-Flash-1017** on **four RTX PRO 6000
Blackwell 96 GB** cards, with **RAM-resident Engrams**, **batch-adaptive DSpark
4/3/2**, **32 sequences**, a **1,048,576-token per-request limit**, and **FP8 GPU KV**.

At fresh context the measured rates are **399.8 tok/s at C1** and **1,823.9
aggregate tok/s at C32** with default reasoning. The reference had **7,832,679
shared GPU KV tokens**. Speculative acceptance and prompt length affect throughput.

## Results First

October 28, 2026 full sweep using llm-inference-bench **0.6.2** on the RAM-Engram
image. Aggregate sustained decode tok/s: 30-second cells, 3-second initial C1
warmup at the largest tested context, 8192 output-token cap per request, and
server-default sampling and reasoning. Context labels are nominal targets; the
0K row includes 91 actual prompt tokens.

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0K | 399.8 | 580.6 | 673.9 | 958.0 | 1283.2 | 1823.9 |
| 16K | 371.6 | 589.8 | 660.0 | 942.6 | 1252.2 | 1795.4 |
| 32K | 367.7 | 551.5 | 665.8 | 918.9 | 1257.0 | 1746.1 |
| 64K | 341.1 | 526.9 | 642.1 | 891.8 | 1225.3 | 1722.0 |
| 128K | 346.4 | 522.6 | 623.5 | 885.9 | 1187.7 | 1654.2 |

**30/30 usable cells.** The sweep started after at least three minutes idle.
All cells maintained the requested sampled concurrency, including 32 active
streams, with no queueing, underfill, request errors or detected repetition.
Client and server counters agreed within 0.126%. The repetition guard stayed
enabled. C1 and C2 renewed requests at the output cap; these are sustained
stream rates, not strict single-request tests. The benchmark ignores EOS for
decode timing; ordinary requests still stop normally.

### Correctness And Prefill

| Test | Conditions | Result |
| --- | --- | --- |
| Estonia | 30 requests, C30, explicit maximum reasoning | **30/30 pass**, no truncation/error/stall/timeout; timing not isolated |
| LAVD | 30 requests, C30, explicit maximum reasoning | **30/30 exact**, no truncation/error/stall/timeout |
| Text and vision | Smoke test on the RAM-Engram image | Passed |
| Engram residency | Two layers on each of four TP ranks | **8/8 locked and fully resident**; preload file descriptors closed |

Quality profiles used seed base 20261026 (plus run index), normal EOS, default
temperature/top_p, output caps 40000 (Estonia) and 100000 (LAVD). The decode
matrix did **not** request maximum reasoning. Background traffic resumed during
Estonia (31 sampled active requests), so its scores are reported without a C30
timing claim. These are synthetic checks; the full configured context was not tested.

Prefill scouts from that same sweep, one sample per context, **client prompt tokens / TTFT**:

| Nominal | Actual tokens | TTFT (s) | tok/s |
| --- | ---: | ---: | ---: |
| 8K | 8190 | 0.415 | 19722 |
| 16K | 16212 | 0.831 | 19506 |
| 32K | 32255 | 1.704 | 18928 |
| 64K | 64354 | 3.550 | 18127 |
| 128K | 128521 | 7.719 | 16650 |

These are single integrated scouts, not repeated standalone cold-prefill trials.
Prefix caching can be reused. The [results record](validation/observed.json)
contains numeric summaries and the source report checksum, not prompts,
saved completions or machine telemetry.

## Requirements

| Component | Tested / required |
| --- | --- |
| GPUs | 4x RTX PRO 6000 Blackwell Workstation Edition, 96 GB, PCIe Gen5 x16, no NVLink |
| Fabric | All twelve directed GPU peer-access pairs available; selected GPUs on one NUMA node, mixed NODE/PHB relationships |
| Power | Measured at 395 W per selected card; scripts never change power limits |
| CPU/RAM | Tested on a 24-core Threadripper PRO with 256 GB installed. The launcher requires 220.06 GiB of available RAM before starting; check actual free headroom, not installed capacity |
| Storage | Complete 510,311,045,295-byte checkpoint (~475 GiB) on local ext4/XFS supporting direct I/O. Fast NVMe helps startup; Engram lookup no longer reads from SSD |
| Runtime | NVIDIA driver **615.65+** (tested 615.71.09), Docker Engine 29.6.1, Compose 5.3.1, NVIDIA Container Toolkit 1.19.1 |
| OS | x86_64 Ubuntu 24.04, kernel 7.0.0-38; required host configuration below |
| Disk overhead | Reference image ~12.9 GB, plus build layers and persistent JIT/autotune cache; allow tens of GB extra |
| Host tools | Python 3.12 recommended, venv/pip, git, Docker with BuildKit, nvidia-smi, ss, findmnt |

The eight table shards occupy **196,613,849,600 bytes (183.1 GiB)** of locked,
anonymous host RAM, plus scale, staging and runtime memory. The backend preloads
with direct I/O, closes the source descriptors and serves rows from RAM into
BRRR's existing pinned staging buffers. **KV remains GPU-only.** All four
selected GPUs must be free. The container has a **220 GiB limit with no swap**
and does not auto-restart. Do not run other large RAM offloads without checking
headroom. The [SSD-backed revision](https://github.com/shihanqu/RTX-Pro-6000-Recipes/tree/6f4fd8379724afe6f7b51473581f3459a4f3f0ca/deepseek-v41-brrr-b12x-hybrid-tp4)
is available for hosts with less RAM.

## Quick Start

```bash
git clone https://github.com/shihanqu/RTX-Pro-6000-Recipes
cd RTX-Pro-6000-Recipes/deepseek-v41-brrr-b12x-hybrid-tp4
cp .env.example .env
$EDITOR .env
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

./build.sh                    # pinned upstream, dispatch patch, RAM backend and CPU tests
./download_checkpoint.sh      # ~510 GB from the original publisher; verifies every byte
./launch.sh check             # host/RAM/fabric/idle-device/image/checkpoint checks
./launch.sh                   # starts only this recipe; verifies graphs and RAM residency
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

The build verifies 35 pinned BRRR source files and builds from the LIL image
digest in `pins.json`. It applies the dispatch and RAM-loader patches with exact
before/after SHA-256 checks, compiles the RAM reader, and exercises it on the CPU.
It does not pull the private reference image. The reference image ID is recorded
in [observed.json](validation/observed.json) for provenance; local build IDs
can differ. The patched files, kernel source and entrypoint are hash-checked
against the measured image. See [packaging verification](validation/packaging.json)
for the scope of those checks.

Build on an idle Linux Docker host before loading the model. Cold builds need
at least 8 GiB of available RAM and compile one extension at a time; this
build-only change does not alter the runtime sources or kernel flags. If
upgrading the SSD recipe, stop its container explicitly before starting this
one. No launcher replaces a running service automatically.

First build/start compiles kernels and can take considerably longer than a warm
start. Keep the runtime cache. The launcher prints the actual KV pool, usually
around **7.83M shared logical tokens** on the reference setup. It is not per
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
The reference host cleared ACS request/completion redirects on the selected GPU
root ports only. This is platform-specific; the launcher does not write PCIe
registers or disable ACS on other devices.
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
| Draft | DSpark max depth 4; depth **4** at batch 1, **3** at 2-4, **2** at 5-32; probabilistic draft, block rejection, B12X draft attention/MoE |
| Engrams | RAM reader with locked anonymous tables; direct preload from checkpoint, no runtime SSD lookup; 220 GiB no-swap container |
| Sampling | Inherited temperature 1.0/top_p 1.0; default reasoning retained; normal EOS for ordinary requests |
| Features | Vision, auto tool choice, DeepSeek-V4.1 reasoning/tool parsers, prefix cache behavior inherited |
| NCCL/PCIe | Inherited SYS peer level/read enable; LL/LL128/Simple; 16 channels; 2 MiB buffers; B12X + BRRR phased collectives |

The entrypoint and low-level environment come from checksum-verified BRRR source,
not hidden host scripts. `compose.yaml` contains the overrides. The native
`num_speculative_tokens_per_batch_size` schedule uses the active decode batch
size, not the number of clients waiting. It is separate from adaptive
verification, which stays off. The fused decoder threshold is eight scheduled
token rows, **not eight users**. Keep graph capacity sufficient for speculative
expansion and padding; changing the depths or graph cap requires rebenchmarking.

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

Run the sweep on an idle endpoint; the reference run waited at least three
minutes before starting. The full sweep uses an 8192 cap and integrated prefill
scouts. Do not substitute standalone-prefill or exact token targeting
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
| Draft depth | DSpark 7 | Native batch-depth schedule **4 / 3 / 2** |
| Engram lookup | SSD reads through BRRR's NVMe backend | Preload table shards into locked host RAM and reuse BRRR's pinned staging and GPU dequantization |
| API naming | Official BRRR model ID | Preserve it and add `deepseek-v4.1-flash` compatibility alias |
| Deployment | Publisher's serve script | Generic paths/devices, safe launcher, verified downloads/builds and repeatable validation |
| Host | Upstream IOMMU/P2P guidance | Explicit loaded host settings above, with risks and no automatic mutation |

**No new checkpoint, quantization, training, CUDA kernel arithmetic or tokenizer
was introduced.** The fork adds the dispatch guard and a host RAM row reader
to BRRR's existing interface; it uses the pinned runtime's batch-depth support.
The fused kernels, attention, communication, sharded sampling, prompt fixes and
original SSD Engram design are **BRRR's work**.

### Attribution

- **[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):** native model, tokenizer, vision and draft-head work.
- **[Fraser Price / BRRR](https://huggingface.co/fraserprice/DeepSeek-4.1-Flash-1017-BRRR):** persistent fused decode/prologue kernels, SSD Engram
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
