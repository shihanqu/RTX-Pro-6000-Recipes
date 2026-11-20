"""CPU-only check run inside the public builder against the real export arguments."""

import hashlib
import json
import sys
from pathlib import Path

import ignition
from ignition import patches, planning
from ignition.cli import _parser, preparation
from ignition.cli.config import Configuration


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    check(not list(Path("/dev").glob("nvidia*")), "Do not expose GPU devices to the CPU plan")
    check(not patches.apply(), "Bake the bundled Torch/Triton fixes into the preparation image first")
    root = Path(ignition.__file__).resolve().parent / "recipes/deepseek41"
    for name, pin in {
        "decoder.py": "55d131cc4cf716f902acdcea84ebcab3d150adbf8a02071ba4c66f16e12e99f8",
        "native_brrr/__init__.py": "36857c65ecfbf2a14cd7a5fa62e2da9148fc54b67c09f317f9d1e6e49298e279",
        "native_brrr/weights.py": "b4447c5f636f298e7954a9f579c5d9326edf4302c50732338d82b86f20119f4a",
        "native_brrr/prologue.cu": "0b959ab2930d1cc5dc45826c4cdc0d5e7be6b494f95ea495a09edaedb1d0dae4",
    }.items():
        check(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == pin,
            f"Different BRRR integration: {name}",
        )
    contents = Path(
        "/opt/flare-venv/lib/python3.12/site-packages/flashinfer/experimental/b12x/moe/fused_moe/workloads.py"
    ).read_bytes()
    check(
        hashlib.sha1(f"blob {len(contents)}\0".encode() + contents, usedforsecurity=False).hexdigest()
        == "f0288ee4cf4b7625b52c1e305170e38c54e1d3c7",
        "Missing expert-reuse workload patch",
    )
    a = _parser().parse_args(sys.argv[1:])
    cfg = Configuration.load(a.config)
    model = preparation.resolve_model(cfg, a.model)
    check(
        model.directory == Path("/inputs/DeepSeek-V4.1-Flash-1017-warmind-orth4c")
        and model.recipe.name == "deepseek41"
        and model.cached(cfg.directory("hub_cache")),
        "Expected all native WARMIND shards",
    )
    check(
        not (a.no_weights or a.reuse_weights or a.no_compile or a.language_model_only),
        "Fresh compiled text+vision export required",
    )
    check(
        hashlib.sha256(Path(a.collective_ladder).read_bytes()).hexdigest()
        == "d25e68194bdd6b370f7c6c50552f359f909fbde7afcb2d508a427df409c88229",
        "Wrong PCIe ladder",
    )
    # Idle planning assumptions only. Export selects and checks the actual four devices.
    gpus = tuple(
        planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"plan-rank-{i}", 96 << 30, 95 << 30, (12, 0))
        for i in range(4)
    )
    p = preparation.derive(
        cfg,
        model,
        preparation.Host(gpus, None),
        gpus=a.gpus,
        concurrency=cfg.max_concurrency(model.checkpoint.alias, model.recipe.concurrency, a.max_concurrency),
        speculation=preparation.speculation(a, model, cfg),
        prefill_chunk=a.prefill_chunk_size,
        image_concurrency=a.image_concurrency,
        collective_tuning=a.collective_tuning,
        collective_ladder=a.collective_ladder,
        expert_scale_compression=a.expert_scale_compression,
        mla_compute_mode=a.mla_compute_mode,
        compile_workers=a.compile_workers,
        autotune_effort=a.autotune_effort,
    )
    s = p.speculation
    check(
        (p.world_size, p.concurrency, p.context, p.prefill_chunk, p.image_concurrency)
        == (4, 32, 1048576, 4096, 4),
        "Wrong serving shape",
    )
    check(
        (s.draft_tokens, s.graph_budget, p.speculative_token_capacity) == (4, 64, 128)
        and s.packed_verification
        and s.adaptive
        and s.vocabulary_parallel_sampling
        and (s.draft_head_quantization, s.markov_head_quantization) == ("nvfp4", "fp8"),
        "Wrong draft plan",
    )
    check(
        p.expert_scale_compression is None
        and p.mla_compute_mode == "bf16"
        and p.collective_tuning == "runtime"
        and p.collective_ladder["name"] == "gb202_pcie_tp4_clustered_full",
        "Wrong precision or collective plan",
    )
    print(
        json.dumps(
            dict(
                root=model.model_id,
                tp=p.world_size,
                draft_tokens=s.draft_tokens,
                slots=p.concurrency,
                context=p.context,
                vision_slots=p.image_concurrency,
                packed_rows=p.speculative_token_capacity,
            )
        )
    )


if __name__ == "__main__":
    main()
