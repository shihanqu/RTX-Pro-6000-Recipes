#!/usr/bin/env python3
"""Download, build, export, inspect and serve WARMIND v1.1 FLARE-Fused D4."""

import argparse
import csv
import fcntl
import ipaddress
import json
import os
import shutil
import socket
import subprocess as sp
import sys
import time
import urllib.request
from pathlib import Path

from tools import catalog, checkpoint, sources

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", HERE / "work")).expanduser().resolve()
NATIVE = Path(os.environ.get("MODEL_DIR", HERE / "checkpoint")).expanduser().resolve()
PREP = os.environ.get("PREP_IMAGE", "local/warmind-v11-flare-fused-d4-prepare:972e28e")
RUNTIME = "ghcr.io/local-inference-lab/flare@sha256:08742b9e683425a5f2e86c8c4668ea9bac0c9cf041aaa1f3df9a1916de638b52"
PROJECT = "warmind-v11-flare-fused-d4"
INPUT = "/inputs/" + checkpoint.ROOT_ID
GIB = 1 << 30
PY = "/opt/flare-venv/bin/python"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def output(*cmd, env=None):
    return sp.check_output(cmd, text=True, env=env).strip()


def atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def base_env():
    return {
        **os.environ,
        "WORK_DIR": str(WORK),
        "MODEL_DIR": str(NATIVE),
        "PREP_IMAGE": PREP,
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid()),
    }


def gpu_env(env, *, idle):
    ids = env.get("GPU_IDS", "").split(",")
    require(
        len(ids) == len(set(ids)) == 4 and all(u.startswith("GPU-") for u in ids),
        "Set GPU_IDS to four physical GPU UUIDs, in rank order",
    )

    def smi(query):
        return list(
            csv.reader(
                output("nvidia-smi", query, "--format=csv,noheader,nounits").splitlines(),
                skipinitialspace=True,
            )
        )

    cards = {u: (n, float(w)) for u, n, w in smi("--query-gpu=uuid,name,power.limit")}
    watts = int(env.get("EXPECTED_WATTS", "450"))
    for u in ids:
        require(
            cards.get(u) == ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", watts),
            "Card model, UUID or configured power cap differs",
        )
    if idle:
        require(
            not any(r and r[0] in ids for r in smi("--query-compute-apps=gpu_uuid,pid")),
            "A selected card is busy; nothing stopped",
        )
    env.update({f"GPU{i}": u for i, u in enumerate(ids)})
    return ids


def host_port():
    p = int(os.environ.get("HOST_PORT", "1230"))
    require(1 <= p <= 65535, "Invalid HOST_PORT")
    return p


def free_resources(port):
    # Docker may publish via NAT without a userspace listening socket.
    require(
        not output("ss", "-Hltn", f"sport = :{port}")
        and not output("docker", "ps", "-q", "--filter", f"publish={port}"),
        f"Port {port} is occupied; nothing stopped",
    )
    available = (
        int(
            next(
                s.split()[1]
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    require(available >= 200 * GIB, "At least 200 GiB available host RAM required for this profile")


def compose(kind):
    export = kind == "export"
    return [
        "docker",
        "compose",
        "--env-file",
        os.devnull,
        "-p",
        PROJECT + ("-export" if export else ""),
        "-f",
        str(HERE / ("compose.export.yaml" if export else "compose.yaml")),
    ]


def verify_native(full=False):
    checkpoint.verify(NATIVE, WORK / "native-verified.json", full=full)


def download(stock_dir, link_identical):
    require(not link_identical or stock_dir is not None, "--link-identical requires --stock-dir")
    NATIVE.mkdir(parents=True, exist_ok=True)
    marker = NATIVE / ".cache/huggingface/warmind-native-revision"
    require(marker.resolve() == marker, "Unsafe cache marker path")
    if marker.exists():
        require(marker.read_text().strip() == checkpoint.REV, "This directory belongs to another revision")
    elif any(NATIVE.iterdir()):
        # Never let a downloader overwrite a stock or mixed checkpoint of identical shape.
        verify_native(full=True)
        return
    else:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(checkpoint.REV + "\n")
    if stock_dir is None:
        sp.run(
            ["hf", "download", checkpoint.REPO, "--revision", checkpoint.REV, "--local-dir", str(NATIVE)],
            check=True,
        )
    else:
        from tools.delta import stage

        stage(stock_dir.expanduser().resolve(strict=True), NATIVE, link_identical)
    verify_native(full=True)


def builder():
    image = json.loads(output("docker", "image", "inspect", PREP))[0]
    labels = image["Config"].get("Labels") or {}
    require(
        image["Architecture"] == "amd64"
        and labels.get("org.opencontainers.image.revision") == sources.FLARE_REV
        and labels.get("local.flashinfer.revision") == sources.FLASHINFER_REV
        and labels.get("local.flashinfer.workloads") == sources.REUSE_REV,
        "Builder pins do not match; run recipe.py build",
    )
    return image["Id"]


def build(only_sources):
    src = sources.prepare(HERE, WORK)
    print(f"Flare sources prepared in {src}", flush=True)
    if only_sources:
        return
    sp.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-f",
            str(src / "Dockerfile.recipe"),
            "-t",
            PREP,
            str(src),
        ],
        check=True,
    )
    print("Pinned source builder:", builder())


def prepare(idle):
    verify_native()
    image = builder()
    env = base_env()
    gpu_env(env, idle=idle)
    settings = WORK / "settings"
    settings.mkdir(mode=0o700, exist_ok=True)
    config = settings / "prepare.toml"
    if not config.exists():
        shutil.copyfile(HERE / "prepare.toml", config)
    require(
        config.read_bytes() == (HERE / "prepare.toml").read_bytes(),
        "Use an isolated WORK_DIR with this recipe configuration",
    )
    rendered = json.loads(output(*compose("export"), "config", "--format", "json", env=env))["services"][
        "export"
    ]
    args = rendered["command"]
    require(args[:2] == ["export", INPUT] and rendered["image"] == PREP, "Unexpected export input or builder")
    cmd = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--runtime",
        "runc",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--memory-swap",
        "2g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,mode=1777,size=256m",
        "-e",
        "HOME=/tmp",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=void",
        "-e",
        "PYTHONPATH=" + rendered["environment"]["PYTHONPATH"],
        "-w",
        "/src",
    ]
    for source, dest in [
        (NATIVE, INPUT),
        (config, "/work/settings/prepare.toml"),
        (HERE / "tools/plan.py", "/plan.py"),
    ]:
        cmd += ["--mount", f"type=bind,src={source},dst={dest},readonly"]
    plan = json.loads(output(*cmd, "--entrypoint", PY, PREP, "-B", "/plan.py", *args))
    print("Verified CPU-only export plan:", json.dumps(plan), flush=True)
    return env, image, args


def package_manifests():
    return sorted((WORK / "packages/models" / checkpoint.ROOT_ID).glob("*/model.safetensors"))


def export():
    free_resources(host_port())
    require(
        shutil.disk_usage(WORK).free >= 1200 * GIB,
        "Need 1,200 GiB free on WORK_DIR after the native checkpoint is staged",
    )
    require(
        not (WORK / "export.json").exists()
        and not any((WORK / "packages").iterdir() if (WORK / "packages").exists() else ())
        and not any((WORK / "settings").glob("development-*")),
        "Use a fresh package cache and signer; never reuse stock weights or overwrite an export",
    )
    env, image, args = prepare(True)
    gpu_env(env, idle=True)
    free_resources(host_port())
    sp.run([*compose("export"), "run", "--rm", "--no-deps", "export"], env=env, check=True)
    files = package_manifests()
    require(len(files) == 1, "Expected exactly one WARMIND D4 export")
    _, key = catalog.publisher(WORK / "settings/development-publisher.json")
    atomic_json(
        WORK / "export.json",
        dict(
            native_revision=checkpoint.REV,
            builder=image,
            args=args,
            package=str(files[0].parent.relative_to(WORK)),
            manifest_sha256=checkpoint.digest_file(files[0]),
            signer=key,
        ),
    )
    print("Local signed WARMIND D4 export ready for inspection:", files[0].parent)


def inspect_package():
    record = json.loads((WORK / "export.json").read_text())
    pkg = (WORK / record["package"]).resolve(strict=True)
    require(
        record.get("native_revision") == checkpoint.REV
        and pkg.is_relative_to((WORK / "packages/models" / checkpoint.ROOT_ID).resolve())
        and checkpoint.digest_file(pkg / "model.safetensors") == record["manifest_sha256"],
        "Wrong package path, native revision or manifest",
    )
    profile, fingerprint = catalog.publisher(WORK / "settings/development-publisher.json")
    require(record.get("signer") == fingerprint, "Signer differs from the one recorded at export")
    # Bind only a public profile. Keep the private key and its directory away from the serving container.
    pub = WORK / "serve-public.json"
    public_bytes = (json.dumps([profile]) + "\n").encode()
    if not pub.exists():
        with pub.open("xb") as f:
            f.write(public_bytes)
        pub.chmod(0o644)
    require(
        not pub.is_symlink() and pub.read_bytes() == public_bytes, "Unexpected public profile in WORK_DIR"
    )
    data = json.loads(
        output(
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "1g",
            "--memory-swap",
            "1g",
            "--mount",
            f"type=bind,src={pkg},dst=/model,readonly",
            "--mount",
            f"type=bind,src={pub},dst=/publisher.json,readonly",
            RUNTIME,
            "inspect",
            "/model/model.safetensors",
            "--publisher-keys",
            "/publisher.json",
            "--json",
            "--graphs",
            "--frontend",
        )
    )
    counts = catalog.check_catalog(data, fingerprint, checkpoint.ROOT_ID)
    print(
        "Verified signed WARMIND D4. Compact graph node counts per rank (C1; four C2 layouts):",
        counts,
        flush=True,
    )
    return pkg, pub, fingerprint


def bindings():
    loop = socket.getaddrinfo("localhost", None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    require(ipaddress.ip_address(loop).is_loopback, "localhost did not resolve to loopback")
    return loop, socket.getaddrinfo(None, 0, socket.AF_INET, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)[0][4][
        0
    ]


def private_workspace(work, native):
    require(
        work != native and not work.is_relative_to(native) and not native.is_relative_to(work),
        "WORK_DIR and MODEL_DIR must be disjoint",
    )
    for path in (work, native):
        require(
            path not in (Path("/"), Path.home()) and not HERE.is_relative_to(path),
            "Use dedicated model and work directories; not the project root, home or filesystem root",
        )
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = work.stat()
    require(
        state.st_uid == os.getuid() and not state.st_mode & 0o077,
        "Use a private, operator-owned WORK_DIR with mode 0700",
    )


def check_layers():
    pkg, _, _ = inspect_package()
    sample = json.loads((HERE / "manifests/edited-layer-samples.json").read_text())["files"]
    for f in sample:
        path = pkg / f["file"]
        require(
            path.resolve().is_relative_to(pkg)
            and path.stat().st_size == f["bytes"]
            and checkpoint.digest_file(path) == f["sha256"],
            f"Exported layer differs from published WARMIND: {f['file']}",
        )
    print(f"{len(sample)}/{len(sample)} sampled rank-layer weight files match the publisher.")


def get(path, port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://localhost:{port}" + path, timeout=5) as resp:
        return json.load(resp)


def live_id():
    ids = output(
        "docker",
        "ps",
        "-q",
        "--no-trunc",
        "--filter",
        f"label=com.docker.compose.project={PROJECT}",
        "--filter",
        "label=com.docker.compose.service=serve",
    ).splitlines()
    require(len(ids) == 1, "Expected exactly one live WARMIND D4 recipe container")
    return ids[0]


def health(cid, port):
    require(live_id() == cid, "The live container changed")
    require(get("/health", port)["status"] == "ok", "Not healthy")
    card = get("/v1/models", port)["data"][0]
    metrics = get("/metrics?format=json", port)
    require(
        card["id"] == checkpoint.MODEL
        and card["root"] == checkpoint.ROOT_ID
        and card["max_model_len"] == 1048576
        and card["flare"]["max_concurrency"] == 32
        and "image" in card["flare"]["input_modalities"],
        "Wrong API model or capacity",
    )
    return metrics["total_pages"] * metrics["page_tokens"]


def start():
    port = host_port()
    free_resources(port)
    env = base_env()
    ids = gpu_env(env, idle=True)
    pkg, pub, signer = inspect_package()
    loop, listen = bindings()
    env.update(
        PACKAGE_DIR=str(pkg),
        PUBLISHER_PROFILE=str(pub),
        PUBLISHER_FINGERPRINT=signer,
        LOCAL_BIND=loop,
        CONTAINER_LISTEN=listen,
        HOST_PORT=str(port),
    )
    cmd = compose("serve")
    desired = json.loads(output(*cmd, "config", "--format", "json", env=env))["services"]["serve"]
    sp.run([*cmd, "create", "--no-build", "--no-recreate", "--pull", "never", "serve"], env=env, check=True)
    cid = output(*cmd, "ps", "-a", "-q", "serve", env=env)
    d = json.loads(output("docker", "inspect", "--type", "container", cid))[0]
    mounts = {(m["Source"], m["Destination"], m["RW"]) for m in d["Mounts"] if m["Type"] == "bind"}
    g = d["HostConfig"]["DeviceRequests"]
    ports = d["HostConfig"]["PortBindings"]
    require(
        not d["State"]["Running"]
        and d["Config"]["Image"] == RUNTIME
        and d["Config"]["Cmd"] == desired["command"]
        and mounts == {(str(pkg), "/model", False), (str(pub), "/publisher.json", False)}
        and len(g) == 1
        and g[0]["DeviceIDs"] == ids
        and ports == {"1230/tcp": [{"HostIp": loop, "HostPort": str(port)}]},
        "A stopped container has different settings; inspect it before removal",
    )
    gpu_env(env, idle=True)
    free_resources(port)
    sp.run(["docker", "start", cid], check=True)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            require(
                output("docker", "inspect", "-f", "{{.State.Running}}", cid) == "true",
                "Container exited; inspect docker logs",
            )
            try:
                kv = health(d["Id"], port)
                print(f"Ready: {checkpoint.MODEL}, {kv:,} shared KV tokens")
                return
            except OSError:
                time.sleep(4)
        raise ValueError("Not ready within 15 minutes")
    except (Exception, KeyboardInterrupt):
        sp.run(["docker", "stop", "--timeout", "90", cid], check=False)
        raise


def stop():
    cid = live_id()
    d = json.loads(output("docker", "inspect", cid))[0]
    port = host_port()
    loop, _ = bindings()
    require(
        d["Config"]["Image"] == RUNTIME
        and d["HostConfig"]["PortBindings"] == {"1230/tcp": [{"HostIp": loop, "HostPort": str(port)}]},
        "Unexpected port or image",
    )
    health(cid, port)
    previous = None
    quiet = 0
    for _ in range(300):
        v = get("/metrics?format=json", port)
        quiet = (quiet + 1 if v["completed"] == previous else 1) if v["active"] == v["pending"] == 0 else 0
        previous = v["completed"]
        if quiet >= 15:
            require(live_id() == cid, "Port owner changed")
            sp.run(["docker", "stop", "--timeout", "90", cid], check=True)
            return
        time.sleep(1)
    raise ValueError("Requests did not drain; nothing was stopped")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for a in ("verify", "plan", "export", "inspect", "check-layers", "start", "stop", "status"):
        sub.add_parser(a)
    dl = sub.add_parser("download")
    dl.add_argument("--stock-dir", type=Path)
    dl.add_argument("--link-identical", action="store_true")
    sub.add_parser("build").add_argument("--source-only", action="store_true")
    a = p.parse_args()
    os.umask(0o077)
    private_workspace(WORK, NATIVE)
    with (WORK / ".recipe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.action == "download":
            download(a.stock_dir, a.link_identical)
        elif a.action == "verify":
            verify_native(full=True)
        elif a.action == "build":
            build(a.source_only)
        elif a.action == "plan":
            prepare(False)
        elif a.action == "export":
            export()
        elif a.action == "inspect":
            inspect_package()
        elif a.action == "check-layers":
            check_layers()
        elif a.action == "start":
            start()
        elif a.action == "stop":
            stop()
        else:
            print(f"{checkpoint.MODEL}: {health(live_id(), host_port()):,} shared KV tokens")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, sp.CalledProcessError) as error:
        sys.exit(str(error))
