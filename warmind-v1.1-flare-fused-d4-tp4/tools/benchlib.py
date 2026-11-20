"""Shared local-only benchmark setup. Raw client output stays in the operator's run directory."""

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request

from checkpoint import MODEL, ROOT_ID

REV = "52216d70865a015b34689626d04a80b89c9dc45f"
SHA = "d15646fcf09993597b75887c134d812721f4ccad6cdc846bea299cceae1f8a90"
SOURCE = (
    f"https://raw.githubusercontent.com/local-inference-lab/llm-inference-bench/{REV}/llm_decode_bench.py"
)
PROJECT = "warmind-v11-flare-fused-d4"
RUNTIME = "ghcr.io/local-inference-lab/flare@sha256:08742b9e683425a5f2e86c8c4668ea9bac0c9cf041aaa1f3df9a1916de638b52"
IDLE = (
    "flare_active_requests",
    "flare_pending_requests",
    "flare_retiring_request_slots",
    "flare_http_inflight_requests",
    "flare_frontend_active_workers",
)
COUNTERS = ("flare_generated_tokens_total", "flare_requests_admitted_total", "flare_requests_completed_total")
API_NAMES = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
    "/v1/messages/count_tokens": "count_tokens",
}
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(base, path):
    with LOCAL.open(base + path, timeout=10) as r:
        return r.read().decode()


def snapshot(base):
    text = get(base, "/metrics")
    names = {*IDLE, *COUNTERS, "flare_total_kv_tokens"}
    gauges = {
        p[0]: float(p[1]) for line in text.splitlines() if len(p := line.split()) >= 2 and p[0] in names
    }
    http = {
        k: int(float(v))
        for k, v in re.findall(r'^flare_http_requests_total\{api="([^"]+)"\}\s+(\S+)', text, re.M)
    }
    if gauges.keys() != names or set(http) != set(API_NAMES.values()):
        raise ValueError("Incomplete native Flare metrics")
    return gauges, http


def wait_idle(base):
    until = time.monotonic() + 900
    quiet = None
    previous = None
    while time.monotonic() < until:
        g, http = snapshot(base)
        now = time.monotonic()
        counts = tuple(g[k] for k in COUNTERS) + tuple(http[k] for k in sorted(http))
        if not any(g[k] for k in IDLE) and (previous is None or previous == counts):
            quiet = now if quiet is None else quiet
            if now - quiet >= 6:
                return
        else:
            quiet = None
        previous = counts
        time.sleep(1)
    raise RuntimeError("No quiet measurement window; no other requests were stopped")


def client(folder, local_source=None):
    if local_source:
        data = local_source.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as r:
            data = r.read(2 << 20)
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("Pinned benchmark digest mismatch")
    p = folder / "llm_decode_bench.py"
    p.write_bytes(data)
    return p


def env():
    e = {**os.environ, "LLM_BENCH_NO_UPDATE_CHECK": "1", "NO_PROXY": "localhost", "no_proxy": "localhost"}
    e.pop("LLM_BENCH_API_KEY", None)
    return e


def target(base, port, depth, container=None):
    if not container:
        container = subprocess.check_output(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                f"label=com.docker.compose.project={PROJECT}",
                "--filter",
                "label=com.docker.compose.service=serve",
            ],
            text=True,
        ).strip()
    if not container or "\n" in container:
        raise ValueError("Choose one serving container")
    d = json.loads(
        subprocess.check_output(["docker", "inspect", "--type", "container", container], text=True)
    )[0]
    card = json.loads(get(base, "/v1/models"))["data"][0]
    cmd = d["Config"]["Cmd"]
    params = {
        "--mtp-draft-tokens": str(depth),
        "--max-concurrency": "32",
        "--scheduler-policy": "balanced",
        "--embedding-table-storage": "ram",
        "--mtp-adaptive-verification-alpha": "0.6",
        "--mtp-max-verification-token-budget": "128",
        "--prefix-cache-memory": "0",
        "--kv-cache-limit": "memory",
        "--served-model-name": MODEL,
    }
    ports = {x["HostPort"] for bindings in d["NetworkSettings"]["Ports"].values() for x in bindings or []}
    image = subprocess.check_output(
        ["docker", "image", "inspect", "-f", "{{.Id}}", RUNTIME], text=True
    ).strip()
    if (
        not d["State"]["Running"]
        or d["Image"] != image
        or str(port) not in ports
        or card["id"] != MODEL
        or card["root"] != ROOT_ID
        or card["max_model_len"] != 1048576
        or card["flare"]["max_concurrency"] != 32
        or "image" not in card["flare"]["input_modalities"]
        or "--mtp-adaptive-verification" not in cmd
        or any(flag not in cmd or cmd[cmd.index(flag) + 1] != v for flag, v in params.items())
    ):
        raise ValueError("Wrong WARMIND container, draft depth, endpoint or serving policy")
    return d["Id"], d["State"]["StartedAt"], card["created"]
