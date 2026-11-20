"""Count inference POSTs on both sides of each benchmark cell, outside the timed window."""

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx
from benchlib import API_NAMES, IDLE, SHA, snapshot


def main():
    source = Path(sys.argv.pop(1))
    sys.argv[0] = str(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA:
        raise ValueError("Wrong client source")
    spec = importlib.util.spec_from_file_location("pinned_decode_bench", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    base = f"http://localhost:{port}"
    original_send = httpx.AsyncClient.send
    original_cell = module.run_one_cell
    confirmed = Counter()

    async def send(self, request, *args, **kwargs):
        resp = await original_send(self, request, *args, **kwargs)
        if (
            request.method == "POST"
            and request.url.host == "localhost"
            and request.url.port == port
            and request.url.path in API_NAMES
        ):
            confirmed[API_NAMES[request.url.path]] += 1
        return resp

    async def cell(*args, **kwargs):
        if kwargs["base_url"] != base:
            raise ValueError("Unexpected target")
        gauges, first = snapshot(base)
        before = confirmed.copy()
        try:
            return await original_cell(*args, **kwargs)
        finally:
            _, last = snapshot(base)
            server = {k: last[k] - first[k] for k in first}
            client = {k: confirmed[k] - before[k] for k in first}
            row = {
                "context": kwargs["context_tokens"],
                "concurrency": kwargs["concurrency"],
                "duration": kwargs["duration"],
                "quiet_at_entry": not any(gauges[k] for k in IDLE),
                "server_requests": server,
                "client_requests": client,
            }
            with Path(os.environ["WARMIND_CELL_AUDIT"]).open("a") as f:
                f.write(json.dumps(row) + "\n")

    httpx.AsyncClient.send = send
    module.run_one_cell = cell
    try:
        module.main()
    finally:
        module._restore_terminal()


if __name__ == "__main__":
    main()
