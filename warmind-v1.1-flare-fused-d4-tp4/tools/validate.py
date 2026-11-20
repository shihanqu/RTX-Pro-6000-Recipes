"""Run local WARMIND arithmetic, image, concurrent-request and Estonia correctness gates."""

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import threading
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchlib import LOCAL, MODEL, ROOT_ID, client, env, target, wait_idle

COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0)}
FONT = {
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "9": ("111", "101", "111", "001", "111"),
}


def fixture(color, number=None):
    size = 544 if number is None else 160
    rgb = COLORS[color]
    pixels = bytearray(bytes(rgb) * size * size)
    ink = bytes((0, 0, 0) if color in ("green", "yellow") else (255, 255, 255))
    for i, n in enumerate(str(number) if number is not None else ""):
        for y, row in enumerate(FONT[n]):
            for x, bit in enumerate(row):
                if bit == "1":
                    for dy in range(8):
                        offset = ((56 + y * 8 + dy) * size + 48 + i * 32 + x * 8) * 3
                        pixels[offset : offset + 24] = ink * 8

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\0" + pixels[y * size * 3 : (y + 1) * size * 3] for y in range(size))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()},
    }


def chat(base, content, schema=None):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }
    req = urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with LOCAL.open(req, timeout=120) as r:
        reply = json.load(r)
    choice = reply["choices"][0]
    if reply["model"] != ROOT_ID or choice["finish_reason"] != "stop":
        raise ValueError("Wrong model root or truncated response")
    return choice["message"]["content"].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=1230)
    p.add_argument("--container")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--client", type=Path)
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only structured arithmetic and four simultaneous color images",
    )
    a = p.parse_args()
    base = f"http://localhost:{a.port}"
    identity = target(base, a.port, 4, a.container)
    os.umask(0o077)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    wait_idle(base)
    arithmetic = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = {
        "structured_arithmetic": json.loads(
            chat(base, "What is 17 times 23? Return the integer in the answer field.", arithmetic)
        )
        == {"answer": 391}
    }

    def color(name):
        answer = chat(
            base,
            [
                {"type": "text", "text": "What color fills this image? Answer with only one color word."},
                fixture(name),
            ],
        )
        return answer.strip(" .!\n").casefold() == name

    with ThreadPoolExecutor(4) as pool:
        result["concurrent_colors"] = sum(pool.map(color, COLORS))
    passed = result["structured_arithmetic"] and result["concurrent_colors"] == 4
    if not a.smoke_only:
        wait_idle(base)
        schema = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": list(COLORS)}, "number": {"type": "integer"}},
            "required": ["color", "number"],
            "additionalProperties": False,
        }

        def ocr(case):
            name, n = case
            answer = chat(
                base,
                [
                    {
                        "type": "text",
                        "text": "Read the large number and identify the background color. Return them in the requested fields.",
                    },
                    fixture(name, n),
                ],
                schema,
            )
            return json.loads(answer) == {"color": name, "number": n}

        with ThreadPoolExecutor(4) as pool:
            result["concurrent_color_and_ocr"] = sum(pool.map(ocr, zip(COLORS, [17, 29, 42, 63])))
        barrier = threading.Barrier(32)

        def multiply(i):
            x, y = 19 + i, 21 + i
            barrier.wait(timeout=30)
            return chat(base, f"What is {x} times {y}? Answer with only the integer.") == str(x * y)

        with ThreadPoolExecutor(32) as pool:
            result["concurrent_arithmetic"] = sum(pool.map(multiply, range(32)))
        wait_idle(base)
        source = client(out, a.client)
        path = out / "estonia.json"
        cmd = [
            sys.executable,
            "-u",
            str(source),
            "--host",
            "localhost",
            "--port",
            str(a.port),
            "--model",
            MODEL,
            "--display-mode",
            "plain",
            "--no-resume",
            "--no-hw-monitor",
            "--test-profile",
            "estonia",
            "--profile-concurrency",
            "10",
            "--profile-runs",
            "10",
            "--max-tokens",
            "40000",
            "--completion-stats-temperature",
            "1",
            "--completion-stats-top-p",
            "1",
            "--completion-stats-seed",
            "20261113",
            "--completion-stats-request-timeout",
            "480",
            "--completion-stats-stall-timeout",
            "240",
            "--output",
            str(path),
        ]
        with (out / "estonia.log").open("x") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env(), timeout=720, check=True)
        s = json.loads(path.read_text())["selected_summary"]
        result["estonia"] = {
            k: s.get(k, 0)
            for k in (
                "attempted",
                "completed",
                "correct",
                "wrong",
                "errors",
                "truncated",
                "stalled",
                "timeout",
                "hit_max_tokens",
            )
        }
        passed = (
            passed
            and result["concurrent_color_and_ocr"] == 4
            and result["concurrent_arithmetic"] == 32
            and s["correct"] == s["completed"] == s["attempted"] == 10
            and not any(
                s.get(k, 0) for k in ("wrong", "errors", "truncated", "stalled", "timeout", "hit_max_tokens")
            )
        )
    result["passed"] = passed and target(base, a.port, 4, a.container) == identity
    (out / "quality.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
