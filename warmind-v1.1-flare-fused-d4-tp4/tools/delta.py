"""Optional bandwidth-saving assembly from a separate, pristine native stock checkpoint."""

import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from .checkpoint import MANIFEST, REPO, REV, digest_file

BASE = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE_REV = "1bae0e1088993b9b82de08f59de5be5a34e42170"
PROVENANCE_SHA = "82d806b1a327061ff37384e1d19bc81e35e50e98fa6ea3e7e3dae5f3e5ecb425"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fetch(url, limit, span=None):
    headers = {"Accept-Encoding": "identity"}
    if span:
        start, end, total = span
        headers["Range"] = f"bytes={start}-{end}"
        url += f"?range={start}-{end}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                if span:
                    require(
                        r.status == 206 and r.headers.get("Content-Range") == f"bytes {start}-{end}/{total}",
                        "Server did not honor the exact byte range",
                    )
                body = r.read(limit + 1)
            require(
                len(body) <= limit and (not span or len(body) == end - start + 1), "Invalid response length"
            )
            return body
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def assemble(src, dst, changes, entry, old, get):
    size = entry["bytes"]
    require(src.is_file() and src.stat().st_size == size == old["size"], "Wrong stock shard size")
    before, after = hashlib.sha256(), hashlib.sha256()
    with src.open("rb") as inp, dst.open("wb") as out:
        (size_header,) = struct.unpack("<Q", inp.read(8))
        require(0 < size_header <= 64 << 20, "Invalid safetensors header")
        header = json.loads(inp.read(size_header))
        inp.seek(0)

        def put(raw, replaced):
            before.update(raw)
            after.update(replaced)
            out.write(replaced)

        def copy_to(end):
            while inp.tell() < end:
                data = inp.read(min(8 << 20, end - inp.tell()))
                require(data, "Unexpected end of base file")
                put(data, data)

        for change in sorted(changes, key=lambda e: e["byte_offset"]):
            start, length = change["byte_offset"], change["byte_length"]
            a, b = header[change["tensor"]]["data_offsets"]
            require(
                (8 + size_header + a, b - a) == (start, length)
                and inp.tell() <= start < start + length <= size
                and length <= 64 << 20,
                "Invalid tensor range",
            )
            copy_to(start)
            raw = inp.read(length)
            require(hashlib.sha256(raw).hexdigest() == change["pristine_sha256"], "Stock tensor mismatch")
            edited = get(start, length)
            require(
                len(edited) == length and hashlib.sha256(edited).hexdigest() == change["derivative_sha256"],
                "WARMIND tensor mismatch",
            )
            put(raw, edited)
        copy_to(size)
    require(
        before.hexdigest() == old["lfs"]["sha256"] and after.hexdigest() == entry["hash"],
        "Whole-shard hash mismatch",
    )


def stage(stock, target, link_identical=False):
    require(
        stock.is_dir() and not target.is_relative_to(stock) and not stock.is_relative_to(target),
        "Use separate native directories",
    )
    if link_identical:
        require(stock.stat().st_dev == target.stat().st_dev, "Hard links require the same filesystem")
    base = json.loads(
        fetch(f"https://huggingface.co/api/models/{BASE}/revision/{BASE_REV}?blobs=true", 4 << 20)
    )
    require(base["id"] == BASE and base["sha"] == BASE_REV, "Wrong stock manifest")
    old = {f["rfilename"]: f for f in base["siblings"]}
    files = {f["file"]: f for f in json.loads(MANIFEST.read_text())["files"]}

    def url(n):
        return f"https://huggingface.co/{REPO}/resolve/{REV}/" + urllib.parse.quote(n, safe="/")

    provenance = fetch(url("DERIVATIVE.json"), files["DERIVATIVE.json"]["bytes"])
    require(hashlib.sha256(provenance).hexdigest() == PROVENANCE_SHA, "Unexpected derivative manifest")
    model = json.loads(provenance)
    require(
        model["base_model"]["revision"] == BASE_REV and model["artifact"]["variant"] == "orth4c",
        "Wrong lineage",
    )
    edits = defaultdict(list)
    for e in model["derivation"]["tensor_edits"]:
        edits[e["shard"]].append(e)
    shards = {n for n in files if n.endswith(".safetensors")}
    require(
        len(shards) == 48
        and len(edits) == 33
        and sum(map(len, edits.values())) == 132
        and {n for n in shards if files[n]["hash"] != old[n]["lfs"]["sha256"]} == edits.keys(),
        "Unexpected shard delta",
    )
    tmpdir = target / ".cache/huggingface/recipe-staging"
    tmpdir.mkdir(parents=True, exist_ok=True)
    require(tmpdir.resolve() == tmpdir, "Unsafe temporary directory")
    for name, entry in files.items():
        dest = target / name
        require(dest.resolve() == dest and dest.is_relative_to(target), "Unsafe output path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            require(
                dest.stat().st_size == entry["bytes"]
                and digest_file(dest, entry["algorithm"]) == entry["hash"],
                f"Existing file differs: {name}",
            )
            continue
        if name in shards and name not in edits and link_identical:
            src = stock / name
            require(
                src.is_file() and src.resolve() == src and digest_file(src) == entry["hash"],
                f"Stock file differs: {name}",
            )
            os.link(src, dest)  # Never open this link for writing or change its permissions.
            continue
        with tempfile.NamedTemporaryFile(dir=tmpdir, delete=False) as f:
            tmp = Path(f.name)
        try:
            if name in edits:
                assemble(
                    stock / name,
                    tmp,
                    edits[name],
                    entry,
                    old[name],
                    lambda pos, n: fetch(url(name), n, (pos, pos + n - 1, entry["bytes"])),
                )
            elif name in shards:
                shutil.copyfile(stock / name, tmp)
            else:
                require(entry["bytes"] <= 32 << 20, "Unexpected non-weight asset size")
                tmp.write_bytes(provenance if name == "DERIVATIVE.json" else fetch(url(name), entry["bytes"]))
            require(
                tmp.stat().st_size == entry["bytes"]
                and digest_file(tmp, entry["algorithm"]) == entry["hash"],
                f"File did not verify: {name}",
            )
            tmp.chmod(0o444)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
