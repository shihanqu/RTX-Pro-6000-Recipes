"""Verify every native file against the pinned public manifest, with optional local receipt reuse."""

import hashlib
import json
import os
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "manifests/native.json"
REPO = "jcartu/WARMIND-1017"
REV = "a6382f222da49b7109586eecbe1e53a10a4e0637"
ROOT_ID = "DeepSeek-V4.1-Flash-1017-warmind-orth4c"
MODEL = "deepseek-v4.1-flash"


def digest_file(path, algorithm="sha256"):
    if algorithm == "sha256":
        h = hashlib.sha256()
    elif algorithm == "git-blob-sha1":
        h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    else:
        raise ValueError(f"Unknown digest algorithm: {algorithm}")
    offset = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
    return h.hexdigest()


def signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify(root, receipt, *, full=False):
    raw = MANIFEST.read_bytes()
    meta = json.loads(raw)
    if (meta["repository"], meta["revision"]) != (REPO, REV):
        raise ValueError("Incorrect native checkpoint manifest")
    files = {f["file"]: f for f in meta["files"]}
    if (
        len(files) != len(meta["files"])
        or len(files) != 88
        or sum(f["bytes"] for f in files.values()) != 510318572683
    ):
        raise ValueError("Unexpected file count or total size")
    shards = {f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)}
    if not shards <= files.keys():
        raise ValueError("The native manifest is missing weight shards")
    verify_files(root, receipt, files, hashlib.sha256(raw).hexdigest(), full=full, expected_shards=shards)


def verify_files(root, receipt, files, manifest_hash, *, full, expected_shards):
    actual = set()
    for parent, dirs, names in os.walk(root):
        if Path(parent) == root / ".cache" and "huggingface" in dirs:
            dirs.remove("huggingface")
        actual.update((Path(parent) / n).relative_to(root).as_posix() for n in names)
    if actual != files.keys():
        raise ValueError(
            f"Native file set differs: missing={sorted(files.keys() - actual)}, extra={sorted(actual - files.keys())}"
        )
    prior = json.loads(receipt.read_text()) if receipt.is_file() and not full else {}
    cached = (
        prior.get("verified", {})
        if prior.get("manifest") == manifest_hash and prior.get("directory") == str(root)
        else {}
    )
    checked = {}
    reread = 0
    for n, f in files.items():
        p = root / n
        if p.resolve() != p or not p.is_file():
            raise ValueError(f"Expected a regular local file, not a symlink: {n}")
        sig = signature(p)
        if sig[2] != f["bytes"]:
            raise ValueError(f"Wrong size: {n}")
        if cached.get(n) != sig:
            if digest_file(p, f["algorithm"]) != f["hash"] or signature(p) != sig:
                raise ValueError(f"Digest mismatch or concurrent modification: {n}")
            reread += 1
        checked[n] = sig
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index.values()) != expected_shards:
        raise ValueError("The native weight index is incomplete")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"manifest": manifest_hash, "directory": str(root), "verified": checked}) + "\n"
    )
    tmp.replace(receipt)
    print(
        f"Native WARMIND verified: {len(files)} files; {reread} read in full, {len(files) - reread} unchanged since hashing.",
        flush=True,
    )
