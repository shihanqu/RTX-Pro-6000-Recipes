#!/usr/bin/env python3
"""Download one pinned Hugging Face revision into a cache directory, verify every file
(size plus LFS SHA-256 or git blob SHA-1 against the Hub manifest) and write a receipt.

Called by download_checkpoint.sh; needs huggingface_hub. The receipt is what launch.sh checks:
{"repo", "revision", "verified_at_utc", "files": [{"file", "bytes", "hash"}], "total_bytes"}.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile


def expected_hash(item: RepoFile) -> str:
    return item.lfs.sha256 if item.lfs else item.blob_id


def verify(snapshot: Path, item: RepoFile) -> dict:
    path = snapshot / item.path
    if not path.is_file() or path.stat().st_size != item.size:
        raise RuntimeError(f"Missing or wrong-size file: {path}")
    digest = hashlib.sha256() if item.lfs else hashlib.sha1()
    if not item.lfs:
        digest.update(f"blob {item.size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash(item):
        raise RuntimeError(f"Hash mismatch: {path}")
    print(f"  ok  {item.path}", flush=True)
    return {"file": item.path, "bytes": item.size, "hash": expected_hash(item)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", required=True, help="HF cache dir (files go to <dir>/hub)")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expect-files", type=int, required=True)
    parser.add_argument("--expect-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true", help="skip the download step")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest = sorted(
        (item for item in HfApi().list_repo_tree(args.repo, revision=args.revision,
                                                  recursive=True, expand=True)
         if isinstance(item, RepoFile)),
        key=lambda item: item.path,
    )
    total = sum(item.size for item in manifest)
    print(f"{args.repo} @ {args.revision}: {len(manifest)} files, {total:,} bytes", flush=True)
    if len(manifest) != args.expect_files or total != args.expect_bytes:
        sys.exit(f"The Hub manifest does not match the pinned checkpoint "
                 f"(expected {args.expect_files} files, {args.expect_bytes:,} bytes). Not downloading.")

    hub = Path(args.cache_dir) / "hub"
    snapshot = hub / f"models--{args.repo.replace('/', '--')}" / "snapshots" / args.revision
    if not args.verify_only:
        hub.mkdir(parents=True, exist_ok=True)
        resolved = snapshot_download(repo_id=args.repo, revision=args.revision,
                                     cache_dir=str(hub), max_workers=args.workers)
        if Path(resolved) != snapshot:
            raise RuntimeError(f"Unexpected snapshot path: {resolved}")
        print("Download complete.", flush=True)
    print("Verifying sizes and hashes (reads every byte; this takes a while) ...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(lambda item: verify(snapshot, item), manifest))

    receipt = {
        "repo": args.repo,
        "revision": args.revision,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    path = Path(args.receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({"files": len(files), "total_bytes": receipt["total_bytes"],
                      "receipt": str(path)}), flush=True)


if __name__ == "__main__":
    main()
