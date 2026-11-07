#!/usr/bin/env python3
"""Verify the pinned checkpoint without loading the tensors into RAM."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--sizes-only', action='store_true')
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'manifests/model.json'
    files = json.loads(manifest.read_text())['files']
    for entry in files:
        rel = Path(entry['file'])
        if rel.is_absolute() or '..' in rel.parts:
            raise SystemExit(f'Unsafe manifest entry: {rel}')
        path = args.directory / rel
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            raise SystemExit(f'Missing file or incorrect size: {path}')
        if args.sizes_only:
            continue
        if entry['algorithm'] == 'git-blob-sha1':
            digest = hashlib.sha1(f'blob {entry["bytes"]}\0'.encode(), usedforsecurity=False)
        elif entry['algorithm'] == 'sha256':
            digest = hashlib.sha256()
        else:
            raise SystemExit(f'Unknown digest algorithm: {entry["algorithm"]}')
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != entry['hash']:
            raise SystemExit(f'Incorrect digest: {path}')
    check = 'sizes' if args.sizes_only else 'sizes and digests'
    print(f'Checked {len(files)} files ({check}).')


if __name__ == '__main__':
    main()
