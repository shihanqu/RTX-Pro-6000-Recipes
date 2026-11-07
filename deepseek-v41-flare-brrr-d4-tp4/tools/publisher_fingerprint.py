#!/usr/bin/env python3
"""Derive the exact allowlist fingerprint from a locally generated public profile."""
import hashlib
import json
from pathlib import Path
import sys

profiles = json.loads(Path(sys.argv[1]).read_text())
if not isinstance(profiles, list) or len(profiles) != 1:
    raise SystemExit('Expected exactly one local-development publisher profile.')
profile = profiles[0]
if not isinstance(profile, dict):
    raise SystemExit('Invalid publisher profile.')
key = profile.get('public_key')
if (profile.get('namespace') != 'local-development' or profile.get('development') is not True
        or not isinstance(key, list) or len(key) != 32
        or any(type(n) is not int or not 0 <= n <= 255 for n in key)):
    raise SystemExit('Invalid local-development Ed25519 public key.')
print(hashlib.sha256(bytes(key)).hexdigest())
