"""Check the local publisher and the structural signature of the qualified D4 catalog."""

import hashlib
import json


def publisher(path):
    profiles = json.loads(path.read_text())
    if not isinstance(profiles, list) or len(profiles) != 1:
        raise ValueError("Expected one local development publisher")
    p = profiles[0]
    key = p.get("public_key")
    if (
        p.get("namespace") != "local-development"
        or p.get("development") is not True
        or not isinstance(key, list)
        or len(key) != 32
        or any(type(x) is not int or not 0 <= x < 256 for x in key)
    ):
        raise ValueError("Invalid Ed25519 development publisher profile")
    # Project only the public fields into the serving mount; never mount the settings directory.
    return {
        "name": "Local recipe export",
        "namespace": "local-development",
        "development": True,
        "public_key": key,
    }, hashlib.sha256(bytes(key)).hexdigest()


def check_catalog(meta, signer, root):
    package, model = meta["package"], meta["model"]
    if (
        package["signatures_verified"] is not True
        or package["publisher"]["fingerprint"] != signer
        or package["tensor_parallel"] != 4
    ):
        raise ValueError("Invalid package signature, signer, or TP size")
    if (
        (model["model_id"], model["max_active"], model["max_sequence_tokens"], model["prefill_chunk_size"])
        != (root, 32, 1048576, 4096)
        or meta["vision"]["feature_slots"] != 4
        or meta["speculation"]["token_capacity"] != 128
    ):
        raise ValueError("Not the expected WARMIND checkpoint, capacity, or vision layout")
    if sorted(r["rank"] for r in meta["ranks"]) != list(range(4)):
        raise ValueError("Expected all four TP ranks")
    result = []
    for rank in sorted(meta["ranks"], key=lambda r: r["rank"]):
        graphs = rank["execution_graphs"]
        if rank["architecture"] != [12, 0] or {
            g["draft_tokens"] for g in graphs if g.get("draft_tokens")
        } != {4}:
            raise ValueError("Expected SM120 and four learned draft tokens on every rank")
        nodes = {g["name"]: g["node_count"] for g in graphs}
        compact = [nodes["packed-speculative1x5-drafts4-tokens5"]]
        compact += [nodes[f"packed-speculative2x5-drafts4-tokens{i}"] for i in range(5, 9)]
        if (
            min(compact) <= 0
            or max(compact) >= 1200
            or nodes["packed-speculative2x5-drafts4-tokens10"] <= max(compact)
        ):
            raise ValueError("Expected compact C1/C2 verification graphs and a distinct wider C2 graph")
        result.append(compact)
    return result
