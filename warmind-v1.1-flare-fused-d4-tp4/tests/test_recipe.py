# The harnesses are also standalone CLIs; import from their native search paths.
# ruff: noqa: E402
import base64
import hashlib
import ipaddress
import json
import statistics
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "tools")]

import benchmark
import validate

import recipe
from tools import catalog, checkpoint, delta


def sha(x):
    return hashlib.sha256(x).hexdigest()


def catalog_fixture():
    graphs = [("packed-speculative1x5-drafts4-tokens5", 925)]
    graphs.extend(
        (f"packed-speculative2x5-drafts4-tokens{i}", n)
        for i, n in zip((5, 6, 7, 8, 10), (916, 910, 910, 908, 1700))
    )
    return dict(
        package=dict(signatures_verified=True, publisher=dict(fingerprint="test-signer"), tensor_parallel=4),
        model=dict(
            model_id=checkpoint.ROOT_ID, max_active=32, max_sequence_tokens=1048576, prefill_chunk_size=4096
        ),
        vision=dict(feature_slots=4),
        speculation=dict(token_capacity=128),
        ranks=[
            dict(
                rank=i,
                architecture=[12, 0],
                execution_graphs=[dict(name=name, node_count=n, draft_tokens=4) for name, n in graphs],
            )
            for i in range(4)
        ],
    )


class CatalogTests(unittest.TestCase):
    def test_compact_and_wide_variants_on_four_ranks(self):
        self.assertEqual(
            catalog.check_catalog(catalog_fixture(), "test-signer", checkpoint.ROOT_ID),
            [[925, 916, 910, 910, 908]] * 4,
        )

    def test_mixed_depth_stock_missing_vision_or_unverified_package(self):
        for path, value in [
            (("ranks", 3, "execution_graphs", 2, "draft_tokens"), 7),
            (("ranks", 1, "architecture"), [9, 0]),
            (("ranks", 2, "execution_graphs", 5, "node_count"), 900),
            (("ranks", 2, "execution_graphs", 0, "node_count"), 1500),
            (("ranks", 0, "rank"), 1),
            (("model", "model_id"), "stock"),
            (("vision", "feature_slots"), 0),
            (("package", "signatures_verified"), False),
            (("package", "publisher", "fingerprint"), "different"),
        ]:
            with self.subTest(path=path):
                obj = catalog_fixture()
                dest = obj
                for key in path[:-1]:
                    dest = dest[key]
                dest[path[-1]] = value
                with self.assertRaises(ValueError):
                    catalog.check_catalog(obj, "test-signer", checkpoint.ROOT_ID)

    def test_only_public_fields_exported(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "profile.json"
            public = list(range(32))
            profile = dict(
                namespace="local-development",
                development=True,
                public_key=public,
                private_example="not for serving",
            )
            p.write_text(json.dumps([profile]))
            stripped, fingerprint = catalog.publisher(p)
            self.assertEqual(set(stripped), {"namespace", "name", "public_key", "development"})
            self.assertEqual(fingerprint, sha(bytes(public)))
            profile["public_key"] = [True] * 32
            p.write_text(json.dumps([profile]))
            with self.assertRaises(ValueError):
                catalog.publisher(p)


class CheckpointTests(unittest.TestCase):
    def test_native_manifest_and_edited_samples(self):
        m = json.loads(checkpoint.MANIFEST.read_text())
        self.assertEqual((m["repository"], m["revision"]), (checkpoint.REPO, checkpoint.REV))
        self.assertEqual((len(m["files"]), sum(f["bytes"] for f in m["files"])), (88, 510318572683))
        self.assertEqual(len([f for f in m["files"] if f["file"].endswith(".safetensors")]), 48)
        sampled = json.loads((HERE / "manifests/edited-layer-samples.json").read_text())["files"]
        self.assertEqual(len({f["file"] for f in sampled}), 12)
        self.assertTrue(all(f["bytes"] > 0 and len(bytes.fromhex(f["sha256"])) == 32 for f in sampled))

    def test_receipt_requires_unchanged_bytes_and_matching_directory(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t).resolve() / "native"
            root.mkdir()
            receipt = root.parent / "receipt.json"
            shard = root / "weights"
            shard.write_bytes(b"test")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"layer": "weights"}}))
            files = {
                p.name: {"bytes": p.stat().st_size, "algorithm": "sha256", "hash": sha(p.read_bytes())}
                for p in (index, shard)
            }

            def verify(full=False):
                checkpoint.verify_files(
                    root, receipt, files, "manifest", full=full, expected_shards={"weights"}
                )

            verify(True)
            with patch.object(
                checkpoint, "digest_file", side_effect=AssertionError("unchanged files were reread")
            ):
                verify()
            shard.write_bytes(b"fail")
            with self.assertRaisesRegex(ValueError, "Digest mismatch"):
                verify()
            shard.write_bytes(b"test")
            (root / "extra").touch()
            with self.assertRaisesRegex(ValueError, "extra"):
                verify()
            (root / "extra").unlink()
            shard.unlink()
            shard.symlink_to(index)
            with self.assertRaisesRegex(ValueError, "symlink"):
                verify()

    def test_delta_preserves_base_and_authenticates_full_output(self):
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "base"
            dst = Path(t) / "derived"
            header = json.dumps(
                {
                    "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                    "b": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]},
                }
            ).encode()
            prefix = struct.pack("<Q", len(header)) + header
            pristine = prefix + b"abcdefgh"
            expected = prefix + b"abcdEDIT"
            src.write_bytes(pristine)
            change = {
                "tensor": "b",
                "byte_offset": len(prefix) + 4,
                "byte_length": 4,
                "pristine_sha256": sha(b"efgh"),
                "derivative_sha256": sha(b"EDIT"),
            }
            old = {"size": len(pristine), "lfs": {"sha256": sha(pristine)}}
            entry = {"bytes": len(pristine), "hash": sha(expected)}

            def make(changes, meta, payload):
                delta.assemble(src, dst, changes, meta, old, lambda offset, length: payload)

            make([change], entry, b"EDIT")
            self.assertEqual(dst.read_bytes(), expected)
            for changes, meta, payload in [
                ([change, change], entry, b"EDIT"),
                ([change], entry, b"BAD!"),
                ([change], {**entry, "hash": "wrong"}, b"EDIT"),
                ([{**change, "byte_offset": 0}], entry, b"EDIT"),
            ]:
                with self.assertRaises(ValueError):
                    make(changes, meta, payload)
            self.assertEqual(src.read_bytes(), pristine)


class BenchmarkTests(unittest.TestCase):
    def test_filters_foreign_traffic_bad_rates_and_bad_occupancy(self):
        row = {
            "concurrency": 2,
            "aggregate_source": "openai_continuous_usage",
            "aggregate_tps": 500,
            "server_gen_throughput": 500,
            "avg_running_reqs": 2,
            "max_running_reqs": 2,
            "max_queue_reqs": 0,
            "measurement_wall_seconds": 30,
        }
        audit = {"quiet_at_entry": True, "client_requests": {"chat": 2}, "server_requests": {"chat": 2}}
        self.assertEqual(benchmark.issues(row, audit), [])
        for key, value in [
            ("num_errors", 1),
            ("loop_detected", True),
            ("underfilled", True),
            ("aggregate_tps", 400),
            ("server_gen_throughput", float("nan")),
            ("avg_running_reqs", 1),
            ("max_running_reqs", 3),
            ("max_queue_reqs", 1),
            ("measurement_wall_seconds", 25),
            ("aggregate_source", "other"),
        ]:
            with self.subTest(key=key):
                self.assertTrue(benchmark.issues({**row, key: value}, audit))
        for other in [
            None,
            {**audit, "quiet_at_entry": False},
            {**audit, "server_requests": {"chat": 3}},
            {**audit, "client_requests": {}, "server_requests": {}},
        ]:
            self.assertIn("HTTP audit", benchmark.issues(row, other))

    def test_published_cells_are_derived_from_the_listed_windows(self):
        r = json.loads((HERE / "results/d4-vs-d7.json").read_text())
        self.assertEqual(
            {(v["context_tokens"], v["concurrency"]) for v in r["rows"]},
            {(c, n) for c in (0, 32768) for n in (1, 2, 4, 8, 16, 32)},
        )
        self.assertEqual(sum(v["d4"]["n"] for v in r["rows"]), r["d4_windows"])
        self.assertEqual((r["d4_windows"], r["d4_windows_rejected"]), (64, 0))
        for row in r["rows"]:
            for tag in ("d4", "d7"):
                block = row[tag]
                windows = block["windows_tok_s"]
                self.assertEqual(len(windows), block["n"])
                self.assertAlmostEqual(statistics.mean(windows), block["mean"])
                if len(windows) > 1:
                    self.assertAlmostEqual(statistics.stdev(windows), block["sample_sd"])
            self.assertAlmostEqual(
                100 * (row["d4"]["mean"] / row["d7"]["mean"] - 1), row["d4_change_percent"]
            )

    def test_vision_fixtures_are_distinct_valid_pngs(self):
        fixtures = set()
        for color, number in zip(validate.COLORS, [17, 29, 42, 63]):
            img = validate.fixture(color, number)["image_url"]["url"]
            raw = base64.b64decode(img.split(",", 1)[1], validate=True)
            self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", raw[16:24]), (160, 160))
            fixtures.add(raw)
        self.assertEqual(len(fixtures), 4)


class GuardTests(unittest.TestCase):
    def test_workspace_refuses_broad_or_shared_destination(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t).resolve()
            recipe.private_workspace(root / "work", root / "weights")
            self.assertEqual((root / "work").stat().st_mode & 0o777, 0o700)
            for work, native in [
                (root, root / "weights"),
                (root / "work", Path.home()),
                (root / "work", root / "work"),
            ]:
                with self.assertRaises(ValueError):
                    recipe.private_workspace(work, native)
            (root / "work").chmod(0o755)
            with self.assertRaises(ValueError):
                recipe.private_workspace(root / "work", root / "weights")

    def test_gpu_order_power_and_busy_guard(self):
        labels = ["GPU-fixture-" + x for x in "dbca"]
        selection = ",".join(labels)
        inventory = "\n".join(
            f"{x}, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 450" for x in sorted(labels)
        )

        def smi(*args):
            return inventory if "power.limit" in args[1] else ""

        env = {"GPU_IDS": selection}
        with patch.object(recipe, "output", side_effect=smi):
            self.assertEqual(recipe.gpu_env(env, idle=True), labels)
            self.assertEqual([env[f"GPU{i}"] for i in range(4)], labels)
            with self.assertRaises(ValueError):
                recipe.gpu_env({**env, "EXPECTED_WATTS": "395"}, idle=False)
            with self.assertRaises(ValueError):
                recipe.gpu_env({"GPU_IDS": ",".join([labels[0]] * 4)}, idle=False)
        with patch.object(recipe, "output", side_effect=[inventory, labels[0] + ", 42"]):
            with self.assertRaisesRegex(ValueError, "busy"):
                recipe.gpu_env(env, idle=True)

    def test_port_and_loopback_guards(self):
        host, container = recipe.bindings()
        self.assertTrue(ipaddress.ip_address(host).is_loopback)
        self.assertTrue(ipaddress.ip_address(container).is_unspecified)
        with patch.object(recipe, "output", return_value="occupied"):
            with self.assertRaisesRegex(ValueError, "occupied"):
                recipe.free_resources(1230)
        with patch.object(recipe, "output", side_effect=["", "published"]):
            with self.assertRaisesRegex(ValueError, "occupied"):
                recipe.free_resources(1230)


if __name__ == "__main__":
    unittest.main()
