#!/usr/bin/env python3
"""CPU-only byte correctness, boundary, residency, and poller tests."""
import json
from pathlib import Path
import tempfile
import time

import torch
from brrr import cuext

ext = cuext.extension('engram_ram')
checks = 0
with tempfile.TemporaryDirectory() as directory:
    count = 32
    source = torch.arange(count * 256, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(count, 256)
    for base in (0, 17, 256, 4095):
        path = Path(directory) / f'offset-{base}.bin'
        path.write_bytes(b'X' * base + bytes(source.flatten().tolist()) + b'Y' * 4096)
        ids = torch.zeros(16, dtype=torch.int64)
        rows = torch.zeros((16, 256), dtype=torch.uint8)
        issued = torch.zeros(1, dtype=torch.int64)
        ready = torch.zeros(1, dtype=torch.int64)
        handle = ext.create_store(str(path), base, count, ids, rows, issued, ready)
        requested = torch.tensor([0, 31, 1, -1, 15, 31, 0], dtype=torch.int64)
        expected = source[requested.clamp(min=0)].clone()
        expected[requested < 0] = 0
        assert torch.equal(ext.test_rows(handle, requested), expected)
        checks += 1
        info = ext.store_info(handle)
        assert info['locked'] and info['source_fd_closed'] and info['resident_bytes'] == count * 256
        checks += 1
        ids[:len(requested)].copy_(requested)
        issued[0] = (1 << 32) | len(requested)
        deadline = time.monotonic() + 5
        while ready.item() != 1 and time.monotonic() < deadline:
            time.sleep(.001)
        assert ready.item() == 1 and torch.equal(rows[:len(requested)], expected)
        checks += 1
        try:
            ext.test_rows(handle, torch.tensor([count], dtype=torch.int64))
            raise AssertionError('Out-of-bounds row accepted')
        except RuntimeError:
            checks += 1
print(json.dumps({'cpu_checks_passed': checks}), flush=True)
