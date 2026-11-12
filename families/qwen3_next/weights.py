# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, mmap-backed access to BF16 safetensors without allocating the model."""

from contextlib import contextmanager
import json
import mmap
from pathlib import Path
import struct


class Weights:
    def __init__(self, directory: Path):
        self.directory = directory
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.files = index["weight_map"]
        self.headers = {}
        self.consumed = set()

    def describe(self, name):
        filename = self.files[name]
        if filename not in self.headers:
            with (self.directory / filename).open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if size > 100 * 1024 * 1024:
                    raise ValueError("oversized safetensors header")
                self.headers[filename] = (8 + size, json.loads(stream.read(size)))
        offset, header = self.headers[filename]
        entry = header[name]
        start, end = entry["data_offsets"]
        if entry["dtype"] not in {"BF16", "F32"}:
            raise ValueError(f"unsupported checkpoint dtype for {name}: {entry['dtype']}")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError(f"invalid safetensors offsets for {name}")
        elements = 1
        for dimension in entry["shape"]:
            elements *= dimension
        if end - start != elements * (2 if entry["dtype"] == "BF16" else 4):
            raise ValueError(f"invalid safetensors range for {name}")
        if offset + end > (self.directory / filename).stat().st_size:
            raise ValueError(f"truncated checkpoint shard for {name}")
        return filename, offset + start, end - start, entry

    @contextmanager
    def view(self, name):
        filename, start, size, _ = self.describe(name)
        with (self.directory / filename).open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                view = memoryview(mapping)[start : start + size]
                try:
                    self.consumed.add(name)
                    yield view
                finally:
                    view.release()

    def copy(self, name, stream):
        with self.view(name) as view:
            stream.write(view)

    def finish(self):
        missing = self.files.keys() - self.consumed
        if missing:
            raise ValueError(f"unmapped Qwen3Next checkpoint weights: {sorted(missing)[:10]}")
