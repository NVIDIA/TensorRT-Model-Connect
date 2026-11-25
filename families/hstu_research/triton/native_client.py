# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned NumPy transport to the native test bridge; no PyTorch at runtime."""

import ctypes
import json
import os
from pathlib import Path
import struct
import tempfile
import threading

import numpy as np

from .generate_repository import runtime_metadata


def snapshot_metadata(snapshot):
    """Read metadata for routing/revisions. Native restore performs integrity validation."""
    if len(snapshot) < 16 or snapshot[:8] != b"HSTUKV1\0":
        raise ValueError("invalid history snapshot")
    length = struct.unpack_from("<Q", snapshot, 8)[0]
    if not 0 < length <= 65536 or length > len(snapshot) - 16:
        raise ValueError("invalid snapshot metadata length")
    return json.loads(snapshot[16 : 16 + length])


def save_snapshot(path, snapshot):
    """Atomically publish a complete file; existing readers retain their old inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix="." + path.name, delete=False
        ) as stream:
            temporary = stream.name
            stream.write(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class NativeClient:
    def __init__(
        self,
        bundle,
        runtime_root,
        *,
        device=0,
        hydration=False,
        kv_cache_size_bytes=None,
        kv_cache_memory_ratio=None,
        max_entries=None,
        ttl_seconds=None,
        lease_seconds=None,
    ):
        if not hydration and any(
            value is not None for value in (max_entries, ttl_seconds, lease_seconds)
        ):
            raise ValueError("history cache settings require hydration=True")
        self.lock = threading.RLock()
        self.stats_lock = threading.Lock()
        self.handle = None
        self.metadata = runtime_metadata(Path(bundle))
        self.library = ctypes.CDLL(str(Path(runtime_root) / "libhstu_research_triton.so"))
        for name, arguments, result in [
            ("hstu_research_create", [ctypes.c_char_p], ctypes.c_void_p),
            ("hstu_research_score", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_char_p),
            ("hstu_research_history", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_char_p),
            ("hstu_research_cache_stats", [ctypes.c_void_p], ctypes.c_char_p),
            ("hstu_research_last_error", [], ctypes.c_char_p),
            ("hstu_research_destroy", [ctypes.c_void_p], None),
        ]:
            function = getattr(self.library, name)
            function.argtypes, function.restype = arguments, result
        options = dict(bundle=str(bundle), runtime_root=str(runtime_root), device=device)
        if kv_cache_size_bytes is not None:
            options["kv_cache_size_bytes"] = kv_cache_size_bytes
        if kv_cache_memory_ratio is not None:
            options["kv_cache_memory_ratio"] = kv_cache_memory_ratio
        if hydration:
            max_entries = 32 if max_entries is None else max_entries
            ttl_seconds = 300 if ttl_seconds is None else ttl_seconds
            options.update(
                kv_cache_hydration_enabled=True,
                kv_cache_max_entries=max_entries,
                kv_cache_ttl_seconds=ttl_seconds,
                kv_cache_lease_seconds=min(30, ttl_seconds)
                if lease_seconds is None
                else lease_seconds,
            )
        self.handle = self.library.hstu_research_create(
            json.dumps(options, allow_nan=False).encode()
        )
        if not self.handle:
            raise RuntimeError(self.library.hstu_research_last_error().decode())

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        with self.lock, self.stats_lock:
            if self.handle:
                self.library.hstu_research_destroy(self.handle)
                self.handle = None

    def _check(self, value):
        if not value:
            raise RuntimeError(self.library.hstu_research_last_error().decode())
        return json.loads(value)

    def _ready(self):
        if not self.handle:
            raise RuntimeError("native client is closed")

    @staticmethod
    def _inputs(inputs):
        arrays = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
        specs = [
            dict(
                name=name,
                dtype=str(value.dtype),
                shape=list(value.shape),
                bytes=value.nbytes,
                pointer=value.ctypes.data,
            )
            for name, value in arrays.items()
        ]
        return arrays, specs

    @staticmethod
    def _outputs(specs):
        result = {}
        for spec in specs:
            data = (ctypes.c_uint8 * spec["bytes"]).from_address(spec["pointer"])
            result[spec["name"]] = (
                np.frombuffer(data, dtype=spec["dtype"]).reshape(spec["shape"]).copy()
            )
        return result

    def _history(self, operation, **arguments):
        self._ready()
        command = dict(operation=operation, **arguments)
        return self._check(
            self.library.hstu_research_history(
                self.handle, json.dumps(command, allow_nan=False).encode()
            )
        )

    def score(self, inputs):
        with self.lock:
            self._ready()
            arrays, specs = self._inputs(inputs)
            return self._outputs(
                self._check(
                    self.library.hstu_research_score(self.handle, json.dumps(specs).encode())
                )
            )

    def export_history(self, inputs, revisions):
        with self.lock:
            arrays, specs = self._inputs(inputs)
            result = self._history("export", inputs=specs, revisions=revisions)
            return ctypes.string_at(result["pointer"], result["bytes"])

    def restore_history(self, snapshot):
        with self.lock:
            buffer = ctypes.create_string_buffer(snapshot)
            result = self._history("restore", pointer=ctypes.addressof(buffer), bytes=len(snapshot))
            return result["handles"]

    def score_with_history(self, handles, revisions, inputs):
        with self.lock:
            arrays, specs = self._inputs(inputs)
            result = self._history("score", handles=handles, revisions=revisions, inputs=specs)
            return self._outputs(result["outputs"])

    def history_info(self, handle):
        with self.lock:
            return self._history("info", handle=handle)

    def release_history(self, handles):
        with self.lock:
            return self._history("release", handles=handles)

    def stats(self):
        with self.stats_lock:
            self._ready()
            return self._check(self.library.hstu_research_cache_stats(self.handle))
