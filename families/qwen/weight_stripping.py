# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen's named placeholders and streaming, self-contained refit sections."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import numpy as np

DEFAULT_MIN_STRIP_BYTES = 1 << 20
_COPY_BYTES = 1 << 20
_ACTIVE: ContextVar[StripSession | None] = ContextVar("qwen_weight_stripping", default=None)


class StripSession:
    """Names and source values omitted from one serialized engine."""

    def __init__(self, min_bytes: int = DEFAULT_MIN_STRIP_BYTES) -> None:
        self.min_bytes = min_bytes
        self.entries: dict[str, object] = {}

    def should_strip(self, name: str | None, nbytes: int) -> bool:
        return bool(name) and nbytes >= self.min_bytes

    def record(self, name: str, values) -> None:
        if name in self.entries and not same_value(self.entries[name], values):
            raise ValueError(f"conflicting refit weight {name!r}")
        self.entries[name] = values

    @property
    def total_bytes(self) -> int:
        return sum(value.nbytes for value in self.entries.values())


def same_value(first, second) -> bool:
    """Compare source identity or array bytes, without materializing lazy tensors."""
    if first is second:
        return True
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        if first.dtype != second.dtype or first.shape != second.shape:
            return False
        a, b = memoryview(first.view(np.uint8)).cast("B"), memoryview(second.view(np.uint8)).cast("B")
        return all(a[i:i + _COPY_BYTES] == b[i:i + _COPY_BYTES]
                   for i in range(0, len(a), _COPY_BYTES))
    return first == second if type(first) is type(second) else False


def write_sections(sessions: dict[str, StripSession], writer) -> None:
    """Store one raw weight payload with explicit bindings for each engine."""
    from .checkpoint_mapper import ST_DTYPE_NAMES, TensorRef

    if set(sessions) != {"prefill", "decode"}:
        raise ValueError("stripped Qwen requires both prefill and decode weights")
    merged = StripSession()
    engines = {}
    for role, session in sorted(sessions.items()):
        if not session.entries:
            raise ValueError(f"empty refit weights for {role}")
        engines["prefill.plan" if role == "prefill" else "engine.plan"] = sorted(session.entries)
        for name, value in session.entries.items():
            merged.record(name, value)
    metadata = {}
    locations = {}
    with writer.open_section("refit_weights") as output:
        for name, value in sorted(merged.entries.items()):
            dtype = ST_DTYPE_NAMES.get(np.dtype(value.dtype))
            if dtype is None or math.prod(value.shape) * value.dtype.itemsize != value.nbytes:
                raise ValueError(f"invalid refit tensor {name}")
            identity = value if isinstance(value, TensorRef) else id(value)
            if identity in locations:
                metadata[name] = dict(locations[identity])
                continue
            padding = (-output.tell()) % 64
            output.write(b"\0" * padding)
            metadata[name] = {"offset": output.tell(), "nbytes": value.nbytes,
                              "dtype": dtype, "shape": list(value.shape)}
            locations[identity] = metadata[name]
            if isinstance(value, TensorRef):
                with value.path.open("rb") as source:
                    source.seek(value.offset)
                    remaining = value.nbytes
                    while remaining:
                        chunk = source.read(min(remaining, _COPY_BYTES))
                        if not chunk:
                            raise ValueError(f"truncated checkpoint tensor {name}")
                        output.write(chunk)
                        remaining -= len(chunk)
            else:
                if not value.flags.c_contiguous:
                    raise ValueError(f"noncontiguous refit tensor {name}")
                data = memoryview(value.view(np.uint8)).cast("B")
                for begin in range(0, len(data), _COPY_BYTES):
                    output.write(data[begin:begin + _COPY_BYTES])
    writer.add_json("refit.json", {"version": 1, "engines": engines, "weights": metadata})


def active() -> StripSession | None:
    return _ACTIVE.get()


@contextmanager
def stripping(session: StripSession) -> Iterator[StripSession]:
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)
