# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a large checkpoint's host weights in file-backed pages during the TensorRT build (family-owned copy).

TensorRT references the host weight arrays for the whole build while it also places a device copy of
them. On a GPU that shares its memory with the CPU (DGX Spark, Jetson Thor) the two copies of a 27B
FP16 checkpoint (about 54 GB each) do not fit next to the builder's workspace. Moving the large host
arrays into mappings of an anonymous temporary file lets the kernel write them back and evict them when the
device copy needs the memory; the file's space is released when the last mapping is dropped.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator, MutableMapping
from pathlib import Path

import numpy as np

# Arrays below this size are left in place: norms, biases and scales are a negligible share.
SPILL_MIN_BYTES = 64 * 1024**2
# Spill when the host weights exceed this share of physical memory.
SPILL_MEMORY_FRACTION = 1 / 3

# Directory of the current build's spill, or None when the build keeps its weights in anonymous memory.
# Set only inside ``spilled``; graph code that derives large constants (file_backed_empty) follows it.
_spill_directory: Path | None = None


def physical_memory_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def host_weight_bytes(weights: MutableMapping[str, object]) -> int:
    return sum(value.nbytes for value in weights.values() if isinstance(value, np.ndarray))


def should_spill(weights: MutableMapping[str, object], memory_bytes: int | None = None) -> bool:
    memory = physical_memory_bytes() if memory_bytes is None else memory_bytes
    return host_weight_bytes(weights) > memory * SPILL_MEMORY_FRACTION


def spill_large_weights(weights: MutableMapping[str, object], directory: Path) -> int:
    """Replace every array of at least SPILL_MIN_BYTES with a mapping of an anonymous temporary file.

    The mapping is swapped into ``weights`` in place so the anonymous array is released; callers that
    already hold the old array keep their own reference. Returns the number of bytes moved.
    """
    moved = 0
    for name, value in list(weights.items()):
        if not isinstance(value, np.ndarray) or isinstance(value, np.memmap):
            continue
        if value.nbytes < SPILL_MIN_BYTES:
            continue
        mapped = _anonymous_mapping(value.shape, value.dtype, directory)
        mapped[...] = value
        weights[name] = mapped
        moved += value.nbytes
    return moved


def file_backed_empty(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """An uninitialized array, file-backed when the current build spilled and the array is large."""
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if _spill_directory is None or nbytes < SPILL_MIN_BYTES:
        return np.empty(shape, dtype=dtype)
    return _anonymous_mapping(shape, dtype, _spill_directory)


def _anonymous_mapping(shape: tuple[int, ...], dtype: np.dtype, directory: Path) -> np.memmap:
    # An anonymous temporary file (O_TMPFILE where available) is never reachable by name, and the
    # mapping stays valid after the file object is closed.
    with tempfile.TemporaryFile(dir=directory, prefix=".weights-") as backing:
        return np.memmap(backing, dtype=dtype, mode="w+", shape=shape)


def spill_if_large(weights: MutableMapping[str, object], output_path: Path) -> int:
    """Spill the large host weights next to the bundle when they exceed SPILL_MEMORY_FRACTION of memory."""
    global _spill_directory
    _spill_directory = None
    if not should_spill(weights):
        return 0
    _spill_directory = Path(output_path).resolve().parent
    moved = spill_large_weights(weights, _spill_directory)
    print(
        f"Moved {moved / 1024**3:.1f} GiB of host weights to file-backed pages for the build",
        file=sys.stderr,
    )
    return moved


@contextmanager
def spilled(weights: MutableMapping[str, object], output_path: Path) -> Iterator[int]:
    """``spill_if_large`` for the duration of a build; afterwards no spill directory is in effect."""
    global _spill_directory
    try:
        yield spill_if_large(weights, output_path)
    finally:
        _spill_directory = None
