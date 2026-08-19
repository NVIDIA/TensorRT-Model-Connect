# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loading for the public Diffusers MiniMax-H3 checkpoint."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any, Iterable

import ml_dtypes
import numpy as np


class _TensorOwnedArray(np.ndarray):
    """NumPy view that keeps its source checkpoint tensor alive."""

    _tensor_owner: Any

    def __array_finalize__(self, source) -> None:
        if source is not None:
            self._tensor_owner = getattr(source, "_tensor_owner", None)


def load_selected_component_state_dict(
    component_dir: str | Path, names: Iterable[str]
) -> dict[str, Any]:
    """Load only selected indexed tensors, avoiding unused H3 language/vision weights."""

    from safetensors import safe_open

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(f"Selective loading requires one safetensors index in {root}")
    weight_map = json.loads(indexes[0].read_text())["weight_map"]
    requested = tuple(names)
    missing = sorted(set(requested) - set(weight_map))
    if missing:
        raise ValueError(f"MiniMax-H3 checkpoint is missing tensors: {missing}")
    by_file: dict[str, list[str]] = {}
    for name in requested:
        by_file.setdefault(weight_map[name], []).append(name)
    state: dict[str, Any] = {}
    for filename, tensor_names in sorted(by_file.items()):
        with safe_open(root / filename, framework="pt", device="cpu") as reader:
            for name in tensor_names:
                state[name] = reader.get_tensor(name)
    return state


def validate_component_key_partition(
    component_dir: str | Path, groups: Iterable[Iterable[str]]
) -> None:
    """Require selected groups to partition an indexed component exactly."""

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ValueError(f"Partition validation requires one safetensors index in {root}")
    indexed = set(json.loads(indexes[0].read_text())["weight_map"])
    selected: set[str] = set()
    overlap: set[str] = set()
    for group in groups:
        names = set(group)
        overlap.update(selected & names)
        selected.update(names)
    if overlap:
        raise ValueError(f"MiniMax-H3 checkpoint partitions overlap: {sorted(overlap)}")
    missing = sorted(selected - indexed)
    unassigned = sorted(indexed - selected)
    if missing or unassigned:
        raise ValueError(
            "MiniMax-H3 checkpoint partition is not exhaustive: "
            f"missing={missing}, unassigned={unassigned}"
        )


def numpy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Expose CPU tensors to NumPy without expanding checkpoint-native BF16.

    NumPy does not natively understand PyTorch BF16, so ``Tensor.numpy()`` is
    not available for those tensors.  Viewing the storage as ``uint16`` first
    and then as :mod:`ml_dtypes` BF16 preserves every checkpoint bit while
    keeping the NumPy array zero-copy.  TensorRT consumes that buffer through
    its explicit BF16 ``Weights`` constructor.
    """

    arrays: dict[str, Any] = {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        if str(value.dtype) == "torch.bfloat16":
            # ``Tensor.numpy()`` rejects BF16, so expose its CPU storage as
            # uint16 before reinterpreting the same bits.
            storage_type = ctypes.c_uint16 * value.numel()
            storage = storage_type.from_address(value.data_ptr())
            raw = np.ctypeslib.as_array(storage).reshape(tuple(value.shape)).view(_TensorOwnedArray)
            raw._tensor_owner = value
            arrays[name] = raw.view(ml_dtypes.bfloat16).reshape(tuple(value.shape))
        else:
            arrays[name] = np.asarray(value.numpy())
    return arrays
