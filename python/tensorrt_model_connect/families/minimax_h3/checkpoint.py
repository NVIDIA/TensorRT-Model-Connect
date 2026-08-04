# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loading for the public Diffusers MiniMax-H3 checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_component_state_dict(component_dir: str | Path) -> dict[str, Any]:
    """Load a safetensors component without materializing duplicate tensors."""

    from safetensors.torch import load_file

    root = Path(component_dir)
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) != 1:
            raise ValueError(f"Expected one safetensors index in {root}, found {len(indexes)}")
        weight_map = json.loads(indexes[0].read_text())["weight_map"]
        paths = [root / name for name in sorted(set(weight_map.values()))]
    else:
        paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors checkpoint found in {root}")

    state: dict[str, Any] = {}
    for path in paths:
        for name, tensor in load_file(path, device="cpu").items():
            if name in state:
                raise ValueError(f"Duplicate MiniMax-H3 tensor {name!r}")
            state[name] = tensor
    return state


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


def require_keys(state: dict[str, Any], names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(state))
    if missing:
        raise ValueError(f"MiniMax-H3 checkpoint is missing tensors: {missing}")


def numpy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert tensors only after checkpoint validation, preserving FP32 values."""

    return {name: tensor.detach().float().cpu().numpy() for name, tensor in state.items()}
