# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read one pipeline component's safetensors weights as stored.

SD3.5 Medium stores its denoiser in BF16 and its text encoders in FP16.
Tensors are returned in their stored dtype (``ml_dtypes.bfloat16`` for BF16):
an FP32 build widens them exactly; a BF16 build keeps BF16 tensors bit for bit
and rounds FP16 ones to BF16, as the official pipeline does when it loads them
in BF16.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy for safetensors)
from safetensors import safe_open

# The diffusers and transformers names of the default (non-variant) weights.
_SINGLE_FILES = ("diffusion_pytorch_model.safetensors", "model.safetensors")
_INDEX_FILES = ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json")


def _shards(directory: Path) -> list[Path]:
    for name in _INDEX_FILES:
        index = directory / name
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            return [directory / shard for shard in sorted(set(weight_map.values()))]
    for name in _SINGLE_FILES:
        path = directory / name
        if path.is_file():
            return [path]
    raise FileNotFoundError(f"{directory} has no safetensors weights")


def _open_safetensors(path: Path, keep) -> dict:
    tensors = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            if keep(key):
                tensors[key] = reader.get_tensor(key)
    return tensors


def load_component(model_dir: Path, component: str, prefix: str = "") -> dict:
    """All tensors of ``component`` whose names start with ``prefix``."""
    directory = Path(model_dir) / component
    tensors: dict = {}
    for shard in _shards(directory):
        for key, value in _open_safetensors(shard, lambda key: key.startswith(prefix)).items():
            if key in tensors:
                raise ValueError(f"{component} stores {key} twice")
            tensors[key] = value
    if not tensors:
        raise ValueError(f"{component} has no tensors under {prefix!r}")
    return tensors


def take(weights: dict, name: str):
    """One tensor, with the component context in the error when it is missing."""
    try:
        return weights[name]
    except KeyError:
        raise KeyError(f"stable_diffusion_3 checkpoint lacks {name}") from None
