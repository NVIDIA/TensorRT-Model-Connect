# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy NumPy view over the exact SAM2.1 HOI PyTorch checkpoint."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .source_package import (
    inspect_source_package,
    verify_checkpoint,
    verify_source_patches,
)


class Sam2HoiWeights(Mapping[str, np.ndarray]):
    """Checkpoint tensor mapping with explicit linear/conv layout helpers."""

    def __init__(self, tensors: Mapping[str, Any]) -> None:
        self._tensors = dict(tensors)
        self._cache: dict[str, np.ndarray] = {}

    def __iter__(self) -> Iterator[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._tensors:
            raise KeyError(f"Missing SAM2 HOI checkpoint parameter: {key}")
        cached = self._cache.get(key)
        if cached is None:
            tensor = self._tensors[key]
            if not hasattr(tensor, "detach"):
                raise TypeError(f"SAM2 HOI checkpoint entry is not a tensor: {key}")
            tensor = tensor.detach().cpu()
            if str(tensor.dtype) == "torch.bfloat16":
                tensor = tensor.float()
            cached = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
            self._cache[key] = cached
        return cached

    def linear_weight(self, prefix: str) -> np.ndarray:
        weight = self[f"{prefix}.weight"]
        if weight.ndim != 2:
            raise ValueError(f"Expected rank-2 linear weight {prefix!r}, got {weight.shape}")
        return np.ascontiguousarray(weight.T, dtype=np.float32)

    def linear_bias(self, prefix: str) -> np.ndarray:
        return self[f"{prefix}.bias"]

    def convolution_weight(self, prefix: str) -> np.ndarray:
        weight = self[f"{prefix}.weight"]
        if weight.ndim not in {3, 4, 5}:
            raise ValueError(f"Expected convolution weight {prefix!r}, got rank {weight.ndim}")
        return weight


_REQUIRED_PARAMETERS = (
    "image_encoder.trunk.patch_embed.proj.weight",
    "image_encoder.neck.convs.0.conv.weight",
    "image_encoder.hoi_head.query_head.transformer.query_embed.weight",
    "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
    "sam_mask_decoder.transformer.layers.0.self_attn.q_proj.weight",
    "memory_attention.layers.0.self_attn.q_proj.weight",
    "memory_encoder.mask_downsampler.encoder.0.weight",
)


def load_checkpoint(model_dir: str | Path) -> Sam2HoiWeights:
    package = inspect_source_package(model_dir)
    if package is None:
        raise RuntimeError(f"Not a SAM2 HOI source package: {model_dir}")
    verify_checkpoint(package)
    verify_source_patches(package)

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Loading the SAM2 HOI checkpoint requires PyTorch in the build environment"
        ) from error

    checkpoint = torch.load(package.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model"}:
        raise RuntimeError("SAM2 HOI checkpoint must contain exactly one 'model' state dict")
    state = checkpoint["model"]
    if not isinstance(state, Mapping):
        raise RuntimeError("SAM2 HOI checkpoint 'model' entry is not a state dict")
    missing = [name for name in _REQUIRED_PARAMETERS if name not in state]
    if missing:
        raise RuntimeError(
            "SAM2 HOI checkpoint is missing required parameters: " + ", ".join(missing)
        )
    if len(state) != 1092:
        raise RuntimeError(f"SAM2 HOI checkpoint expected 1092 parameters, found {len(state)}")
    return Sam2HoiWeights(state)
