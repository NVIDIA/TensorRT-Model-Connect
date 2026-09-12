# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict, family-local mapping of flow.pt estimator tensors to native TRT."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import numpy as np

from .config import FlowConfig


def expected_shapes(cfg: FlowConfig) -> dict[str, tuple[int, ...]]:
    shapes = {}

    def linear(name, out_dim, in_dim):
        shapes[name + ".weight"] = (out_dim, in_dim)
        shapes[name + ".bias"] = (out_dim,)

    d = cfg.dim
    linear("time_embed.time_mlp.0", d, cfg.time_dim)
    linear("time_embed.time_mlp.2", d, d)
    linear("input_embed.proj", d, 3 * cfg.mel_dim + cfg.spk_dim)
    for i in (1, 2):
        key = f"input_embed.conv_pos_embed.conv{i}.0"
        shapes[key + ".weight"] = (d, d // cfg.conv_groups, cfg.conv_kernel)
        shapes[key + ".bias"] = (d,)
    for i in range(cfg.depth):
        prefix = f"transformer_blocks.{i}."
        linear(prefix + "attn_norm.linear", 6 * d, d)
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            linear(prefix + "attn." + name, d, d)
        linear(prefix + "ff.ff.0.0", cfg.ff_mult * d, d)
        linear(prefix + "ff.ff.2", d, cfg.ff_mult * d)
    linear("norm_out.linear", 2 * d, d)
    linear("proj_out", cfg.mel_dim, d)
    return shapes


def validate_weights(weights: Mapping, cfg: FlowConfig) -> dict[str, np.ndarray]:
    expected = expected_shapes(cfg)
    missing = sorted(expected.keys() - weights.keys())
    extra = sorted(weights.keys() - expected.keys() - {"rotary_embed.inv_freq"})
    if missing or extra:
        raise ValueError(f"Estimator checkpoint keys mismatch: missing={missing}, unexpected={extra}")
    result = {}
    for key, shape in expected.items():
        value = weights[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().float().numpy()
        array = np.asarray(value)
        if array.shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {array.shape}")
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError(f"{key}: expected finite floating-point weights")
        result[key] = np.ascontiguousarray(array, dtype=np.float32)
    if "rotary_embed.inv_freq" in weights:
        inv = np.asarray(weights["rotary_embed.inv_freq"])
        expected_inv = 1.0 / 10000 ** (np.arange(0, cfg.head_dim, 2) / cfg.head_dim)
        if inv.shape != expected_inv.shape or not np.allclose(inv, expected_inv, rtol=1e-6, atol=1e-8):
            raise ValueError("Unsupported rotary embedding frequencies")
        result["rotary_embed.inv_freq"] = np.ascontiguousarray(inv, dtype=np.float32)
    return result


def load_flow_weights(model_dir: str | Path, cfg: FlowConfig) -> dict[str, np.ndarray]:
    import torch

    # No HyperPyYAML constructors, arbitrary pickle execution, GPU allocation,
    # downloaded model code, or mutation of the original checkpoint.
    state = torch.load(Path(model_dir) / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping):
        raise ValueError("flow.pt must contain a state dict")
    prefix = "decoder.estimator."
    estimator = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    return validate_weights(estimator, cfg)
