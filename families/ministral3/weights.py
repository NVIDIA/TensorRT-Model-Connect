# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact safetensors weight mapping of a bare ``Ministral3Model`` checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

# Registers the NumPy bfloat16 type that safetensors returns for BF16 tensors.
import ml_dtypes
import numpy as np
from safetensors import safe_open

from .config import ModelConfig

_FLOAT_TYPES = (np.dtype(np.float32), np.dtype(np.float16), np.dtype(ml_dtypes.bfloat16))


def _open_safetensors(model_dir: Path) -> dict[str, object]:
    """Map every tensor name to the NumPy safetensors reader that holds it."""
    single = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"
    if single.is_file():
        reader = safe_open(str(single), framework="numpy")
        return {name: reader for name in reader.keys()}
    if not index_path.is_file():
        raise ValueError(f"ministral3 checkpoint has no model.safetensors in {model_dir}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    readers = {
        shard: safe_open(str(model_dir / shard), framework="numpy")
        for shard in sorted(set(map(str, weight_map.values())))
    }
    return {str(name): readers[str(shard)] for name, shard in weight_map.items()}


def expected_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "embed_tokens.weight": (config.vocab_size, hidden),
        "norm.weight": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (config.attention_size, hidden),
                f"{prefix}.self_attn.k_proj.weight": (config.kv_attention_size, hidden),
                f"{prefix}.self_attn.v_proj.weight": (config.kv_attention_size, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, config.attention_size),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (config.intermediate_size, hidden),
                f"{prefix}.mlp.up_proj.weight": (config.intermediate_size, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, config.intermediate_size),
            }
        )
    return shapes


def load_weights(model_dir: str | Path, config: ModelConfig) -> dict[str, np.ndarray]:
    """Load every tensor of the checkpoint in its stored float dtype.

    The checkpoint must contain exactly the tensors of the configured encoder:
    a missing, extra or misshaped tensor is an error, never silently dropped.
    """
    readers = _open_safetensors(Path(model_dir))
    shapes = expected_shapes(config)
    missing = sorted(set(shapes) - set(readers))
    extra = sorted(set(readers) - set(shapes))
    if missing or extra:
        raise ValueError(
            f"ministral3 checkpoint tensors do not match Ministral3Model: "
            f"missing={missing[:4]} extra={extra[:4]}"
        )
    weights: dict[str, np.ndarray] = {}
    for name, shape in shapes.items():
        tensor = np.asarray(readers[name].get_tensor(name))
        if tensor.shape != shape:
            raise ValueError(f"{name} has shape {tensor.shape}, expected {shape}")
        if tensor.dtype not in _FLOAT_TYPES:
            raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
        weights[name] = tensor
    return weights
