# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact widening of the original BF16/FP16 safetensors weights to FP32."""

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .config import Config


def load_weights(directory: Path, config: Config) -> dict[str, np.ndarray]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
        shards = set(mapping.values())
        if any(Path(name).name != name for name in shards):
            raise ValueError("cohere2 safetensors shard must be a local filename")
    else:
        shards = {"model.safetensors"}
    result = {}
    for shard in sorted(shards):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                value = reader.get_tensor(name)
                if not value.is_floating_point():
                    raise ValueError(f"cohere2 tensor {name} must be floating point")
                if name in result:
                    raise ValueError(f"cohere2 duplicate tensor {name}")
                result[name] = value.float().numpy()
    h, d = config.hidden_size, config.head_dim
    shapes = {"model.embed_tokens.weight": (config.vocab_size, h), "model.norm.weight": (h,)}
    if not config.tie_word_embeddings or "lm_head.weight" in result:
        shapes["lm_head.weight"] = (config.vocab_size, h)
    for i in range(config.num_hidden_layers):
        base = f"model.layers.{i}"
        shapes[f"{base}.input_layernorm.weight"] = (h,)
        for key, shape in {
            "self_attn.q_proj": (h, h),
            "self_attn.k_proj": (config.num_key_value_heads * d, h),
            "self_attn.v_proj": (config.num_key_value_heads * d, h),
            "self_attn.o_proj": (h, h),
            "mlp.gate_proj": (config.intermediate_size, h),
            "mlp.up_proj": (config.intermediate_size, h),
            "mlp.down_proj": (h, config.intermediate_size),
        }.items():
            shapes[f"{base}.{key}.weight"] = shape
    if set(result) != set(shapes):
        raise ValueError(
            f"cohere2 checkpoint keys disagree: {sorted(set(result) ^ set(shapes))[:12]}"
        )
    for name, shape in shapes.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"cohere2 invalid tensor {name}, expected {shape}")
    if config.tie_word_embeddings:
        if "lm_head.weight" in result and not np.array_equal(
            result["lm_head.weight"], result["model.embed_tokens.weight"]
        ):
            raise ValueError("cohere2 tied LM head differs from embedding")
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
    return result
