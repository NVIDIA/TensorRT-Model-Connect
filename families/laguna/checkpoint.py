# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna safetensors reader.

Reads the Hugging Face layout exactly as published (per-expert projections,
``mlp.shared_expert.*``, ``mlp.experts.e_score_correction_bias``) and returns
the tensors the engine graph consumes, in the build's storage dtype. Linear
weights keep the checkpoint's ``[out, in]`` layout; the graph multiplies by
their transpose. Routed experts are stacked per layer into ``[experts, out,
in]`` blocks. Router weights and the selection bias are kept in FP32.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

# Register NumPy bfloat16 support required by safetensors.
import ml_dtypes
import numpy as np
from safetensors import safe_open

from .config import LagunaConfig


BF16 = np.dtype(ml_dtypes.bfloat16)


def _open_safetensors(model_dir: Path) -> dict[str, object]:
    """Map every tensor name to its NumPy safetensors reader."""
    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json has no weight_map")
        readers = {
            shard: safe_open(str(model_dir / shard), framework="numpy")
            for shard in sorted(set(str(value) for value in weight_map.values()))
        }
        return {str(name): readers[str(shard)] for name, shard in weight_map.items()}
    if single.is_file():
        reader = safe_open(str(single), framework="numpy")
        return {name: reader for name in reader.keys()}
    raise FileNotFoundError(f"No Laguna safetensors checkpoint in {model_dir}")


def expected_tensors(config: LagunaConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor name of a Laguna checkpoint with its shape."""
    hidden = config.hidden_size
    head_dim = config.head_dim
    kv = config.kv_width
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, hidden)
    for layer in range(config.num_hidden_layers):
        hf = f"model.layers.{layer}"
        heads = config.heads_per_layer[layer]
        shapes.update(
            {
                f"{hf}.input_layernorm.weight": (hidden,),
                f"{hf}.post_attention_layernorm.weight": (hidden,),
                f"{hf}.self_attn.q_proj.weight": (heads * head_dim, hidden),
                f"{hf}.self_attn.k_proj.weight": (kv, hidden),
                f"{hf}.self_attn.v_proj.weight": (kv, hidden),
                f"{hf}.self_attn.o_proj.weight": (hidden, heads * head_dim),
                f"{hf}.self_attn.g_proj.weight": (heads, hidden),
                f"{hf}.self_attn.q_norm.weight": (head_dim,),
                f"{hf}.self_attn.k_norm.weight": (head_dim,),
            }
        )
        if config.mlp_layer_types[layer] == "dense":
            inner = config.intermediate_size
            shapes.update(
                {
                    f"{hf}.mlp.gate_proj.weight": (inner, hidden),
                    f"{hf}.mlp.up_proj.weight": (inner, hidden),
                    f"{hf}.mlp.down_proj.weight": (hidden, inner),
                }
            )
            continue
        inner = config.moe_intermediate_size
        shared = config.shared_expert_intermediate_size
        shapes[f"{hf}.mlp.gate.weight"] = (config.num_experts, hidden)
        shapes[f"{hf}.mlp.experts.e_score_correction_bias"] = (config.num_experts,)
        for expert in range(config.num_experts):
            prefix = f"{hf}.mlp.experts.{expert}"
            shapes[f"{prefix}.gate_proj.weight"] = (inner, hidden)
            shapes[f"{prefix}.up_proj.weight"] = (inner, hidden)
            shapes[f"{prefix}.down_proj.weight"] = (hidden, inner)
        shapes[f"{hf}.mlp.shared_expert.gate_proj.weight"] = (shared, hidden)
        shapes[f"{hf}.mlp.shared_expert.up_proj.weight"] = (shared, hidden)
        shapes[f"{hf}.mlp.shared_expert.down_proj.weight"] = (hidden, shared)
    return shapes


class LagunaWeights:
    """Checkpoint tensors in the engine layout, held until the engine is built."""

    def __init__(self, model_dir: str | Path, config: LagunaConfig, storage: np.dtype) -> None:
        if storage not in (BF16, np.dtype(np.float32)):
            raise ValueError(f"Laguna weight storage must be bfloat16 or float32, got {storage}")
        self.storage = np.dtype(storage)
        self._readers = _open_safetensors(Path(model_dir))
        self._shapes = expected_tensors(config)
        missing = sorted(set(self._shapes) - set(self._readers))
        unexpected = sorted(set(self._readers) - set(self._shapes))
        if missing or unexpected:
            raise ValueError(
                "Laguna checkpoint does not match its config: "
                f"missing={missing[:8]} ({len(missing)}), "
                f"unexpected={unexpected[:8]} ({len(unexpected)})"
            )
        self.tensors: dict[str, np.ndarray] = {}
        self._load(config)
        self._readers = {}
        gc.collect()

    def _read(self, name: str) -> np.ndarray:
        value = self._readers[name].get_tensor(name)
        if tuple(value.shape) != self._shapes[name]:
            raise ValueError(
                f"{name} has shape {tuple(value.shape)}, expected {self._shapes[name]}"
            )
        return value

    def _take(self, name: str, dtype: np.dtype | None = None) -> np.ndarray:
        return np.ascontiguousarray(self._read(name), dtype=dtype or self.storage)

    def _stack_experts(self, hf: str, projection: str, count: int) -> np.ndarray:
        first = f"{hf}.mlp.experts.0.{projection}.weight"
        stacked = np.empty((count, *self._shapes[first]), dtype=self.storage)
        for expert in range(count):
            stacked[expert] = self._read(f"{hf}.mlp.experts.{expert}.{projection}.weight")
        return stacked

    def _load(self, config: LagunaConfig) -> None:
        t = self.tensors
        t["embed"] = self._take("model.embed_tokens.weight")
        t["final_norm"] = self._take("model.norm.weight")
        t["lm_head"] = t["embed"] if config.tie_word_embeddings else self._take("lm_head.weight")
        for layer in range(config.num_hidden_layers):
            hf = f"model.layers.{layer}"
            p = f"layer.{layer}"
            t[f"{p}.input_norm"] = self._take(f"{hf}.input_layernorm.weight")
            t[f"{p}.post_attn_norm"] = self._take(f"{hf}.post_attention_layernorm.weight")
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "q_norm", "k_norm"):
                t[f"{p}.{name}"] = self._take(f"{hf}.self_attn.{name}.weight")
            if config.mlp_layer_types[layer] == "dense":
                for name in ("gate_proj", "up_proj", "down_proj"):
                    t[f"{p}.mlp.{name}"] = self._take(f"{hf}.mlp.{name}.weight")
                continue
            f32 = np.dtype(np.float32)
            t[f"{p}.router"] = self._take(f"{hf}.mlp.gate.weight", f32)
            t[f"{p}.router_bias"] = self._take(f"{hf}.mlp.experts.e_score_correction_bias", f32)
            for name in ("gate_proj", "up_proj", "down_proj"):
                t[f"{p}.experts.{name}"] = self._stack_experts(hf, name, config.num_experts)
                t[f"{p}.shared.{name}"] = self._take(f"{hf}.mlp.shared_expert.{name}.weight")

    def nbytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for value in self.tensors.values():
            if id(value) not in seen:
                seen.add(id(value))
                total += value.nbytes
        return total

    def release(self) -> None:
        self.tensors.clear()
        gc.collect()
