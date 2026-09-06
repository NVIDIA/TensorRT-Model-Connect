# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and conditional-LoRA loader for pinned K2-Horizon-7B-Uno."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - selected family requirements provide it
    ml_dtypes = None

from .config import (
    ADAPTER_FILENAME,
    K2HorizonUnoConfig,
)


WeightDict = dict[str, np.ndarray]
_PROJECTIONS = (
    ("self_attn", "q_proj", "w_q"),
    ("self_attn", "k_proj", "w_k"),
    ("self_attn", "v_proj", "w_v"),
    ("self_attn", "o_proj", "w_o"),
    ("mlp", "gate_proj", "w_gate"),
    ("mlp", "up_proj", "w_up"),
    ("mlp", "down_proj", "w_down"),
)


def _layer_key(layer_index: int, suffix: str) -> str:
    return f"model.layers.{layer_index}.{suffix}"


def _base_tensor_names(num_layers: int) -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer_index in range(num_layers):
        names.add(_layer_key(layer_index, "input_layernorm.weight"))
        names.add(_layer_key(layer_index, "post_attention_layernorm.weight"))
        names.update(
            _layer_key(layer_index, f"{scope}.{projection}.weight")
            for scope, projection, _logical in _PROJECTIONS
        )
    return names


def _adapter_tensor_names(num_layers: int) -> set[str]:
    return {
        f"{_layer_key(layer_index, f'{scope}.{projection}')}.{suffix}"
        for layer_index in range(num_layers)
        for scope, projection, _logical in _PROJECTIONS
        for suffix in ("lora_A.weight", "lora_B.weight")
    }


def _open_base_readers(model_dir: Path) -> dict[str, object]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Uno staged base is missing model.safetensors.index.json: {model_dir}"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Uno base safetensors index has no weight_map")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or not name for name in shards):
        raise ValueError("Uno base safetensors index contains an invalid shard name")
    readers_by_file = {name: safe_open(str(model_dir / name), framework="numpy") for name in shards}
    return {name: readers_by_file[shard] for name, shard in weight_map.items()}


def _open_adapter_reader(model_dir: Path):
    return safe_open(str(model_dir / ADAPTER_FILENAME), framework="numpy")


def _validate_inventory(actual: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        raise ValueError(f"K2-Horizon-Uno {label} tensor inventory mismatch: " + "; ".join(details))


def _get_tensor(readers, name: str):
    if isinstance(readers, dict):
        reader = readers.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return reader.get_tensor(name)
    if name not in readers.keys():
        raise KeyError(f"Tensor not found: {name}")
    return readers.get_tensor(name)


def _to_fp32(tensor) -> np.ndarray:
    source = np.asarray(tensor)
    if source.dtype == np.uint16:
        if ml_dtypes is not None:
            return source.view(ml_dtypes.bfloat16).astype(np.float32)
        return (source.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return np.asarray(source, dtype=np.float32)


def _to_bf16_bits(tensor, *, transpose: bool = False) -> np.ndarray:
    source = np.asarray(tensor)
    if transpose:
        if source.ndim != 2:
            raise ValueError("Uno projection tensors must be rank two")
        source = source.T
    if source.dtype == np.uint16:
        return np.array(source, dtype=np.uint16, order="C", copy=True)
    if ml_dtypes is None:
        source_fp32 = np.ascontiguousarray(source, dtype=np.float32)
        source_bits = source_fp32.view(np.uint32)
        rounding_bias = np.uint32(0x7FFF) + ((source_bits >> np.uint32(16)) & np.uint32(1))
        rounded = source_bits + rounding_bias
        return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))
    bf16 = np.asarray(source, dtype=ml_dtypes.bfloat16)
    return np.array(bf16.view(np.uint16), dtype=np.uint16, order="C", copy=True)


def _load_bf16(readers, name: str, *, transpose: bool = False) -> np.ndarray:
    return _to_bf16_bits(_get_tensor(readers, name), transpose=transpose)


def _load_norm(readers, name: str) -> np.ndarray:
    return np.array(_to_fp32(_get_tensor(readers, name)), dtype=np.float32, copy=True)


def _expected_projection_shapes(cfg: K2HorizonUnoConfig) -> dict[str, tuple[int, int]]:
    return {
        "q_proj": (cfg.hidden_size, cfg.attention_size),
        "k_proj": (cfg.hidden_size, cfg.kv_attention_size),
        "v_proj": (cfg.hidden_size, cfg.kv_attention_size),
        "o_proj": (cfg.attention_size, cfg.hidden_size),
        "gate_proj": (cfg.hidden_size, cfg.intermediate_size),
        "up_proj": (cfg.hidden_size, cfg.intermediate_size),
        "down_proj": (cfg.intermediate_size, cfg.hidden_size),
    }


def load_standard_weights(
    base_dir: str | Path,
    adapter_dir: str | Path,
    cfg: K2HorizonUnoConfig,
) -> WeightDict:
    """Load base BF16 weights plus the exact per-projection Uno LoRA tensors."""

    base_path = Path(base_dir)
    adapter_path = Path(adapter_dir)
    base = _open_base_readers(base_path)
    adapter = _open_adapter_reader(adapter_path)
    _validate_inventory(set(base), _base_tensor_names(cfg.num_hidden_layers), "base")
    _validate_inventory(
        set(adapter.keys()), _adapter_tensor_names(cfg.num_hidden_layers), "adapter"
    )

    weights = WeightDict()
    weights["embedding"] = _load_bf16(base, "model.embed_tokens.weight")
    if weights["embedding"].shape != (cfg.vocab_size, cfg.hidden_size):
        raise ValueError("K2-Horizon-Uno embedding shape does not match config")

    projection_shapes = _expected_projection_shapes(cfg)

    def load_layer(layer_index: int) -> WeightDict:
        layer = WeightDict()
        logical_prefix = f"layer.{layer_index}"
        for logical, source in (
            ("input_norm", "input_layernorm.weight"),
            ("post_attn_norm", "post_attention_layernorm.weight"),
        ):
            value = _load_norm(base, _layer_key(layer_index, source))
            if value.shape != (cfg.hidden_size,):
                raise ValueError(f"K2-Horizon-Uno layer {layer_index} {logical} shape mismatch")
            layer[f"{logical_prefix}.{logical}"] = value

        for scope, projection, logical in _PROJECTIONS:
            source_prefix = _layer_key(layer_index, f"{scope}.{projection}")
            expected = projection_shapes[projection]
            base_weight = _load_bf16(base, f"{source_prefix}.weight", transpose=True)
            lora_a = _load_bf16(adapter, f"{source_prefix}.lora_A.weight", transpose=True)
            lora_b = _load_bf16(adapter, f"{source_prefix}.lora_B.weight", transpose=True)
            if base_weight.shape != expected:
                raise ValueError(
                    f"K2-Horizon-Uno {source_prefix} base shape must be {expected}, "
                    f"got {base_weight.shape}"
                )
            if lora_a.shape != (expected[0], cfg.lora_rank):
                raise ValueError(
                    f"K2-Horizon-Uno {source_prefix} LoRA A shape mismatch: {lora_a.shape}"
                )
            if lora_b.shape != (cfg.lora_rank, expected[1]):
                raise ValueError(
                    f"K2-Horizon-Uno {source_prefix} LoRA B shape mismatch: {lora_b.shape}"
                )
            layer[f"{logical_prefix}.{logical}"] = base_weight
            layer[f"{logical_prefix}.{logical}_lora_a"] = lora_a
            layer[f"{logical_prefix}.{logical}_lora_b"] = lora_b
        return layer

    for layer_index in range(cfg.num_hidden_layers):
        weights.update(load_layer(layer_index))

    weights["final_norm"] = _load_norm(base, "model.norm.weight")
    weights["w_out"] = _load_bf16(base, "lm_head.weight", transpose=True)
    if weights["final_norm"].shape != (cfg.hidden_size,):
        raise ValueError("K2-Horizon-Uno final norm shape does not match config")
    if weights["w_out"].shape != (cfg.hidden_size, cfg.vocab_size):
        raise ValueError("K2-Horizon-Uno LM head shape does not match config")
    return weights
