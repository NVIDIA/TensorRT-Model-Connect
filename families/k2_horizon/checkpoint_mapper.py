# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the exact dense K2-Horizon checkpoint contract from safetensors."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from safetensors import safe_open

try:
    import ml_dtypes
except ImportError:  # pragma: no cover - selected family requirements provide it
    ml_dtypes = None

from .config import K2HorizonConfig, validate_config


def _target_np_dtype(precision: str) -> np.dtype:
    """Map precision string to numpy dtype for weight storage."""
    if precision == "bf16":
        return np.dtype(np.uint16)
    if precision == "fp16":
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def _layer_key(layer_idx: int, suffix: str, model_prefix: str = "model") -> str:
    return f"{model_prefix}.layers.{layer_idx}.{suffix}"


def _copy_to_numpy(tensor, dtype: np.dtype, *, transpose_name: str | None = None) -> np.ndarray:
    """Copy a checkpoint tensor directly into an owned contiguous NumPy array."""
    if transpose_name is not None and tensor.ndim != 2:
        raise ValueError(f"Expected rank-2 tensor for transpose: {transpose_name}")

    source = np.asarray(tensor)
    if dtype == np.dtype(np.uint16):
        if transpose_name is not None:
            source = source.T
        if source.dtype == np.uint16:
            return np.array(source, dtype=np.uint16, order="C", copy=True)
        if ml_dtypes is None:
            raise RuntimeError("ml_dtypes is required for K2-Horizon BF16 weights")
        bf16 = np.asarray(source, dtype=ml_dtypes.bfloat16)
        return np.array(bf16.view(np.uint16), dtype=np.uint16, order="C", copy=True)
    if source.dtype == np.uint16:
        if ml_dtypes is None:
            raise RuntimeError("ml_dtypes is required for K2-Horizon BF16 weights")
        source = source.view(ml_dtypes.bfloat16)
    if transpose_name is not None:
        source = source.T
    return np.array(source, dtype=dtype, order="C", copy=True)


class WeightDict(dict):
    """Logical K2-Horizon weights consumed by ``model.py``."""


def _expected_tensor_names(num_layers: int) -> set[str]:
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer_idx in range(num_layers):
        names.update(
            {
                _layer_key(layer_idx, "input_layernorm.weight"),
                _layer_key(layer_idx, "post_attention_layernorm.weight"),
                _layer_key(layer_idx, "self_attn.q_proj.weight"),
                _layer_key(layer_idx, "self_attn.k_proj.weight"),
                _layer_key(layer_idx, "self_attn.v_proj.weight"),
                _layer_key(layer_idx, "self_attn.o_proj.weight"),
                _layer_key(layer_idx, "mlp.gate_proj.weight"),
                _layer_key(layer_idx, "mlp.up_proj.weight"),
                _layer_key(layer_idx, "mlp.down_proj.weight"),
            }
        )
    return names


def _validate_checkpoint_tensor_names(readers: list, num_layers: int) -> None:
    actual = set(getattr(readers, "tensor_map", {}))
    if not actual:
        actual = {name for reader in readers for name in reader.keys()}
    expected = _expected_tensor_names(num_layers)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected[:8]))
        raise ValueError(
            "K2-Horizon checkpoint tensors do not match the qualified dense graph: "
            + "; ".join(details)
        )


def load_standard_weights(
    model_dir: str | Path,
    config: object,
    *,
    precision: str = "bf16",
) -> WeightDict:
    """Map the publisher checkpoint directly into compact BF16 build storage."""
    if precision != "bf16":
        raise ValueError("K2-Horizon checkpoint loading supports only BF16")
    cfg: K2HorizonConfig = validate_config(config)
    model_dir = Path(model_dir)
    readers = _open_safetensors(model_dir)

    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    num_layers = cfg.num_hidden_layers
    _validate_checkpoint_tensor_names(readers, num_layers)
    attention_size = cfg.attention_size
    kv_attention_size = cfg.kv_attention_size
    mlp_size = cfg.intermediate_size
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()

    # Embedding
    embedding = _load_tensor_as_dtype(readers, "model.embed_tokens.weight", target_dtype)
    if embedding.shape != (vocab, hidden):
        raise ValueError(f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
    weights["embedding"] = embedding

    def _load_layer(layer_idx: int) -> tuple[int, WeightDict]:
        prefix = f"layer.{layer_idx}"
        layer = WeightDict()

        # Norms
        input_norm = _load_tensor(readers, _layer_key(layer_idx, "input_layernorm.weight"))
        post_norm = _load_tensor(readers, _layer_key(layer_idx, "post_attention_layernorm.weight"))
        if input_norm.shape != (hidden,) or post_norm.shape != (hidden,):
            raise ValueError(
                f"K2-Horizon layer {layer_idx} norm weights must have shape ({hidden},)"
            )
        layer[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        layer[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        # Transpose [out, in] -> [in, out] while copying directly to the
        # final storage dtype. In particular, FP16/BF16 builds must not stage
        # these model-sized tensors through FP32 first.
        q_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.q_proj.weight"), "q_proj", target_dtype
        )
        k_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.k_proj.weight"), "k_proj", target_dtype
        )
        v_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.v_proj.weight"), "v_proj", target_dtype
        )
        o_t = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "self_attn.o_proj.weight"), "o_proj", target_dtype
        )

        expected_attention_shapes = {
            "q_proj": (hidden, attention_size),
            "k_proj": (hidden, kv_attention_size),
            "v_proj": (hidden, kv_attention_size),
            "o_proj": (attention_size, hidden),
        }
        for name, tensor in (
            ("q_proj", q_t),
            ("k_proj", k_t),
            ("v_proj", v_t),
            ("o_proj", o_t),
        ):
            if tensor.shape != expected_attention_shapes[name]:
                raise ValueError(
                    f"K2-Horizon layer {layer_idx} {name} must have shape "
                    f"{expected_attention_shapes[name]}, got {tensor.shape}"
                )

        layer[f"{prefix}.w_q"] = q_t
        layer[f"{prefix}.w_k"] = k_t
        layer[f"{prefix}.w_v"] = v_t
        layer[f"{prefix}.w_o"] = o_t

        # MLP projections
        layer[f"{prefix}.w_gate"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.gate_proj.weight"), "gate_proj", target_dtype
        )
        layer[f"{prefix}.w_up"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.up_proj.weight"), "up_proj", target_dtype
        )
        layer[f"{prefix}.w_down"] = _load_transposed_tensor(
            readers, _layer_key(layer_idx, "mlp.down_proj.weight"), "down_proj", target_dtype
        )
        expected_mlp_shapes = {
            f"{prefix}.w_gate": (hidden, mlp_size),
            f"{prefix}.w_up": (hidden, mlp_size),
            f"{prefix}.w_down": (mlp_size, hidden),
        }
        for name, expected in expected_mlp_shapes.items():
            if layer[name].shape != expected:
                raise ValueError(
                    f"K2-Horizon {name} must have shape {expected}, got {layer[name].shape}"
                )

        return layer_idx, layer

    layer_results: list[tuple[int, WeightDict] | None] = [None] * num_layers
    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
        for future in as_completed(futures):
            layer_idx, layer = future.result()
            layer_results[layer_idx] = (layer_idx, layer)

    for result in layer_results:
        if result is None:
            continue
        _layer_idx, layer = result
        weights.update(layer)

    # Final norm
    weights["final_norm"] = _load_tensor(readers, "model.norm.weight").astype(np.float32)
    if weights["final_norm"].shape != (hidden,):
        raise ValueError(
            f"K2-Horizon final norm must have shape ({hidden},), got {weights['final_norm'].shape}"
        )

    # LM head
    weights["w_out"] = _load_transposed_tensor(readers, "lm_head.weight", "lm_head", target_dtype)
    if weights["w_out"].shape != (hidden, vocab):
        raise ValueError(
            f"K2-Horizon lm_head must have shape ({hidden}, {vocab}), got {weights['w_out'].shape}"
        )

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
    weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

    return weights


# ---------------------------------------------------------------------------
# Safetensors I/O helpers
# ---------------------------------------------------------------------------


class _ReaderCollection(list):
    """Reader list with a cached tensor-name -> reader lookup table."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {}
            for reader in readers:
                for key in reader.keys():
                    tensor_map[key] = reader
        self.tensor_map = tensor_map


def _open_safetensors(model_dir: Path) -> _ReaderCollection:
    """Open the checkpoint's required NumPy safetensors readers."""

    single = model_dir / "model.safetensors"
    if single.is_file():
        return _ReaderCollection([safe_open(str(single), framework="numpy")])

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        import json

        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("K2-Horizon safetensors index has no weight_map")
        shard_files = sorted(set(str(value) for value in weight_map.values()))
        readers_by_file = {
            shard: safe_open(str(model_dir / shard), framework="numpy") for shard in shard_files
        }
        return _ReaderCollection(
            [readers_by_file[shard] for shard in shard_files],
            tensor_map={
                str(name): readers_by_file[str(shard)] for name, shard in weight_map.items()
            },
        )

    raise FileNotFoundError(f"No K2-Horizon safetensors checkpoint in {model_dir}")


def _to_numpy_fp32(tensor) -> np.ndarray:
    """Convert a NumPy or ml_dtypes checkpoint tensor to float32."""

    return np.asarray(tensor, dtype=np.float32)


def _get_tensor(readers: list, name: str):
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        reader = tensor_map.get(name)
        if reader is None:
            raise KeyError(f"Tensor not found: {name}")
        return reader.get_tensor(name)
    for reader in readers:
        if name in reader.keys():
            return reader.get_tensor(name)
    raise KeyError(f"Tensor not found: {name}")


def _load_tensor_as_dtype(readers: list, name: str, dtype: np.dtype) -> np.ndarray:
    return _copy_to_numpy(_get_tensor(readers, name), dtype)


def _load_transposed_tensor(
    readers: list,
    name: str,
    transpose_name: str,
    dtype: np.dtype,
) -> np.ndarray:
    return _copy_to_numpy(_get_tensor(readers, name), dtype, transpose_name=transpose_name)


def _load_tensor(readers: list, name: str) -> np.ndarray:
    return _to_numpy_fp32(_get_tensor(readers, name))
