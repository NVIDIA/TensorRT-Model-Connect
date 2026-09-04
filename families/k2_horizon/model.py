# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned TensorRT graph and bundle build for dense K2-Horizon."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .native_kv_attention_builder import (
    NativeKvMasks,
    add_active_prefix_causal_masks,
    add_explicit_masked_grouped_query_attention,
)

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import K2HorizonConfig, validate_config


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def _cast(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    keepalive: list[np.ndarray],
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> trt.ITensor:
    array = np.ascontiguousarray(values, dtype=dtype).reshape(shape)
    keepalive.append(array)
    layer = network.add_constant(shape, trt.Weights(array))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a K2-Horizon constant")
    return layer.get_output(0)


def _work_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    work_dtype: trt.DataType,
    constant_keepalive: list[np.ndarray],
) -> trt.ITensor:
    if work_dtype != trt.bfloat16:
        raise ValueError("K2-Horizon weight constants require BF16 TensorRT dtype")
    array = np.asarray(values)
    if array.dtype != np.uint16 or not array.flags.c_contiguous:
        raise ValueError("K2-Horizon BF16 weights must be contiguous uint16 bit patterns")
    if tuple(array.shape) != shape:
        raise ValueError(
            f"K2-Horizon BF16 constant must have shape {shape}, got {tuple(array.shape)}"
        )
    constant_keepalive.append(array)
    # TensorRT retains this pointer until serialization; the build-scoped
    # keepalive owns the exact array passed to this constructor.
    weights = trt.Weights(trt.bfloat16, int(array.ctypes.data), int(array.size))
    layer = network.add_constant(shape, weights)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a K2-Horizon BF16 constant")
    return layer.get_output(0)


def _matmul(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    work_dtype: trt.DataType,
    constant_keepalive: list[np.ndarray],
    name: str,
) -> trt.ITensor:
    expected = (lhs_width, rhs_width)
    if tuple(np.asarray(rhs).shape) != expected:
        raise ValueError(
            f"mapped K2-Horizon weight {name} must have shape {expected}, "
            f"got {tuple(np.asarray(rhs).shape)}"
        )
    rhs_tensor = _work_constant(
        network,
        expected,
        rhs,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )
    layer = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs_tensor,
        trt.MatrixOperation.NONE,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add K2-Horizon matmul {name}")
    layer.name = name
    return layer.get_output(0)


def _grouped_rms_norm(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    gamma: np.ndarray,
    *,
    hidden_size: int,
    num_groups: int,
    epsilon: float,
    work_dtype: trt.DataType,
    constant_keepalive: list[np.ndarray],
) -> trt.ITensor:
    if tuple(np.asarray(gamma).shape) != (hidden_size,):
        raise ValueError(
            f"mapped K2-Horizon grouped RMSNorm weight must have shape ({hidden_size},)"
        )
    group_width = hidden_size // num_groups
    fp32 = _cast(network, tensor, trt.float32)
    shaped = network.add_shuffle(fp32)
    shaped.reshape_dims = (1, num_groups, group_width)
    grouped = shaped.get_output(0)
    squared = network.add_elementwise(
        grouped,
        grouped,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mean = network.add_reduce(
        squared,
        trt.ReduceOperation.AVG,
        1 << 2,
        True,
    ).get_output(0)
    eps = _constant(
        network,
        (1, 1, 1),
        np.array([epsilon], dtype=np.float32),
        constant_keepalive,
    )
    variance = network.add_elementwise(
        mean,
        eps,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        grouped,
        reciprocal,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    flattened = network.add_shuffle(normalized)
    flattened.reshape_dims = (1, hidden_size)
    gamma_tensor = _constant(
        network,
        (1, hidden_size),
        gamma,
        constant_keepalive,
        dtype=np.dtype(np.float32),
    )
    scaled = network.add_elementwise(
        flattened.get_output(0),
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return _cast(network, scaled, work_dtype)


def _silu(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
) -> trt.ITensor:
    sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        tensor,
        sigmoid.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)


def _active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    *,
    head_dim: int,
    rope_theta: float,
    work_dtype: trt.DataType,
    constant_keepalive: list[np.ndarray],
) -> tuple[trt.ITensor, trt.ITensor]:
    inverse_frequency = 1.0 / (
        float(rope_theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    position = _cast(network, position_id, trt.float32)
    position_column = network.add_shuffle(position)
    position_column.reshape_dims = (1, 1)
    inverse = _constant(
        network,
        (1, head_dim // 2),
        inverse_frequency.reshape(1, -1),
        constant_keepalive,
    )
    angles = network.add_elementwise(
        position_column.get_output(0),
        inverse,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_3d = network.add_shuffle(cos)
    cos_3d.reshape_dims = (1, 1, head_dim // 2)
    sin_3d = network.add_shuffle(sin)
    sin_3d.reshape_dims = (1, 1, head_dim // 2)
    return (
        _cast(network, cos_3d.get_output(0), work_dtype),
        _cast(network, sin_3d.get_output(0), work_dtype),
    )


def _reshape_rows_to_heads(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (1, num_heads, 1, head_dim)
    return shaped.get_output(0)


def _reshape_heads_to_rows(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    *,
    width: int,
) -> trt.ITensor:
    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (1, width)
    return shaped.get_output(0)


def _apply_rope(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cos_cache: trt.ITensor,
    sin_cache: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
) -> trt.ITensor:
    heads = _reshape_rows_to_heads(
        network,
        tensor,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    rope = network.add_rotary_embedding(
        heads,
        cos_cache,
        sin_cache,
        False,
        head_dim,
    )
    if rope is None:
        raise RuntimeError("TensorRT failed to create K2-Horizon rotary embedding")
    return _reshape_heads_to_rows(
        network,
        rope.get_output(0),
        width=num_heads * head_dim,
    )


def _native_attention(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    attention_masks: NativeKvMasks,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tag: str,
) -> dict[str, trt.ITensor]:
    if not hasattr(network, "add_kv_cache_update"):
        raise RuntimeError("K2-Horizon requires TensorRT add_kv_cache_update support")
    k_4d = _reshape_rows_to_heads(network, k, num_heads=num_kv_heads, head_dim=head_dim)
    v_4d = _reshape_rows_to_heads(network, v, num_heads=num_kv_heads, head_dim=head_dim)
    update_k = network.add_kv_cache_update(
        cache_k,
        k_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    update_v = network.add_kv_cache_update(
        cache_v,
        v_4d,
        cache_write_indices,
        trt.KVCacheMode.LINEAR,
    )
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT failed to create K2-Horizon KV-cache update layers")
    update_k.name = f"{tag}.cache_k_update"
    update_v.name = f"{tag}.cache_v_update"
    present_k = update_k.get_output(0)
    present_v = update_v.get_output(0)
    q_4d = _reshape_rows_to_heads(network, q, num_heads=num_heads, head_dim=head_dim)
    context_4d = add_explicit_masked_grouped_query_attention(
        network,
        q_4d,
        present_k,
        present_v,
        attention_masks,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=float(1.0 / np.sqrt(head_dim)),
        tag=tag,
    )
    return {
        "context": _reshape_heads_to_rows(
            network,
            context_4d,
            width=num_heads * head_dim,
        ),
        "present_k": present_k,
        "present_v": present_v,
    }


def _validate_mapped_weights(config: K2HorizonConfig, weights: WeightDict) -> None:
    metadata = {
        "_attention_size": config.attention_size,
        "_kv_attention_size": config.kv_attention_size,
        "_mlp_size": config.intermediate_size,
    }
    for name, expected in metadata.items():
        if weights.get(name) != expected:
            raise ValueError(
                f"mapped K2-Horizon metadata {name} must be {expected}, got {weights.get(name)!r}"
            )
    forbidden_suffixes = (
        ".q_bias",
        ".k_bias",
        ".v_bias",
        ".o_bias",
        ".q_norm",
        ".k_norm",
    )
    unexpected = sorted(
        name for name in weights if isinstance(name, str) and name.endswith(forbidden_suffixes)
    )
    if unexpected:
        raise ValueError(
            "K2-Horizon mapped weights contain unsupported attention parameters: "
            + ", ".join(unexpected[:8])
        )


def build_engine(
    config: object,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build the qualified single-token BF16 engine with native KV updates."""

    cfg = validate_config(config)
    if precision != "bf16":
        raise ValueError("K2-Horizon currently supports only BF16 builds")
    if isinstance(max_cache_length, bool) or not isinstance(max_cache_length, int):
        raise ValueError("K2-Horizon max_cache_length must be an integer")
    if max_cache_length <= 0 or max_cache_length > cfg.max_position_embeddings:
        raise ValueError(
            f"K2-Horizon max_cache_length must be in [1, {cfg.max_position_embeddings}]"
        )
    _validate_mapped_weights(cfg, weights)
    work_dtype = trt.bfloat16
    # TensorRT weights reference their NumPy storage through serialization.
    constant_keepalive: list[np.ndarray] = []

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    cache_write_indices = network.add_input("cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    attention_masks = add_active_prefix_causal_masks(
        network,
        token_id,
        cache_write_indices,
        key_value_lengths,
        max_cache_length,
    )

    cache_shape = (
        1,
        cfg.num_key_value_heads,
        max_cache_length,
        cfg.head_dim,
    )
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for layer_index in range(cfg.num_hidden_layers):
        cache_k_inputs.append(
            network.add_input(
                layer_tensor_name("cache_k", layer_index),
                work_dtype,
                cache_shape,
            )
        )
        cache_v_inputs.append(
            network.add_input(
                layer_tensor_name("cache_v", layer_index),
                work_dtype,
                cache_shape,
            )
        )

    embedding = _work_constant(
        network,
        (cfg.vocab_size, cfg.hidden_size),
        weights["embedding"],
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )
    hidden_state = network.add_gather(embedding, token_id, 0).get_output(0)
    hidden_state = _cast(network, hidden_state, work_dtype)
    cos_cache, sin_cache = _active_rope_cache(
        network,
        position_id,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )

    present_k_outputs: list[trt.ITensor] = []
    present_v_outputs: list[trt.ITensor] = []
    for layer_index in range(cfg.num_hidden_layers):
        prefix = f"layer.{layer_index}"
        normed = _grouped_rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.input_norm"],
            hidden_size=cfg.hidden_size,
            num_groups=cfg.layernorm_num_groups,
            epsilon=cfg.rms_norm_eps,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
        )
        q = _matmul(
            network,
            normed,
            weights[f"{prefix}.w_q"],
            lhs_width=cfg.hidden_size,
            rhs_width=cfg.attention_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.self_attn.q_proj",
        )
        k = _matmul(
            network,
            normed,
            weights[f"{prefix}.w_k"],
            lhs_width=cfg.hidden_size,
            rhs_width=cfg.kv_attention_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.self_attn.k_proj",
        )
        v = _matmul(
            network,
            normed,
            weights[f"{prefix}.w_v"],
            lhs_width=cfg.hidden_size,
            rhs_width=cfg.kv_attention_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.self_attn.v_proj",
        )
        q = _apply_rope(
            network,
            q,
            cos_cache,
            sin_cache,
            num_heads=cfg.num_attention_heads,
            head_dim=cfg.head_dim,
        )
        k = _apply_rope(
            network,
            k,
            cos_cache,
            sin_cache,
            num_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
        )
        attention = _native_attention(
            network,
            q,
            k,
            v,
            cache_k_inputs[layer_index],
            cache_v_inputs[layer_index],
            cache_write_indices,
            attention_masks,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            tag=f"{prefix}.self_attn",
        )
        attn_out = _matmul(
            network,
            attention["context"],
            weights[f"{prefix}.w_o"],
            lhs_width=cfg.attention_size,
            rhs_width=cfg.hidden_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.self_attn.o_proj",
        )
        hidden_state = network.add_elementwise(
            hidden_state,
            attn_out,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        mlp_input = _grouped_rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.post_attn_norm"],
            hidden_size=cfg.hidden_size,
            num_groups=cfg.layernorm_num_groups,
            epsilon=cfg.rms_norm_eps,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
        )
        gate = _matmul(
            network,
            mlp_input,
            weights[f"{prefix}.w_gate"],
            lhs_width=cfg.hidden_size,
            rhs_width=cfg.intermediate_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.mlp.gate_proj",
        )
        up = _matmul(
            network,
            mlp_input,
            weights[f"{prefix}.w_up"],
            lhs_width=cfg.hidden_size,
            rhs_width=cfg.intermediate_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.mlp.up_proj",
        )
        gated = network.add_elementwise(
            _silu(network, gate),
            up,
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        mlp_out = _matmul(
            network,
            gated,
            weights[f"{prefix}.w_down"],
            lhs_width=cfg.intermediate_size,
            rhs_width=cfg.hidden_size,
            work_dtype=work_dtype,
            constant_keepalive=constant_keepalive,
            name=f"{prefix}.mlp.down_proj",
        )
        hidden_state = network.add_elementwise(
            hidden_state,
            mlp_out,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        present_k_outputs.append(attention["present_k"])
        present_v_outputs.append(attention["present_v"])

    hidden_state = _grouped_rms_norm(
        network,
        hidden_state,
        weights["final_norm"],
        hidden_size=cfg.hidden_size,
        num_groups=cfg.layernorm_num_groups,
        epsilon=cfg.rms_norm_eps,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )
    logits = _matmul(
        network,
        hidden_state,
        weights["w_out"],
        lhs_width=cfg.hidden_size,
        rhs_width=cfg.vocab_size,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
        name="lm_head",
    )
    logits = _cast(network, logits, trt.float32)
    logits.name = "logits"
    network.mark_output(logits)

    for layer_index, tensor in enumerate(present_k_outputs):
        tensor.name = layer_tensor_name("present_k", layer_index)
        network.mark_output(tensor)
    for layer_index, tensor in enumerate(present_v_outputs):
        tensor.name = layer_tensor_name("present_v", layer_index)
        network.mark_output(tensor)

    if verbose:
        print(
            "[trtmc build] Building K2-Horizon native BF16 engine "
            f"(layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, "
            f"cache={max_cache_length}) ...",
            file=sys.stderr,
        )
    try:
        plan = builder.build_serialized_network(network, builder_config)
    finally:
        # Every trt.Weights consumer has finished once serialization returns
        # (or raises), so the retained host buffers can now be released.
        constant_keepalive.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build the K2-Horizon engine")
    return bytes(plan)


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _load_config(model_dir: Path) -> SimpleNamespace:
    path = model_dir / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"K2-Horizon checkpoint is missing {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("K2-Horizon config.json must contain one JSON object")
    return SimpleNamespace(raw=raw, **raw)


def _token_id(value: object, name: str, vocab_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"K2-Horizon {name} must be an integer")
    if value < 0 or value >= vocab_size:
        raise ValueError(f"K2-Horizon {name} is outside the vocabulary")
    return value


def _eos_token_ids(value: object, vocab_size: int) -> int | list[int]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("K2-Horizon eos_token_id must not be empty")
    resolved = [_token_id(item, "eos_token_id", vocab_size) for item in values]
    if len(resolved) != len(set(resolved)):
        raise ValueError("K2-Horizon eos_token_id must not contain duplicates")
    return resolved[0] if len(resolved) == 1 else resolved


def _runtime_config(
    model_dir: Path,
    raw: dict,
    config: K2HorizonConfig,
    max_cache_length: int,
) -> dict:
    eos = raw.get("eos_token_id")
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("K2-Horizon generation_config.json must be an object")
        eos = generation.get("eos_token_id", eos)
    architectures = raw.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) else ""
    return {
        "model_type": str(raw.get("model_type") or ""),
        "architecture": str(architecture),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "layernorm_num_groups": config.layernorm_num_groups,
        "bos_token_id": _token_id(raw.get("bos_token_id"), "bos_token_id", config.vocab_size),
        "eos_token_id": _eos_token_ids(eos, config.vocab_size),
        "pad_token_id": (
            -1
            if raw.get("pad_token_id") is None
            else _token_id(raw["pad_token_id"], "pad_token_id", config.vocab_size)
        ),
        "precision": "bf16",
        "max_cache_length": max_cache_length,
        "decoder_engine_layout": "single",
        "tensor_parallel_size": 1,
        "tensor_parallel_mode": "single",
        "native_kv_cache": True,
        "native_kv_contract_version": 1,
        "tie_word_embeddings": False,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build the qualified K2-Horizon BF16 bundle."""

    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("K2-Horizon does not support image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("K2-Horizon does not support video inputs")
    if request.max_batch_size != 1:
        raise NotImplementedError("K2-Horizon supports only max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("K2-Horizon does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("K2-Horizon does not support context parallelism")
    if request.task != "text_generation":
        raise ValueError("K2-Horizon supports only task=text_generation")
    if str(request.precision).lower() != "bf16":
        raise ValueError("K2-Horizon currently supports only BF16 builds")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("K2-Horizon does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("K2-Horizon does not support mixed-FP32 layers")

    model_dir = Path(request.model_dir)
    source_config = _load_config(model_dir)
    config = validate_config(source_config)
    max_cache_length = request.max_sequence_length or min(
        256,
        config.max_position_embeddings,
    )
    if max_cache_length > config.max_position_embeddings:
        raise ValueError("K2-Horizon max_sequence_length exceeds checkpoint capacity")
    weights = load_standard_weights(model_dir, source_config, precision="bf16")
    plan = build_engine(
        source_config,
        weights,
        max_cache_length,
        precision="bf16",
        verbose=bool(request.verbose),
    )

    writer.set_header(family="k2_horizon", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        _runtime_config(model_dir, source_config.raw, config, max_cache_length),
    )
    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError("K2-Horizon checkpoint is missing tokenizer.json")
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
