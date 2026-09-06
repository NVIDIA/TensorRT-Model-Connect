# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-block TensorRT graph for conditional-LoRA K2-Horizon Uno."""

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
from .config import (
    ADAPTER_FILENAME,
    BASE_MODEL_ID,
    BASE_REVISION,
    K2HorizonUnoConfig,
    load_and_validate_adapter_config,
    validate_config,
)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def _cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    layer = network.add_cast(tensor, dtype)
    if layer is None:
        raise RuntimeError("TensorRT failed to add a K2-Horizon-Uno cast")
    return layer.get_output(0)


def _constant(
    network,
    shape: tuple[int, ...],
    values: np.ndarray,
    keepalive: list[np.ndarray],
    *,
    dtype: np.dtype = np.dtype(np.float32),
):
    array = np.ascontiguousarray(values, dtype=dtype).reshape(shape)
    keepalive.append(array)
    layer = network.add_constant(shape, trt.Weights(array))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a K2-Horizon-Uno constant")
    return layer.get_output(0)


def _work_constant(
    network,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    work_dtype,
    constant_keepalive: list[np.ndarray],
):
    if work_dtype != trt.bfloat16:
        raise ValueError("K2-Horizon-Uno weight constants require BF16")
    array = np.asarray(values)
    if array.dtype != np.uint16 or not array.flags.c_contiguous:
        raise ValueError("K2-Horizon-Uno BF16 weights must be contiguous uint16 bit patterns")
    if tuple(array.shape) != shape:
        raise ValueError(f"K2-Horizon-Uno BF16 constant must have shape {shape}, got {array.shape}")
    constant_keepalive.append(array)
    weights = trt.Weights(trt.bfloat16, int(array.ctypes.data), int(array.size))
    layer = network.add_constant(shape, weights)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a K2-Horizon-Uno BF16 constant")
    return layer.get_output(0)


def _matmul(
    network,
    lhs,
    rhs: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    work_dtype,
    constant_keepalive: list[np.ndarray],
    name: str,
):
    expected = (lhs_width, rhs_width)
    if tuple(np.asarray(rhs).shape) != expected:
        raise ValueError(
            f"mapped Uno weight {name} must have shape {expected}, got {np.asarray(rhs).shape}"
        )
    rhs_tensor = _work_constant(
        network,
        expected,
        rhs,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )
    layer = network.add_matrix_multiply(
        lhs, trt.MatrixOperation.NONE, rhs_tensor, trt.MatrixOperation.NONE
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add K2-Horizon-Uno matmul {name}")
    layer.name = name
    return layer.get_output(0)


def _conditional_projection(
    network,
    lhs,
    base_weight: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    lora_mask,
    *,
    lhs_width: int,
    rhs_width: int,
    rank: int,
    scale: float,
    work_dtype,
    constant_keepalive: list[np.ndarray],
    name: str,
):
    """Compute ``xW + mask * scale * (xA)B`` for every active token row."""

    base = _matmul(
        network,
        lhs,
        base_weight,
        lhs_width=lhs_width,
        rhs_width=rhs_width,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
        name=name + ".base",
    )
    low_rank = _matmul(
        network,
        lhs,
        lora_a,
        lhs_width=lhs_width,
        rhs_width=rank,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
        name=name + ".lora_A",
    )
    delta = _matmul(
        network,
        low_rank,
        lora_b,
        lhs_width=rank,
        rhs_width=rhs_width,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
        name=name + ".lora_B",
    )
    scale_tensor = _constant(
        network,
        (1, 1),
        np.array([[scale]], dtype=np.float32),
        constant_keepalive,
    )
    scale_tensor = _cast(network, scale_tensor, work_dtype)
    scaled = network.add_elementwise(delta, scale_tensor, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError(f"TensorRT failed to scale conditional LoRA for {name}")
    mask_rows = network.add_shuffle(lora_mask)
    mask_rows.reshape_dims = (-1, 1)
    mask_work = _cast(network, mask_rows.get_output(0), work_dtype)
    selected = network.add_elementwise(
        scaled.get_output(0), mask_work, trt.ElementWiseOperation.PROD
    )
    if selected is None:
        raise RuntimeError(f"TensorRT failed to mask conditional LoRA for {name}")
    result = network.add_elementwise(base, selected.get_output(0), trt.ElementWiseOperation.SUM)
    if result is None:
        raise RuntimeError(f"TensorRT failed to apply conditional LoRA for {name}")
    result.name = name
    return result.get_output(0)


def _grouped_rms_norm(
    network,
    tensor,
    gamma: np.ndarray,
    *,
    hidden_size: int,
    num_groups: int,
    epsilon: float,
    work_dtype,
    constant_keepalive: list[np.ndarray],
):
    if tuple(np.asarray(gamma).shape) != (hidden_size,):
        raise ValueError("mapped Uno grouped RMSNorm weight has the wrong shape")
    group_width = hidden_size // num_groups
    fp32 = _cast(network, tensor, trt.float32)
    shaped = network.add_shuffle(fp32)
    shaped.reshape_dims = (-1, num_groups, group_width)
    grouped = shaped.get_output(0)
    squared = network.add_elementwise(grouped, grouped, trt.ElementWiseOperation.PROD).get_output(0)
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
    eps = _constant(
        network,
        (1, 1, 1),
        np.array([epsilon], dtype=np.float32),
        constant_keepalive,
    )
    variance = network.add_elementwise(mean, eps, trt.ElementWiseOperation.SUM).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        grouped, reciprocal, trt.ElementWiseOperation.PROD
    ).get_output(0)
    flattened = network.add_shuffle(normalized)
    flattened.reshape_dims = (-1, hidden_size)
    gamma_tensor = _constant(
        network,
        (1, hidden_size),
        gamma,
        constant_keepalive,
        dtype=np.dtype(np.float32),
    )
    scaled = network.add_elementwise(
        flattened.get_output(0), gamma_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast(network, scaled, work_dtype)


def _silu(network, tensor):
    sigmoid = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        tensor, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


def _active_rope_cache(
    network,
    position_id,
    *,
    head_dim: int,
    rope_theta: float,
    work_dtype,
    constant_keepalive: list[np.ndarray],
):
    inverse_frequency = 1.0 / (
        float(rope_theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / float(head_dim))
    )
    position = _cast(network, position_id, trt.float32)
    position_column = network.add_shuffle(position)
    position_column.reshape_dims = (-1, 1)
    inverse = _constant(
        network,
        (1, head_dim // 2),
        inverse_frequency.reshape(1, -1),
        constant_keepalive,
    )
    angles = network.add_elementwise(
        position_column.get_output(0), inverse, trt.ElementWiseOperation.PROD
    ).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos_3d = network.add_shuffle(cos)
    cos_3d.reshape_dims = (1, -1, head_dim // 2)
    sin_3d = network.add_shuffle(sin)
    sin_3d.reshape_dims = (1, -1, head_dim // 2)
    return (
        _cast(network, cos_3d.get_output(0), work_dtype),
        _cast(network, sin_3d.get_output(0), work_dtype),
    )


def _reshape_rows_to_heads(network, tensor, *, num_heads: int, head_dim: int):
    rows = network.add_shuffle(tensor)
    rows.reshape_dims = (-1, num_heads, head_dim)
    rows.second_transpose = trt.Permutation([1, 0, 2])
    shaped = network.add_shuffle(rows.get_output(0))
    shaped.reshape_dims = (1, num_heads, -1, head_dim)
    return shaped.get_output(0)


def _reshape_heads_to_rows(network, tensor, *, width: int):
    rows = network.add_shuffle(tensor)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (-1, width)
    return rows.get_output(0)


def _apply_rope(
    network,
    tensor,
    cos_cache,
    sin_cache,
    *,
    num_heads: int,
    head_dim: int,
):
    heads = _reshape_rows_to_heads(network, tensor, num_heads=num_heads, head_dim=head_dim)
    rope = network.add_rotary_embedding(heads, cos_cache, sin_cache, False, head_dim)
    if rope is None:
        raise RuntimeError("TensorRT failed to create K2-Horizon-Uno rotary embedding")
    return _reshape_heads_to_rows(network, rope.get_output(0), width=num_heads * head_dim)


def _native_attention(
    network,
    q,
    k,
    v,
    cache_k,
    cache_v,
    cache_write_indices,
    attention_masks: NativeKvMasks,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    constant_keepalive: list[np.ndarray],
    tag: str,
):
    k_4d = _reshape_rows_to_heads(network, k, num_heads=num_kv_heads, head_dim=head_dim)
    v_4d = _reshape_rows_to_heads(network, v, num_heads=num_kv_heads, head_dim=head_dim)
    update_k = network.add_kv_cache_update(
        cache_k, k_4d, cache_write_indices, trt.KVCacheMode.LINEAR
    )
    update_v = network.add_kv_cache_update(
        cache_v, v_4d, cache_write_indices, trt.KVCacheMode.LINEAR
    )
    if update_k is None or update_v is None:
        raise RuntimeError("TensorRT failed to create K2-Horizon-Uno KV updates")
    update_k.name = tag + ".cache_k_update"
    update_v.name = tag + ".cache_v_update"
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
        constant_keepalive=constant_keepalive,
        scale=float(1.0 / np.sqrt(head_dim)),
        tag=tag,
    )
    return {
        "context": _reshape_heads_to_rows(network, context_4d, width=num_heads * head_dim),
        "present_k": present_k,
        "present_v": present_v,
    }


def build_engine(
    cfg: K2HorizonUnoConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    verbose: bool = False,
) -> bytes:
    """Build one dynamic ``S=1..8`` BF16 engine with conditional Uno LoRA."""

    if isinstance(max_cache_length, bool) or not isinstance(max_cache_length, int):
        raise ValueError("K2-Horizon-Uno max_cache_length must be an integer")
    if max_cache_length < cfg.max_block_size or max_cache_length > cfg.max_position_embeddings:
        raise ValueError(
            "K2-Horizon-Uno max_cache_length must be at least max_block_size and no "
            f"greater than {cfg.max_position_embeddings}"
        )
    work_dtype = trt.bfloat16
    constant_keepalive: list[np.ndarray] = []

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input("cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    lora_mask = network.add_input("lora_mask", trt.float32, (-1,))
    if any(
        value is None
        for value in (token_id, position_id, cache_write_indices, key_value_lengths, lora_mask)
    ):
        raise RuntimeError("TensorRT failed to create K2-Horizon-Uno engine inputs")

    profile = builder.create_optimization_profile()
    optimum = cfg.max_block_size
    for name in ("token_id", "position_id", "lora_mask"):
        # TensorRT 11.1 mutates the profile in place and returns ``None`` from
        # this Python binding.  Registration below is the authoritative
        # validation boundary.
        profile.set_shape(name, (1,), (optimum,), (cfg.max_block_size,))
    if builder_config.add_optimization_profile(profile) < 0:
        raise RuntimeError("TensorRT rejected the K2-Horizon-Uno optimization profile")

    attention_masks = add_active_prefix_causal_masks(
        network,
        token_id,
        cache_write_indices,
        key_value_lengths,
        max_cache_length,
        constant_keepalive=constant_keepalive,
    )
    cache_shape = (1, cfg.num_key_value_heads, max_cache_length, cfg.head_dim)
    cache_k_inputs = []
    cache_v_inputs = []
    for layer_index in range(cfg.num_hidden_layers):
        cache_k_inputs.append(
            network.add_input(layer_tensor_name("cache_k", layer_index), work_dtype, cache_shape)
        )
        cache_v_inputs.append(
            network.add_input(layer_tensor_name("cache_v", layer_index), work_dtype, cache_shape)
        )

    embedding = _work_constant(
        network,
        (cfg.vocab_size, cfg.hidden_size),
        weights["embedding"],
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )
    hidden_state = _cast(
        network, network.add_gather(embedding, token_id, 0).get_output(0), work_dtype
    )
    cos_cache, sin_cache = _active_rope_cache(
        network,
        position_id,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        work_dtype=work_dtype,
        constant_keepalive=constant_keepalive,
    )

    present_k_outputs = []
    present_v_outputs = []
    for layer_index in range(cfg.num_hidden_layers):
        prefix = f"layer.{layer_index}"

        def projection(lhs, logical: str, lhs_width: int, rhs_width: int, graph_name: str):
            return _conditional_projection(
                network,
                lhs,
                weights[f"{prefix}.{logical}"],
                weights[f"{prefix}.{logical}_lora_a"],
                weights[f"{prefix}.{logical}_lora_b"],
                lora_mask,
                lhs_width=lhs_width,
                rhs_width=rhs_width,
                rank=cfg.lora_rank,
                scale=cfg.lora_scale,
                work_dtype=work_dtype,
                constant_keepalive=constant_keepalive,
                name=f"{prefix}.{graph_name}",
            )

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
        q = projection(normed, "w_q", cfg.hidden_size, cfg.attention_size, "self_attn.q_proj")
        k = projection(normed, "w_k", cfg.hidden_size, cfg.kv_attention_size, "self_attn.k_proj")
        v = projection(normed, "w_v", cfg.hidden_size, cfg.kv_attention_size, "self_attn.v_proj")
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
            constant_keepalive=constant_keepalive,
            tag=f"{prefix}.self_attn",
        )
        attn_out = projection(
            attention["context"],
            "w_o",
            cfg.attention_size,
            cfg.hidden_size,
            "self_attn.o_proj",
        )
        hidden_state = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM
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
        gate = projection(
            mlp_input,
            "w_gate",
            cfg.hidden_size,
            cfg.intermediate_size,
            "mlp.gate_proj",
        )
        up = projection(
            mlp_input,
            "w_up",
            cfg.hidden_size,
            cfg.intermediate_size,
            "mlp.up_proj",
        )
        gated = network.add_elementwise(
            _silu(network, gate), up, trt.ElementWiseOperation.PROD
        ).get_output(0)
        mlp_out = projection(
            gated,
            "w_down",
            cfg.intermediate_size,
            cfg.hidden_size,
            "mlp.down_proj",
        )
        hidden_state = network.add_elementwise(
            hidden_state, mlp_out, trt.ElementWiseOperation.SUM
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
            "[trtmc build] Building K2-Horizon-Uno BF16 dynamic-block engine "
            f"(layers={cfg.num_hidden_layers}, cache={max_cache_length}, "
            f"block=1..{cfg.max_block_size}, rank={cfg.lora_rank}) ...",
            file=sys.stderr,
        )
    try:
        plan = builder.build_serialized_network(network, builder_config)
    finally:
        constant_keepalive.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build the K2-Horizon-Uno engine")
    return bytes(plan)


_BASE_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors.index.json",
    "pytorch_model-*.safetensors",
    "tokenizer.json",
    "chat_template.jinja",
)
_BUNDLE_FILES = ("tokenizer.json", "chat_template.jinja")


def _load_config(model_dir: Path) -> SimpleNamespace:
    path = model_dir / "config.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid K2-Horizon-Uno base config: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("K2-Horizon-Uno base config.json must contain one object")
    return SimpleNamespace(raw=raw, **raw)


def _resolve_base_model() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=BASE_MODEL_ID,
            revision=BASE_REVISION,
            allow_patterns=_BASE_ALLOW_PATTERNS,
        )
    )


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build the qualified BF16 dynamic-block K2-Horizon-7B-Uno bundle."""

    if request.dynamic_kv_cache:
        raise NotImplementedError("k2_horizon_uno does not support dynamic_kv_cache")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("K2-Horizon-Uno does not support image dimensions")
    if request.video_num_frames is not None:
        raise NotImplementedError("K2-Horizon-Uno does not support video inputs")
    if request.max_batch_size != 1:
        raise NotImplementedError("K2-Horizon-Uno supports only max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("K2-Horizon-Uno does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("K2-Horizon-Uno does not support context parallelism")
    if request.task != "text_generation":
        raise ValueError("K2-Horizon-Uno supports only task=text_generation")
    if str(request.precision).lower() != "bf16":
        raise ValueError("K2-Horizon-Uno currently supports only BF16 builds")
    if request.backend != "trt":
        raise NotImplementedError("K2-Horizon-Uno currently supports only backend=trt")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("K2-Horizon-Uno does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("K2-Horizon-Uno does not support mixed-FP32 layers")

    adapter_dir = Path(request.model_dir)
    load_and_validate_adapter_config(adapter_dir / "adapter_config.json")
    if not (adapter_dir / ADAPTER_FILENAME).is_file():
        raise FileNotFoundError(f"K2-Horizon-Uno adapter is missing {ADAPTER_FILENAME}")
    base_dir = _resolve_base_model()
    source_config = _load_config(base_dir)
    config = validate_config(source_config)
    max_cache_length = (
        min(256, config.max_position_embeddings)
        if request.max_sequence_length is None
        else request.max_sequence_length
    )
    if (
        max_cache_length < config.max_block_size
        or max_cache_length > config.max_position_embeddings
    ):
        raise ValueError(
            "K2-Horizon-Uno max_sequence_length must cover block length 8 and stay "
            "within checkpoint capacity"
        )
    weights = load_standard_weights(
        base_dir,
        adapter_dir,
        config,
    )
    plan = build_engine(
        config,
        weights,
        max_cache_length,
        verbose=bool(request.verbose),
    )

    tokenizer = base_dir / "tokenizer.json"
    chat_template = base_dir / "chat_template.jinja"
    if not tokenizer.is_file() or not chat_template.is_file():
        raise FileNotFoundError(
            "K2-Horizon-Uno base must contain tokenizer.json and chat_template.jinja"
        )
    writer.set_header(family="k2_horizon_uno", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", {"max_cache_length": max_cache_length})
    for filename in _BUNDLE_FILES:
        path = base_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
