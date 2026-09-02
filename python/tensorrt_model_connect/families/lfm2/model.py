# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT-Python fixed-step graph for dense LFM2 causal decoders."""

from __future__ import annotations

import sys

import numpy as np
from tensorrt_model_connect import trt_compat
from ...native_kv_attention_builder import (
    NativeKvMasks,
    add_active_prefix_causal_masks,
    add_explicit_masked_grouped_query_attention,
)

from .checkpoint_mapper import WeightDict
from .config import DenseLfm2Config, validate_dense_lfm2_config


trt = trt_compat.get_trt()


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def _precision_dtypes(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    if precision == "bf16":
        # NumPy has no portable BF16 weights type. Build constants in FP32 and
        # cast once inside the strongly typed TensorRT graph.
        return np.dtype(np.float32), trt.bfloat16
    raise ValueError("dense LFM2 supports only fp16 and bf16 builds")


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
    *,
    np_dtype: np.dtype = np.dtype(np.float32),
) -> trt.ITensor:
    array = np.ascontiguousarray(values, dtype=np_dtype).reshape(shape)
    return network.add_constant(shape, trt.Weights(array)).get_output(0)


def _work_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    *,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    tensor = _constant(network, shape, values, np_dtype=storage_dtype)
    return _cast(network, tensor, work_dtype)


def _matmul(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: np.ndarray,
    *,
    lhs_width: int,
    rhs_width: int,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
    name: str,
) -> trt.ITensor:
    array = np.asarray(rhs)
    expected = (lhs_width, rhs_width)
    if tuple(array.shape) != expected:
        raise ValueError(
            f"mapped LFM2 weight {name} must have shape {expected}, got {tuple(array.shape)}"
        )
    rhs_tensor = _work_constant(
        network,
        expected,
        array,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    layer = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs_tensor,
        trt.MatrixOperation.NONE,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add LFM2 matmul {name}")
    layer.name = name
    return layer.get_output(0)


def _bias(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    values: np.ndarray | None,
    *,
    width: int,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    if values is None:
        return tensor
    if tuple(np.asarray(values).shape) != (width,):
        raise ValueError(f"mapped LFM2 bias must have shape ({width},)")
    bias = _work_constant(
        network,
        (1, width),
        values,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    return network.add_elementwise(
        tensor,
        bias,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _rms_norm(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    gamma: np.ndarray,
    *,
    width: int,
    epsilon: float,
    output_dtype: trt.DataType,
) -> trt.ITensor:
    if tuple(np.asarray(gamma).shape) != (width,):
        raise ValueError(f"mapped LFM2 RMSNorm weight must have shape ({width},)")
    fp32 = _cast(network, tensor, trt.float32)
    rank = len(tuple(fp32.shape))
    axes = 1 << (rank - 1)
    squared = network.add_elementwise(
        fp32,
        fp32,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    mean = network.add_reduce(
        squared,
        trt.ReduceOperation.AVG,
        axes,
        True,
    ).get_output(0)
    eps_shape = (1,) * rank
    eps = _constant(
        network,
        eps_shape,
        np.full(eps_shape, epsilon, dtype=np.float32),
    )
    variance = network.add_elementwise(
        mean,
        eps,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    root = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    reciprocal = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        fp32,
        reciprocal,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    # Match HF Lfm2RMSNorm exactly: variance and normalization are FP32, then
    # normalized activations return to the model dtype before model-dtype gamma
    # is applied. Multiplying gamma in FP32 changes FP16/BF16 rounding.
    normalized = _cast(network, normalized, output_dtype)
    gamma_shape = (1,) * (rank - 1) + (width,)
    gamma_storage_dtype = (
        np.dtype(np.float16) if output_dtype == trt.float16 else np.dtype(np.float32)
    )
    gamma_tensor = _work_constant(
        network,
        gamma_shape,
        np.asarray(gamma, dtype=np.float32).reshape(gamma_shape),
        storage_dtype=gamma_storage_dtype,
        work_dtype=output_dtype,
    )
    scaled = network.add_elementwise(
        normalized,
        gamma_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return scaled


def _per_head_rms_norm(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    gamma: np.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    epsilon: float,
    output_dtype: trt.DataType,
) -> trt.ITensor:
    if tuple(np.asarray(gamma).shape) != (head_dim,):
        raise ValueError(f"mapped LFM2 QK norm must have shape ({head_dim},)")
    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (1, num_heads, head_dim)
    normalized = _rms_norm(
        network,
        shaped.get_output(0),
        gamma,
        width=head_dim,
        epsilon=epsilon,
        output_dtype=output_dtype,
    )
    flattened = network.add_shuffle(normalized)
    flattened.reshape_dims = (1, num_heads * head_dim)
    return flattened.get_output(0)


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
) -> tuple[trt.ITensor, trt.ITensor]:
    if head_dim % 2:
        raise ValueError("dense LFM2 RoPE requires an even head_dim")
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
        raise RuntimeError("TensorRT failed to create LFM2 rotary embedding")
    return _reshape_heads_to_rows(
        network,
        rope.get_output(0),
        width=num_heads * head_dim,
    )


def add_native_kv_attention(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k_update: trt.ITensor,
    v_update: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    attention_masks: NativeKvMasks,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    work_dtype: trt.DataType,
    tag: str,
) -> dict[str, trt.ITensor]:
    """Use TensorRT's in-place KV update and explicit masked attention."""

    if not hasattr(network, "add_kv_cache_update"):
        raise RuntimeError("LFM2 requires TensorRT add_kv_cache_update")
    if work_dtype not in {trt.float16, trt.bfloat16}:
        raise ValueError("LFM2 native KV attention requires FP16 or BF16")

    k_4d = _reshape_rows_to_heads(
        network,
        k_update,
        num_heads=num_kv_heads,
        head_dim=head_dim,
    )
    v_4d = _reshape_rows_to_heads(
        network,
        v_update,
        num_heads=num_kv_heads,
        head_dim=head_dim,
    )
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
        raise RuntimeError("TensorRT failed to create LFM2 KV-cache update layers")
    update_k.name = f"{tag}.cache_k_update"
    update_v.name = f"{tag}.cache_v_update"
    present_k = update_k.get_output(0)
    present_v = update_v.get_output(0)

    q_4d = _reshape_rows_to_heads(
        network,
        q,
        num_heads=num_heads,
        head_dim=head_dim,
    )
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
    context = _reshape_heads_to_rows(
        network,
        context_4d,
        width=num_heads * head_dim,
    )
    return {
        "context": context,
        "present_k": present_k,
        "present_v": present_v,
    }


def _add_conv_layer(
    network: trt.INetworkDefinition,
    hidden_state: trt.ITensor,
    conv_state: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    config: DenseLfm2Config,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    hidden = config.hidden_size
    projected = _matmul(
        network,
        hidden_state,
        weights[f"{prefix}.conv_in"],
        lhs_width=hidden,
        rhs_width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.in_proj",
    )
    projected = _bias(
        network,
        projected,
        weights.get(f"{prefix}.conv_in_bias"),
        width=3 * hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )

    parts: list[trt.ITensor] = []
    for index in range(3):
        part = network.add_slice(
            projected,
            (0, index * hidden),
            (1, hidden),
            (1, 1),
        )
        parts.append(part.get_output(0))
    b_gate, c_gate, x_value = parts
    bx = network.add_elementwise(
        b_gate,
        x_value,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    bx_column = network.add_shuffle(bx)
    bx_column.reshape_dims = (hidden, 1)
    if config.conv_l_cache > 1:
        prior = network.add_slice(
            conv_state,
            (0, 1),
            (hidden, config.conv_l_cache - 1),
            (1, 1),
        ).get_output(0)
        updated = network.add_concatenation([prior, bx_column.get_output(0)])
        updated.axis = 1
        present_conv = updated.get_output(0)
    else:
        present_conv = bx_column.get_output(0)

    conv_weights = _work_constant(
        network,
        (hidden, config.conv_l_cache),
        weights[f"{prefix}.conv_weight"],
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    weighted = network.add_elementwise(
        present_conv,
        conv_weights,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    convolved = network.add_reduce(
        weighted,
        trt.ReduceOperation.SUM,
        1 << 1,
        True,
    ).get_output(0)
    conv_row = network.add_shuffle(convolved)
    conv_row.reshape_dims = (1, hidden)
    conv_row_tensor = _bias(
        network,
        conv_row.get_output(0),
        weights.get(f"{prefix}.conv_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    gated = network.add_elementwise(
        c_gate,
        conv_row_tensor,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    output = _matmul(
        network,
        gated,
        weights[f"{prefix}.conv_out"],
        lhs_width=hidden,
        rhs_width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.conv.out_proj",
    )
    output = _bias(
        network,
        output,
        weights.get(f"{prefix}.conv_out_bias"),
        width=hidden,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    return output, present_conv


def _add_swiglu(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    config: DenseLfm2Config,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    left = _matmul(
        network,
        tensor,
        weights[f"{prefix}.w1"],
        lhs_width=config.hidden_size,
        rhs_width=config.intermediate_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w1",
    )
    right = _matmul(
        network,
        tensor,
        weights[f"{prefix}.w3"],
        lhs_width=config.hidden_size,
        rhs_width=config.intermediate_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w3",
    )
    gated = network.add_elementwise(
        _silu(network, left),
        right,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return _matmul(
        network,
        gated,
        weights[f"{prefix}.w2"],
        lhs_width=config.intermediate_size,
        rhs_width=config.hidden_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.w2",
    )


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    output = network.add_identity(_cast(network, tensor, trt.float32)).get_output(0)
    output.name = name
    network.mark_output(output)


def _validate_mapped_metadata(config: DenseLfm2Config, weights: WeightDict) -> None:
    expected = {
        "_layer_types": list(config.layer_types),
        "_num_attention_layers": config.num_attention_layers,
        "_num_conv_layers": config.num_conv_layers,
        "_hidden_size": config.hidden_size,
        "_intermediate_size": config.intermediate_size,
        "_head_dim": config.head_dim,
        "_conv_l_cache": config.conv_l_cache,
    }
    for name, value in expected.items():
        if weights.get(name) != value:
            raise ValueError(
                f"mapped LFM2 metadata {name} must be {value!r}, got {weights.get(name)!r}"
            )


def build_lfm2_engine(
    config: object,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build a static one-token LFM2 engine with native TensorRT KV updates."""

    cfg = validate_dense_lfm2_config(config)
    _validate_mapped_metadata(cfg, weights)
    if max_cache_length <= 0:
        raise ValueError("max_cache_length must be positive")
    if max_cache_length > cfg.max_position_embeddings:
        raise ValueError(
            f"max_cache_length exceeds LFM2 max_position_embeddings ({cfg.max_position_embeddings})"
        )
    storage_dtype, work_dtype = _precision_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(trt_compat.network_creation_flags(strongly_typed=True))
    builder_config = builder.create_builder_config()
    # Match the qualified native-KV builder ceiling. This is build-time tactic
    # scratch, not an eager runtime allocation.
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    cache_write_indices = network.add_input(
        "cache_write_indices",
        trt.int32,
        (1,),
    )
    key_value_lengths = network.add_input(
        "key_value_lengths",
        trt.int32,
        (1,),
    )
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
    for attention_index in range(cfg.num_attention_layers):
        cache_k_inputs.append(
            network.add_input(
                layer_tensor_name("cache_k", attention_index),
                work_dtype,
                cache_shape,
            )
        )
        cache_v_inputs.append(
            network.add_input(
                layer_tensor_name("cache_v", attention_index),
                work_dtype,
                cache_shape,
            )
        )

    conv_state_inputs: list[trt.ITensor] = []
    for conv_index in range(cfg.num_conv_layers):
        conv_state_inputs.append(
            network.add_input(
                layer_tensor_name("conv_state", conv_index),
                work_dtype,
                (cfg.hidden_size, cfg.conv_l_cache),
            )
        )

    embedding = _work_constant(
        network,
        (cfg.vocab_size, cfg.hidden_size),
        weights["embedding"],
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    hidden_state = network.add_gather(embedding, token_id, 0).get_output(0)
    hidden_state = _cast(network, hidden_state, work_dtype)
    cos_cache, sin_cache = _active_rope_cache(
        network,
        position_id,
        head_dim=cfg.head_dim,
        rope_theta=cfg.rope_theta,
        work_dtype=work_dtype,
    )

    present_k_outputs: list[trt.ITensor] = []
    present_v_outputs: list[trt.ITensor] = []
    present_conv_outputs: list[trt.ITensor] = []
    attention_index = 0
    conv_index = 0

    for layer_index, layer_type in enumerate(cfg.layer_types):
        prefix = f"layer.{layer_index}"
        operator_input = _rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.operator_norm"],
            width=cfg.hidden_size,
            epsilon=cfg.norm_eps,
            output_dtype=work_dtype,
        )
        if layer_type == "conv":
            operator_output, present_conv = _add_conv_layer(
                network,
                operator_input,
                conv_state_inputs[conv_index],
                weights,
                prefix=prefix,
                config=cfg,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
            )
            present_conv_outputs.append(present_conv)
            conv_index += 1
        else:
            q = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_q"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.q_proj",
            )
            k = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_k"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.kv_attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.k_proj",
            )
            v = _matmul(
                network,
                operator_input,
                weights[f"{prefix}.w_v"],
                lhs_width=cfg.hidden_size,
                rhs_width=cfg.kv_attention_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.v_proj",
            )
            q = _per_head_rms_norm(
                network,
                q,
                weights[f"{prefix}.q_norm"],
                num_heads=cfg.num_attention_heads,
                head_dim=cfg.head_dim,
                epsilon=cfg.norm_eps,
                output_dtype=work_dtype,
            )
            k = _per_head_rms_norm(
                network,
                k,
                weights[f"{prefix}.k_norm"],
                num_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                epsilon=cfg.norm_eps,
                output_dtype=work_dtype,
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
            native = add_native_kv_attention(
                network,
                q,
                k,
                v,
                cache_k_inputs[attention_index],
                cache_v_inputs[attention_index],
                cache_write_indices,
                attention_masks,
                num_heads=cfg.num_attention_heads,
                num_kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                work_dtype=work_dtype,
                tag=f"{prefix}.self_attn",
            )
            operator_output = _matmul(
                network,
                native["context"],
                weights[f"{prefix}.w_o"],
                lhs_width=cfg.attention_size,
                rhs_width=cfg.hidden_size,
                storage_dtype=storage_dtype,
                work_dtype=work_dtype,
                name=f"{prefix}.self_attn.out_proj",
            )
            present_k_outputs.append(native["present_k"])
            present_v_outputs.append(native["present_v"])
            attention_index += 1

        hidden_state = network.add_elementwise(
            hidden_state,
            operator_output,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        ffn_input = _rms_norm(
            network,
            hidden_state,
            weights[f"{prefix}.ffn_norm"],
            width=cfg.hidden_size,
            epsilon=cfg.norm_eps,
            output_dtype=work_dtype,
        )
        ffn_output = _add_swiglu(
            network,
            ffn_input,
            weights,
            prefix=prefix,
            config=cfg,
            storage_dtype=storage_dtype,
            work_dtype=work_dtype,
        )
        hidden_state = network.add_elementwise(
            hidden_state,
            ffn_output,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        if debug_layer_outputs:
            _mark_debug_output(
                network,
                hidden_state,
                f"debug_hidden_{layer_index}",
            )

    if attention_index != cfg.num_attention_layers or conv_index != cfg.num_conv_layers:
        raise RuntimeError("LFM2 layer/state counters did not consume every input")

    hidden_state = _rms_norm(
        network,
        hidden_state,
        weights["final_norm"],
        width=cfg.hidden_size,
        epsilon=cfg.norm_eps,
        output_dtype=work_dtype,
    )
    logits = _matmul(
        network,
        hidden_state,
        weights["w_lm_head"],
        lhs_width=cfg.hidden_size,
        rhs_width=cfg.vocab_size,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name="lm_head",
    )
    logits = _cast(network, logits, trt.float32)
    logits.name = "logits"
    network.mark_output(logits)

    for index, tensor in enumerate(present_k_outputs):
        tensor.name = layer_tensor_name("present_k", index)
        network.mark_output(tensor)
    for index, tensor in enumerate(present_v_outputs):
        tensor.name = layer_tensor_name("present_v", index)
        network.mark_output(tensor)
    for index, tensor in enumerate(present_conv_outputs):
        tensor.name = layer_tensor_name("present_conv", index)
        network.mark_output(tensor)

    if verbose:
        print(
            "[trtmc build] Building LFM2 native hybrid engine "
            f"(layers={cfg.num_hidden_layers}, conv={cfg.num_conv_layers}, "
            f"attention={cfg.num_attention_layers}, hidden={cfg.hidden_size}, "
            f"cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the LFM2 engine")
    return bytes(plan)
