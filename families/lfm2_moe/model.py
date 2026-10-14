# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure TensorRT-Python fixed-step graph for LFM2-MoE causal decoders."""

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

from .checkpoint_mapper import WeightDict, load_lfm2_moe_weights
from .config import Lfm2MoeConfig, validate_lfm2_moe_config


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


def _precision_dtypes(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    if precision == "bf16":
        # NumPy has no portable BF16 weights type. Build constants in FP32 and
        # cast once inside the strongly typed TensorRT graph.
        return np.dtype(np.float32), trt.bfloat16
    raise ValueError("LFM2-MoE supports only fp16 and bf16 builds")


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
            f"mapped LFM2-MoE weight {name} must have shape {expected}, got {tuple(array.shape)}"
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
        raise RuntimeError(f"TensorRT failed to add LFM2-MoE matmul {name}")
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
        raise ValueError(f"mapped LFM2-MoE bias must have shape ({width},)")
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
        raise ValueError(f"mapped LFM2-MoE RMSNorm weight must have shape ({width},)")
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
    # Match HF Lfm2MoeRMSNorm exactly: variance and normalization are FP32, then
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
        raise ValueError(f"mapped LFM2-MoE QK norm must have shape ({head_dim},)")
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
    # F.silu rounds once after sigmoid and multiplication, not between them.
    dtype = tensor.dtype
    fp32 = _cast(network, tensor, trt.float32)
    sigmoid = network.add_activation(fp32, trt.ActivationType.SIGMOID)
    output = network.add_elementwise(
        fp32, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    return _cast(network, output, dtype)


def _active_rope_cache(
    network: trt.INetworkDefinition,
    position_id: trt.ITensor,
    *,
    head_dim: int,
    rope_theta: float,
    work_dtype: trt.DataType,
) -> tuple[trt.ITensor, trt.ITensor]:
    if head_dim % 2:
        raise ValueError("LFM2-MoE RoPE requires an even head_dim")
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
    # The reference rounds the cosine and sine products separately before
    # adding them. Keep native rotary operators, with a model-dtype boundary
    # for each component, instead of one fused multiply-add rounding.
    zero = _cast(
        network,
        _constant(network, (1, 1, head_dim // 2), np.zeros((1, 1, head_dim // 2), np.float32)),
        heads.dtype,
    )
    cosine = network.add_rotary_embedding(heads, cos_cache, zero, False, head_dim)
    sine = network.add_rotary_embedding(heads, zero, sin_cache, False, head_dim)
    if cosine is None or sine is None:
        raise RuntimeError("TensorRT failed to create rotary components")
    summed = network.add_elementwise(
        _cast(network, cosine.get_output(0), trt.float32),
        _cast(network, sine.get_output(0), trt.float32),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _reshape_heads_to_rows(
        network,
        _cast(network, summed, heads.dtype),
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

    if work_dtype not in {trt.float16, trt.bfloat16}:
        raise ValueError("LFM2-MoE native KV attention requires FP16 or BF16")

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
        raise RuntimeError("TensorRT failed to create LFM2-MoE KV-cache update layers")
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
    decode_phase: trt.ITensor,
    weights: WeightDict,
    *,
    position_id: trt.ITensor,
    prefix: str,
    config: Lfm2MoeConfig,
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
        first = network.add_slice(
            conv_state, (0, 0), (hidden, 1), (1, 1)
        ).get_output(0)
        rolled = network.add_concatenation([prior, first])
        rolled.axis = 1
        position = network.add_shuffle(position_id)
        position.reshape_dims = (1, 1)
        last = _constant(
            network, (1, 1), np.array([[config.conv_l_cache - 1]], dtype=np.int32),
            np_dtype=np.dtype(np.int32),
        )
        cached_index = network.add_elementwise(
            position.get_output(0), last, trt.ElementWiseOperation.MIN
        ).get_output(0)
        # The original slow cached path writes at clamp(cache_position, 0, L-1)
        # after rolling, even when a short prefix has not yet filled the cache.
        # Prefill remains an append at the last column.
        write_index = network.add_select(decode_phase, cached_index, last).get_output(0)
        columns = _constant(
            network, (1, config.conv_l_cache), np.arange(config.conv_l_cache, dtype=np.int32),
            np_dtype=np.dtype(np.int32),
        )
        write_mask = network.add_elementwise(
            columns, write_index, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
        present_conv = network.add_select(
            write_mask, bx_column.get_output(0), rolled.get_output(0)
        ).get_output(0)
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
        _cast(network, present_conv, trt.float32),
        _cast(network, conv_weights, trt.float32),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    # Cached reference decode rounds elementwise products to model dtype;
    # Conv1d prefill accumulates unrounded products until its output boundary.
    rounded_products = _cast(network, _cast(network, weighted, work_dtype), trt.float32)
    weighted = network.add_select(decode_phase, rounded_products, weighted).get_output(0)
    convolved = network.add_reduce(
        weighted,
        trt.ReduceOperation.SUM,
        1 << 1,
        True,
    ).get_output(0)
    conv_row = network.add_shuffle(_cast(network, convolved, work_dtype))
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
    config: Lfm2MoeConfig,
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


def _add_moe_ffn(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    weights: WeightDict,
    *,
    prefix: str,
    config: Lfm2MoeConfig,
    storage_dtype: np.dtype,
    work_dtype: trt.DataType,
) -> trt.ITensor:
    """Single-token routing and experts matching the Transformers BF16 path.

    Router matmul, sigmoid, and combine weights use the model dtype. The FP32
    expert bias changes selection only; it never changes the gathered weights.
    """

    hidden = config.hidden_size
    experts = config.num_experts
    top_k = config.num_experts_per_tok
    inter = config.moe_intermediate_size

    router_weight = np.asarray(weights[f"{prefix}.router"])
    if tuple(router_weight.shape) != (experts, hidden):
        raise ValueError(
            f"mapped LFM2-MoE router {prefix}.router must have shape "
            f"{(experts, hidden)}, got {tuple(router_weight.shape)}"
        )
    expert_bias = np.asarray(weights[f"{prefix}.expert_bias"])
    if tuple(expert_bias.shape) != (experts,):
        raise ValueError(
            f"mapped LFM2-MoE bias {prefix}.expert_bias must have shape "
            f"{(experts,)}, got {tuple(expert_bias.shape)}"
        )
    stacked_w13 = np.asarray(weights[f"{prefix}.moe_w13"])
    if tuple(stacked_w13.shape) != (experts, hidden, 2 * inter):
        raise ValueError(
            f"mapped LFM2-MoE weight {prefix}.moe_w13 must have shape "
            f"{(experts, hidden, 2 * inter)}, got {tuple(stacked_w13.shape)}"
        )
    stacked_w2 = np.asarray(weights[f"{prefix}.moe_w2"])
    if tuple(stacked_w2.shape) != (experts, inter, hidden):
        raise ValueError(
            f"mapped LFM2-MoE weight {prefix}.moe_w2 must have shape "
            f"{(experts, inter, hidden)}, got {tuple(stacked_w2.shape)}"
        )

    logits = _matmul(
        network,
        tensor,
        np.ascontiguousarray(router_weight.T),
        lhs_width=hidden,
        rhs_width=experts,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
        name=f"{prefix}.feed_forward.gate",
    )
    scores = network.add_activation(logits, trt.ActivationType.SIGMOID).get_output(0)

    # 2. Selection uses biased scores; the bias never reaches combine weights.
    bias_tensor = _constant(
        network,
        (1, experts),
        np.ascontiguousarray(expert_bias.reshape(1, experts), dtype=np.float32),
    )
    biased = network.add_elementwise(
        _cast(network, scores, trt.float32),
        bias_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    topk = network.add_topk(biased, trt.TopKOperation.MAX, top_k, 1 << 1)
    if topk is None:
        raise RuntimeError(f"TensorRT failed to add LFM2-MoE top-k {prefix}")
    topk.name = f"{prefix}.feed_forward.topk"
    # The default grouped_mm reference restores router top-k order before its
    # final reduction, rather than the eager implementation's expert-index loop.
    indices_flat = network.add_shuffle(topk.get_output(1))
    indices_flat.reshape_dims = (top_k,)
    indices = indices_flat.get_output(0)

    # 3. Normalize the unbiased scores in the reference model dtype.
    scores_flat = network.add_shuffle(scores)
    scores_flat.reshape_dims = (experts,)
    combine = network.add_gather(scores_flat.get_output(0), indices, 0).get_output(0)
    denominator = network.add_reduce(
        combine,
        trt.ReduceOperation.SUM,
        1 << 0,
        True,
    ).get_output(0)
    epsilon = _cast(
        network, _constant(network, (1,), np.array([1e-6], dtype=np.float32)), work_dtype
    )
    denominator = network.add_elementwise(
        denominator,
        epsilon,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    combine = network.add_elementwise(
        combine,
        denominator,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)

    # 4. Gather the two stacked expert constants in work dtype so the per-step
    #    copies move half the bytes of an FP32 gather.
    w13_tensor = _work_constant(
        network,
        (experts, hidden, 2 * inter),
        stacked_w13,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    w2_tensor = _work_constant(
        network,
        (experts, inter, hidden),
        stacked_w2,
        storage_dtype=storage_dtype,
        work_dtype=work_dtype,
    )
    gathered_w13 = network.add_gather(w13_tensor, indices, 0).get_output(0)
    gathered_w2 = network.add_gather(w2_tensor, indices, 0).get_output(0)

    # 5. Broadcast the token row against the selected experts.
    row = network.add_shuffle(tensor)
    row.reshape_dims = (1, 1, hidden)
    up_gate_layer = network.add_matrix_multiply(
        row.get_output(0),
        trt.MatrixOperation.NONE,
        gathered_w13,
        trt.MatrixOperation.NONE,
    )
    if up_gate_layer is None:
        raise RuntimeError(f"TensorRT failed to add LFM2-MoE expert w13 matmul {prefix}")
    up_gate_layer.name = f"{prefix}.feed_forward.experts.w13"
    up_gate = up_gate_layer.get_output(0)
    gate = network.add_slice(
        up_gate,
        (0, 0, 0),
        (top_k, 1, inter),
        (1, 1, 1),
    ).get_output(0)
    up = network.add_slice(
        up_gate,
        (0, 0, inter),
        (top_k, 1, inter),
        (1, 1, 1),
    ).get_output(0)
    activated = network.add_elementwise(
        _silu(network, gate),
        up,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    down_layer = network.add_matrix_multiply(
        activated,
        trt.MatrixOperation.NONE,
        gathered_w2,
        trt.MatrixOperation.NONE,
    )
    if down_layer is None:
        raise RuntimeError(f"TensorRT failed to add LFM2-MoE expert w2 matmul {prefix}")
    down_layer.name = f"{prefix}.feed_forward.experts.w2"
    per_expert = down_layer.get_output(0)

    # 6. Weighted sum over the selected experts back to a [1, hidden] row.
    combine_column = network.add_shuffle(combine)
    combine_column.reshape_dims = (top_k, 1, 1)
    combine_work = _cast(network, combine_column.get_output(0), work_dtype)
    weighted = network.add_elementwise(
        per_expert,
        combine_work,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    # grouped_mm sums the already-rounded model-dtype contributions with FP32
    # accumulation and rounds once. Sequential BF16 additions differ.
    summed = network.add_reduce(
        _cast(network, weighted, trt.float32), trt.ReduceOperation.SUM, 1 << 0, True
    ).get_output(0)
    output = network.add_shuffle(_cast(network, summed, work_dtype))
    output.reshape_dims = (1, hidden)
    return output.get_output(0)


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    output = network.add_identity(_cast(network, tensor, trt.float32)).get_output(0)
    output.name = name
    network.mark_output(output)


def _validate_mapped_metadata(config: Lfm2MoeConfig, weights: WeightDict) -> None:
    expected = {
        "_layer_types": list(config.layer_types),
        "_num_attention_layers": config.num_attention_layers,
        "_num_conv_layers": config.num_conv_layers,
        "_hidden_size": config.hidden_size,
        "_intermediate_size": config.intermediate_size,
        "_head_dim": config.head_dim,
        "_conv_l_cache": config.conv_l_cache,
        "_num_experts": config.num_experts,
        "_num_experts_per_tok": config.num_experts_per_tok,
        "_moe_intermediate_size": config.moe_intermediate_size,
        "_num_dense_layers": config.num_dense_layers,
    }
    for name, value in expected.items():
        if weights.get(name) != value:
            raise ValueError(
                f"mapped LFM2-MoE metadata {name} must be {value!r}, got {weights.get(name)!r}"
            )


def build_lfm2_moe_engine(
    config: object,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
    debug_layer_outputs: bool = False,
) -> bytes:
    """Build a static one-token LFM2-MoE engine with native TensorRT KV updates."""

    cfg = validate_lfm2_moe_config(config)
    _validate_mapped_metadata(cfg, weights)
    if max_cache_length <= 0:
        raise ValueError("max_cache_length must be positive")
    if max_cache_length > cfg.max_position_embeddings:
        raise ValueError(
            f"max_cache_length exceeds LFM2-MoE max_position_embeddings ({cfg.max_position_embeddings})"
        )
    storage_dtype, work_dtype = _precision_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    # Match the qualified native-KV builder ceiling. This is build-time tactic
    # scratch, not an eager runtime allocation.
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)

    decode_phase = network.add_input("decode_phase", trt.int32, (1, 1))
    zero = _constant(network, (1, 1), np.zeros((1, 1), dtype=np.int32), np_dtype=np.dtype(np.int32))
    decode_phase = network.add_elementwise(
        decode_phase, zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
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
                decode_phase,
                weights,
                position_id=position_id,
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
        ffn_builder = _add_moe_ffn if cfg.is_moe_layer(layer_index) else _add_swiglu
        ffn_output = ffn_builder(
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
        raise RuntimeError("LFM2-MoE layer/state counters did not consume every input")

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
            "[trtmc build] Building LFM2-MoE native hybrid engine "
            f"(layers={cfg.num_hidden_layers}, conv={cfg.num_conv_layers}, "
            f"attention={cfg.num_attention_layers}, hidden={cfg.hidden_size}, "
            f"cache={max_cache_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the LFM2-MoE engine")
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
    config_path = model_dir / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"LFM2-MoE checkpoint is missing {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("LFM2-MoE config.json must contain one JSON object")
    return SimpleNamespace(raw=raw)


def _runtime_config(
    model_dir: Path,
    resolved: Lfm2MoeConfig,
    **updates,
) -> dict:
    runtime = dict(resolved.bundle_overrides())
    runtime.update(
        {
            "vocab_size": resolved.vocab_size,
            "bos_token_id": resolved.bos_token_id,
            "eos_token_id": resolved.eos_token_id,
            "pad_token_id": resolved.pad_token_id,
        }
    )
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one LFM2-MoE bundle without shared model orchestration."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("lfm2_moe does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("lfm2_moe does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("lfm2_moe does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("lfm2_moe does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("lfm2_moe does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("lfm2_moe supports only task=text_generation")

    model_dir = Path(request.model_dir)
    config = _load_config(model_dir)
    resolved = validate_lfm2_moe_config(config)
    precision = str(request.precision).lower()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("LFM2-MoE supports only fp16 and bf16 builds")
    max_cache_length = _positive_int(
        request.max_sequence_length or resolved.default_cache_length,
        "max_sequence_length",
    )
    if max_cache_length > resolved.max_position_embeddings:
        raise ValueError(
            "max_cache_length exceeds the checkpoint context capacity: "
            f"{max_cache_length} > {resolved.max_position_embeddings}"
        )
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("LFM2-MoE does not support tensor-parallel builds")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("LFM2-MoE does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("LFM2-MoE does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("LFM2-MoE does not support MTP speculative decoding")

    weights = load_lfm2_moe_weights(str(model_dir), config, precision=precision)
    plan = build_lfm2_moe_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        verbose=bool(request.verbose),
        debug_layer_outputs=False,
    )

    runtime_config = _runtime_config(
        model_dir,
        resolved,
        precision=precision,
        max_cache_length=max_cache_length,
        decoder_engine_layout="single",
    )

    writer.set_header(family="lfm2_moe", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", runtime_config)
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
