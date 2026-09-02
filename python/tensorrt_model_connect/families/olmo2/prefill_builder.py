# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched-prefill TensorRT engine for the OLMo-2 post-norm decoder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_blocks, graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


def _slice_last_row(network, tensor, width: int):
    shape = network.add_shape(tensor).get_output(0)
    row_shape = graph_ops.add_constant(
        network, (2,), np.array([1, width], dtype=np.int64), dtype=np.int64)
    start = network.add_elementwise(
        shape, row_shape, trt_compat.get_trt().ElementWiseOperation.SUB).get_output(0)
    size = graph_ops.add_constant(
        network, (2,), np.array([1, width], dtype=np.int64), dtype=np.int64)
    result = network.add_slice(tensor, start=(0, 0), shape=(0, 0), stride=(1, 1))
    result.set_input(1, start)
    result.set_input(2, size)
    return result.get_output(0)


def build_olmo2_prefill_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    workspace_bytes: int,
) -> bytes:
    """Build one dynamic-Sq profile that consumes a prompt in one enqueue."""
    trt = trt_compat.get_trt()
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported OLMo2 precision: {precision}")

    attention_size = int(weights.get("_attention_size", config.attention_size))
    mlp_size = int(weights.get("_mlp_size", config.intermediate_size))
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    max_prefill_length = max(1, int(max_cache_length))
    opt_prefill_length = min(352, max_prefill_length)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))

    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_shape = (max_cache_length, kv_attention_size)
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            work_trt_dtype,
            cache_shape,
        ))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            work_trt_dtype,
            cache_shape,
        ))

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "token_id", (1,), (opt_prefill_length,), (max_prefill_length,))
    profile.set_shape(
        "position_id", (1,), (opt_prefill_length,), (max_prefill_length,))
    profile.set_shape(
        "attention_mask",
        (1, max_cache_length + 1),
        (opt_prefill_length, max_cache_length + opt_prefill_length),
        (max_prefill_length, max_cache_length + max_prefill_length),
    )
    trt_config.add_optimization_profile(profile)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
    cos_table_np = graph_ops.make_rope_table_half_dim(
        max_cache_length + max_prefill_length,
        head_dim,
        config.rope_theta,
        True,
    )
    sin_table_np = graph_ops.make_rope_table_half_dim(
        max_cache_length + max_prefill_length,
        head_dim,
        config.rope_theta,
        False,
    )
    cos_tensor = graph_ops.add_constant(
        network, cos_table_np.shape, cos_table_np, dtype=work_np_dtype)
    sin_tensor = graph_ops.add_constant(
        network, sin_table_np.shape, sin_table_np, dtype=work_np_dtype)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    mask = attention_mask
    if mask.dtype != work_trt_dtype:
        mask = network.add_cast(mask, work_trt_dtype).get_output(0)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, mask)

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        q = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, attention_size,
            weights[f"{prefix}.w_q"], dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, kv_attention_size,
            weights[f"{prefix}.w_k"], dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, kv_attention_size,
            weights[f"{prefix}.w_v"], dtype=work_np_dtype)

        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm(
                network, q, attention_size, q_norm, eps_tensor,
                dtype=work_np_dtype)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm(
                network, k, kv_attention_size, k_norm, eps_tensor,
                dtype=work_np_dtype)

        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_tensor, sin_tensor, position_id, head_dim,
            sequence_length=None)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_tensor, sin_tensor, position_id, head_dim,
            sequence_length=None)
        present_k_outputs.append(k)
        present_v_outputs.append(v)

        all_k = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k.axis = 0
        all_v = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v.axis = 0
        context = graph_ops.add_attention_from_rows(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=None,
            kv_seq=None,
            mask=mask_4d,
            tag=f"{prefix}.attn",
        )

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context, attention_size, hidden,
            weights[f"{prefix}.w_o"], dtype=work_np_dtype)
        normed_attn = graph_ops.add_rms_norm(
            network, attn_out, hidden,
            weights[f"{prefix}.post_attn_norm"], eps_tensor,
            dtype=work_np_dtype)
        residual1 = network.add_elementwise(
            hidden_state, normed_attn, trt.ElementWiseOperation.SUM).get_output(0)

        mlp_out = graph_blocks.add_swiglu_mlp(
            network,
            residual1,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
        )
        normed_mlp = graph_ops.add_rms_norm(
            network, mlp_out, hidden,
            weights[f"{prefix}.post_ff_norm"], eps_tensor,
            dtype=work_np_dtype)
        hidden_state = network.add_elementwise(
            residual1, normed_mlp, trt.ElementWiseOperation.SUM).get_output(0)

    hidden_state = graph_ops.add_rms_norm(
        network, hidden_state, hidden, weights["final_norm"], eps_tensor,
        dtype=work_np_dtype)
    last_hidden = _slice_last_row(network, hidden_state, hidden)
    out_vocab = (
        int(weights["w_out"].shape[1])
        if isinstance(weights["w_out"], np.ndarray)
        else vocab
    )
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(
        network, logits, out_vocab, np.zeros(out_vocab, dtype=work_np_dtype),
        dtype=work_np_dtype)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx, (present_k, present_v) in enumerate(
        zip(present_k_outputs, present_v_outputs, strict=True)
    ):
        present_k.name = graph_ops.layer_tensor_name("present_k", layer_idx)
        present_v.name = graph_ops.layer_tensor_name("present_v", layer_idx)
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            "[trtmc build] Building OLMo2 batched-prefill engine "
            f"(layers={num_layers}, hidden={hidden}, cache={max_cache_length}, "
            f"opt_prefill={opt_prefill_length}, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("OLMo2 batched-prefill engine build failed")
    return bytes(plan)
