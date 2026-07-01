# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM tensor-parallel prefill/decode engine builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..config import ModelConfig
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


def build_dual_profile_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = True,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build a rank-local GLM engine with prefill/decode profiles."""
    if quant_ctx is not None:
        raise ValueError("GLM tensor-parallel builds do not support quantization")
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires parallel.mode=tensor_parallel and tp_size > 1"
        )
    weights = shard_standard_decoder_weights(config, weights, parallel)
    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))
    attention_size = weights.get("_attention_size", config.attention_size)
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_ops.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.float16)
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.bfloat16)
    else:
        work_np_dtype, work_trt_dtype = (np.float32, trt.float32)
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_shape = (max_cache_length, kv_attention_size)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i), work_trt_dtype, cache_shape
        )
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i), work_trt_dtype, cache_shape
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    def _add_profile(
        opt_sq: int,
        max_sq: int,
        *,
        fixed: bool = False,
    ):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq),
        )
        trt_config.add_optimization_profile(prof)

    _add_profile(opt_prefill_length, max_prefill_length)
    _add_profile(1, 1, fixed=True)
    embedding_table = graph_ops.const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"], work_np_dtype, work_trt_dtype
    )
    kmax = max_cache_length + max_prefill_length
    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_np = graph_ops.make_rope_table_half_dim(
        kmax,
        head_dim,
        config.rope_theta,
        True,
        partial_rotary_factor,
        interleaved=interleaved_rope,
    )
    sin_half_np = graph_ops.make_rope_table_half_dim(
        kmax,
        head_dim,
        config.rope_theta,
        False,
        partial_rotary_factor,
        interleaved=interleaved_rope,
    )
    cos_half_table = graph_ops.const_in_work_dtype(
        network, cos_half_np.shape, cos_half_np, work_np_dtype, work_trt_dtype
    )
    sin_half_table = graph_ops.const_in_work_dtype(
        network, sin_half_np.shape, sin_half_np, work_np_dtype, work_trt_dtype
    )
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    matmul = graph_ops.make_matmul_fn(network, work_np_dtype, None)
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask_work)
    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = graph_ops.add_rms_norm(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            eps_tensor,
            dtype=work_np_dtype,
        )
        q = matmul(normed, hidden, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(network, k, kv_attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(network, v, kv_attention_size, v_bias, dtype=work_np_dtype)
        q = graph_ops.add_apply_rope_native(
            network,
            q,
            num_heads,
            head_dim,
            cos_half_table,
            sin_half_table,
            position_id,
            rotary_embedding_dim,
            True,
            sequence_length=None,
        )
        k = graph_ops.add_apply_rope_native(
            network,
            k,
            num_kv_heads,
            head_dim,
            cos_half_table,
            sin_half_table,
            position_id,
            rotary_embedding_dim,
            True,
            sequence_length=None,
        )
        present_k_outs.append(k)
        present_v_outs.append(v)
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0
        context = graph_ops.add_attention_from_rows(
            network,
            q,
            all_k_cat.get_output(0),
            all_v_cat.get_output(0),
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=None,
            kv_seq=None,
            mask=mask_4d,
            scale=attn_scale,
            tag=f"{prefix}.attn",
        )
        attn_out = matmul(
            context, attention_size, hidden, weights[f"{prefix}.w_o"], f"{prefix}.w_o"
        )
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)
        residual1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = graph_ops.add_rms_norm(
            network,
            residual1.get_output(0),
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            eps_tensor,
            dtype=work_np_dtype,
        )
        mlp_out = graph_ops.add_swiglu_mlp(
            network,
            norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            mlp_size=mlp_size,
            dtype=work_np_dtype,
        )
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
        )
        hidden_state = residual2.get_output(0)
    hidden_state = graph_ops.add_rms_norm(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        eps_tensor,
        dtype=work_np_dtype,
    )
    shape_t = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64
    )
    start_sub = network.add_elementwise(shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)
    size_t = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64
    )
    slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    slicer.set_input(1, start_t)
    slicer.set_input(2, size_t)
    last_hidden = slicer.get_output(0)
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for i in range(num_layers):
        pk = present_k_outs[i]
        pv = present_v_outs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)
    if verbose:
        print(
            "[trtmc build] Building GLM tensor-parallel engine "
            f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
            f"kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, "
            f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
            f"precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Tensor-parallel decoder engine build failed")
    return bytes(plan)
