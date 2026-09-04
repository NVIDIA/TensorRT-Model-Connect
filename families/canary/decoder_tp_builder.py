# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Canary decoder builder.

This intentionally mirrors the single-device Canary decoder graph in
``plugin.py`` while making the rank-local TP choices explicit:

* self-attention and cross-attention Q/K/V projections are column-sharded,
* attention output and MLP down projections are row-sharded,
* row-parallel joins use TensorRT distributed ALL_REDUCE,
* embeddings, norms, encoder output inputs, and LM head stay replicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import sys

import numpy as np
import tensorrt as trt

from . import graph_ops
from .batching import CANARY_MAX_DECODER_LANES
from .parallel import (
    add_all_reduce_sum,
    normalize_parallel_config,
)


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from .parallel import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_canary_tp(
    weights: "WeightDict",
    *,
    hidden: int,
    num_heads: int,
    ffn_dim: int,
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Canary tensor-parallel engine build requires a concrete rank")
    tp = parallel.tp_size
    if hidden % tp != 0:
        raise ValueError(
            f"Canary tensor parallel requires hidden size divisible by tp_size "
            f"({hidden} vs {tp})")
    if num_heads % tp != 0:
        raise ValueError(
            f"Canary tensor parallel requires decoder_attention_heads divisible "
            f"by tp_size ({num_heads} vs {tp})")
    if ffn_dim % tp != 0:
        raise ValueError(
            f"Canary tensor parallel requires decoder_ffn_dim divisible by tp_size "
            f"({ffn_dim} vs {tp})")
    dec_layers = int(weights["_dec_layers"])
    for i in range(dec_layers):
        prefix = f"layer.{i}"
        for key in (f"{prefix}.w_q", f"{prefix}.w_k", f"{prefix}.w_v",
                    f"{prefix}.xw_q", f"{prefix}.xw_k", f"{prefix}.xw_v"):
            if weights[key].shape[-1] % tp != 0:
                raise ValueError(f"{key} output dim is not divisible by tp_size={tp}")
        for key in (f"{prefix}.w_o", f"{prefix}.xw_o", f"{prefix}.w_fc2"):
            if weights[key].shape[0] % tp != 0:
                raise ValueError(f"{key} input dim is not divisible by tp_size={tp}")
        if weights[f"{prefix}.w_fc1"].shape[-1] % tp != 0:
            raise ValueError(f"{prefix}.w_fc1 output dim is not divisible by tp_size={tp}")


def shard_canary_decoder_weights(
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local decoder weights for the Canary TP builder."""
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((
            ".w_q", ".w_k", ".w_v",
            ".xw_q", ".xw_k", ".xw_v",
            ".w_fc1",
        )):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((
            ".q_bias", ".k_bias", ".v_bias",
            ".xb_q", ".xb_k", ".xb_v",
            ".fc1_bias",
        )):
            out[key] = _slice_first_dim(value, rank, tp)
        elif key.endswith((".w_o", ".xw_o", ".w_fc2")):
            out[key] = _slice_first_dim(value, rank, tp)
        else:
            out[key] = value

    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _add_linear_with_bias(
    network,
    lhs,
    in_dim: int,
    out_dim: int,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    dtype: np.dtype,
):
    out = graph_ops.add_matmul_rhs_constant(
        network, lhs, in_dim, out_dim, weight, dtype=dtype)
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias, dtype=dtype)
    return out


def _add_batched_attention_manual(
    network,
    q,
    k,
    v,
    *,
    num_heads: int,
    head_dim: int,
    kv_seq: int,
    mask=None,
    fp32_accumulation: bool = False,
):
    """Primitive scaled-dot-product attention for TP head counts.

    TensorRT native IAttention can be tactic-sensitive for some rank-local
    head counts, so TP uses explicit batched matmul + softmax + matmul for
    build stability.
    """
    output_dtype = q.dtype
    q_layer = network.add_shuffle(q)
    q_layer.reshape_dims = (-1, num_heads, 1, head_dim)
    q_4d = q_layer.get_output(0)
    k_rows = network.add_shuffle(k)
    k_rows.reshape_dims = (-1, kv_seq, num_heads, head_dim)
    k_layer = network.add_shuffle(k_rows.get_output(0))
    k_layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    k_4d = k_layer.get_output(0)
    v_rows = network.add_shuffle(v)
    v_rows.reshape_dims = (-1, kv_seq, num_heads, head_dim)
    v_layer = network.add_shuffle(v_rows.get_output(0))
    v_layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    v_4d = v_layer.get_output(0)
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    scale = float(1.0 / np.sqrt(max(head_dim, 1)))
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = graph_ops.add_constant(
        network, (1, 1, 1, 1), np.array([[[[scale]]]], dtype=scale_np_dtype),
        dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)

    scores = network.add_matrix_multiply(
        q_4d, trt.MatrixOperation.NONE,
        k_4d, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    if mask is not None:
        scores = network.add_elementwise(
            scores, mask, trt.ElementWiseOperation.SUM).get_output(0)
    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    context = network.add_matrix_multiply(
        probs.get_output(0), trt.MatrixOperation.NONE,
        v_4d, trt.MatrixOperation.NONE).get_output(0)
    if context.dtype != output_dtype:
        context = network.add_cast(context, output_dtype).get_output(0)
    rows = network.add_shuffle(context)
    rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    rows.reshape_dims = (-1, num_heads * head_dim)
    return rows.get_output(0)


def _add_canary_tp_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    cross_attention_mask,
    eps_tensor,
    weights,
    prefix: str,
    hidden_size: int,
    local_attention_size: int,
    local_heads: int,
    head_dim: int,
    local_ffn_dim: int,
    max_cache_length: int,
    max_source_positions: int,
    dtype: np.dtype,
    work_trt_dtype,
    tp_size: int,
):
    attention_window = max_cache_length + 1

    # Self-attention. Q/K/V are column-parallel, so each rank owns a subset of
    # heads. The output projection is row-parallel and all-reduced before bias.
    normed = graph_ops.add_layer_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"],
        weights[f"{prefix}.input_norm_b"], eps_tensor, dtype=dtype)
    q = _add_linear_with_bias(
        network, normed, hidden_size, local_attention_size,
        weights[f"{prefix}.w_q"], weights[f"{prefix}.q_bias"], dtype=dtype)
    k = _add_linear_with_bias(
        network, normed, hidden_size, local_attention_size,
        weights[f"{prefix}.w_k"], weights[f"{prefix}.k_bias"], dtype=dtype)
    v = _add_linear_with_bias(
        network, normed, hidden_size, local_attention_size,
        weights[f"{prefix}.w_v"], weights[f"{prefix}.v_bias"], dtype=dtype)
    present_k, present_v = k, v

    kr = network.add_shuffle(k)
    kr.reshape_dims = (-1, 1, local_attention_size)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (-1, 1, local_attention_size)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 1
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 1

    mask_4d = graph_ops.add_3d_mask_to_4d(network, attention_mask)
    context = _add_batched_attention_manual(
        network, q, ak.get_output(0), av.get_output(0),
        num_heads=local_heads, head_dim=head_dim,
        kv_seq=attention_window,
        mask=mask_4d)
    sa = graph_ops.add_matmul_rhs_constant(
        network, context, local_attention_size, hidden_size,
        weights[f"{prefix}.w_o"], dtype=dtype)
    sa = add_all_reduce_sum(network, sa, tp_size)
    sa = graph_ops.add_bias_sum(
        network, sa, hidden_size, weights[f"{prefix}.o_bias"], dtype=dtype)
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)

    # Cross-attention. Encoder output is replicated; each rank projects K/V
    # into its local heads, then all-reduces the row-parallel output join.
    cn = graph_ops.add_layer_norm(
        network, psa, hidden_size, weights[f"{prefix}.xattn_norm"],
        weights[f"{prefix}.xattn_norm_b"], eps_tensor, dtype=dtype)
    cq = _add_linear_with_bias(
        network, cn, hidden_size, local_attention_size,
        weights[f"{prefix}.xw_q"], weights[f"{prefix}.xb_q"], dtype=dtype)

    cross_k_typed = cross_k
    cross_v_typed = cross_v
    if work_trt_dtype != trt.float32:
        cross_k_typed = network.add_cast(cross_k, work_trt_dtype).get_output(0)
        cross_v_typed = network.add_cast(cross_v, work_trt_dtype).get_output(0)

    ck_proj = _add_linear_with_bias(
        network, cross_k_typed, hidden_size, local_attention_size,
        weights[f"{prefix}.xw_k"], weights[f"{prefix}.xb_k"], dtype=dtype)
    cv_proj = _add_linear_with_bias(
        network, cross_v_typed, hidden_size, local_attention_size,
        weights[f"{prefix}.xw_v"], weights[f"{prefix}.xb_v"], dtype=dtype)
    cross_mask_4d = graph_ops.add_3d_mask_to_4d(
        network, cross_attention_mask)
    ccf = _add_batched_attention_manual(
        network, cq, ck_proj, cv_proj,
        num_heads=local_heads, head_dim=head_dim,
        kv_seq=max_source_positions,
        mask=cross_mask_4d,
        fp32_accumulation=True)
    ca = graph_ops.add_matmul_rhs_constant(
        network, ccf, local_attention_size, hidden_size,
        weights[f"{prefix}.xw_o"], dtype=dtype)
    ca = add_all_reduce_sum(network, ca, tp_size)
    ca = graph_ops.add_bias_sum(
        network, ca, hidden_size, weights[f"{prefix}.xb_o"], dtype=dtype)
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)

    # ReLU MLP. FC1 is column-parallel; FC2 is row-parallel and all-reduced
    # before adding the full FC2 bias.
    fn = graph_ops.add_layer_norm(
        network, pca, hidden_size, weights[f"{prefix}.ffn_norm"],
        weights[f"{prefix}.ffn_norm_b"], eps_tensor, dtype=dtype)
    fc1 = _add_linear_with_bias(
        network, fn, hidden_size, local_ffn_dim,
        weights[f"{prefix}.w_fc1"], weights[f"{prefix}.fc1_bias"], dtype=dtype)
    act = graph_ops.add_activation(network, fc1, "relu", dtype=dtype)
    fc2 = graph_ops.add_matmul_rhs_constant(
        network, act, local_ffn_dim, hidden_size,
        weights[f"{prefix}.w_fc2"], dtype=dtype)
    fc2 = add_all_reduce_sum(network, fc2, tp_size)
    mlp = graph_ops.add_bias_sum(
        network, fc2, hidden_size, weights[f"{prefix}.fc2_bias"], dtype=dtype)
    out = network.add_elementwise(pca, mlp, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def build_canary_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_canary_tp_decoder_engine requires tensor_parallel mode "
            "with tp_size > 1")
    dec_layers = int(weights["_dec_layers"])
    dec_heads = int(weights["_dec_heads"])
    dec_ffn = int(weights["_dec_ffn"])
    max_source_positions = int(weights["_enc_seq"])
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    head_dim = hidden // dec_heads
    tp = parallel.tp_size

    _validate_canary_tp(
        weights, hidden=hidden, num_heads=dec_heads, ffn_dim=dec_ffn,
        parallel=parallel)
    rank_weights = shard_canary_decoder_weights(weights, parallel=parallel)
    local_attention_size = hidden // tp
    local_heads = dec_heads // tp
    local_ffn_dim = dec_ffn // tp
    attention_window = max_cache_length + 1

    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (-1, 1, attention_window))
    cross_attention_mask = network.add_input(
        "cross_attention_mask", trt.float32,
        (-1, 1, max_source_positions))

    cache_k_inputs, cache_v_inputs = [], []
    for i in range(dec_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, (-1, max_cache_length, local_attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, (-1, max_cache_length, local_attention_size)))

    cross_k_inputs, cross_v_inputs = [], []
    for i in range(dec_layers):
        cross_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cross_k", i),
            work_trt_dtype, (-1, max_source_positions, hidden)))
        cross_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cross_v", i),
            work_trt_dtype, (-1, max_source_positions, hidden)))

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "token_id", (1,), (16,), (CANARY_MAX_DECODER_LANES,))
    profile.set_shape(
        "position_id", (1,), (16,), (CANARY_MAX_DECODER_LANES,))
    profile.set_shape(
        "attention_mask", (1, 1, attention_window),
        (16, 1, attention_window),
        (CANARY_MAX_DECODER_LANES, 1, attention_window))
    profile.set_shape(
        "cross_attention_mask", (1, 1, max_source_positions),
        (16, 1, max_source_positions),
        (CANARY_MAX_DECODER_LANES, 1, max_source_positions))
    for i in range(dec_layers):
        suffix = f"_{i}"
        profile.set_shape(
            "cache_k" + suffix,
            (1, max_cache_length, local_attention_size),
            (16, max_cache_length, local_attention_size),
            (CANARY_MAX_DECODER_LANES, max_cache_length, local_attention_size))
        profile.set_shape(
            "cache_v" + suffix,
            (1, max_cache_length, local_attention_size),
            (16, max_cache_length, local_attention_size),
            (CANARY_MAX_DECODER_LANES, max_cache_length, local_attention_size))
        profile.set_shape(
            "cross_k" + suffix,
            (1, max_source_positions, hidden),
            (16, max_source_positions, hidden),
            (CANARY_MAX_DECODER_LANES, max_source_positions, hidden))
        profile.set_shape(
            "cross_v" + suffix,
            (1, max_source_positions, hidden),
            (16, max_source_positions, hidden),
            (CANARY_MAX_DECODER_LANES, max_source_positions, hidden))
    trt_config.add_optimization_profile(profile)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["dec_emb"],
        dtype=work_np_dtype)
    pos_embed_np = rank_weights["dec_pos"]
    pos_embedding_table = graph_ops.add_constant(
        network, pos_embed_np.shape, pos_embed_np, dtype=work_np_dtype)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([1e-5], dtype=work_np_dtype),
        dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
        cross_attention_mask = network.add_cast(
            cross_attention_mask, work_trt_dtype).get_output(0)

    hidden_state = network.add_elementwise(
        network.add_gather(embedding_table, token_id, 0).get_output(0),
        network.add_gather(pos_embedding_table, position_id, 0).get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)
    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden, rank_weights["emb_ln"],
        rank_weights["emb_ln_b"], eps_tensor, dtype=work_np_dtype)

    present_k_outputs, present_v_outputs = [], []
    for layer_idx in range(dec_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_canary_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            cross_k=cross_k_inputs[layer_idx],
            cross_v=cross_v_inputs[layer_idx],
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            local_attention_size=local_attention_size,
            local_heads=local_heads,
            head_dim=head_dim,
            local_ffn_dim=local_ffn_dim,
            max_cache_length=max_cache_length,
            max_source_positions=max_source_positions,
            dtype=work_np_dtype,
            work_trt_dtype=work_trt_dtype,
            tp_size=tp,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

    hidden_state = graph_ops.add_layer_norm(
        network, hidden_state, hidden, rank_weights["final_norm"],
        rank_weights["final_norm_b"], eps_tensor, dtype=work_np_dtype)
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_out"],
        dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, rank_weights["out_bias"],
        dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(dec_layers):
        present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
        present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k_outputs[i])
        network.mark_output(present_v_outputs[i])

    if verbose:
        print(
            "[trtmc build] Building Canary TP decoder "
            f"(rank={parallel.rank}/{tp}, {dec_layers}L, h={hidden}, "
            f"local_heads={local_heads}, cache={max_cache_length}, "
            f"lanes=1..{CANARY_MAX_DECODER_LANES}, precision={precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Canary TP decoder engine build failed")
    return bytes(plan)
