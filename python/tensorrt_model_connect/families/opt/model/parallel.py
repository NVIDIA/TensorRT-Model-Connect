# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local tensor-parallel OPT graph builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_ops
from .model import (
    const_in_work_dtype as _const_in_work_dtype,
    create_builder_context,
    norm_multi as _norm_multi,
)
from ....parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ....quantization.context import QuantContext


_make_matmul_fn = graph_ops.make_matmul_fn


def _validate_tp_quantization(quant_ctx: "QuantContext | None") -> None:
    if quant_ctx is None:
        return
    format_name = getattr(getattr(quant_ctx, "profile", None), "format", None)
    if getattr(format_name, "name", None) != "fp8":
        raise ValueError("Tensor-parallel decoder quantization currently supports fp8 only")


def _relu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1 = graph_ops.add_bias_sum(
        network, fc1, mlp_size, weights[f"{prefix}.fc1_bias"], dtype=work_np_dtype
    )
    activated = network.add_activation(fc1, trt.ActivationType.RELU).get_output(0)
    fc2 = matmul(activated, mlp_size, hidden, weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config, weights: "WeightDict") -> None:
    if config.model_type.lower() != "opt":
        raise ValueError(f"Expected model_type='opt', got {config.model_type!r}")
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError("OPT requires num_key_value_heads == num_attention_heads")
    for name in ("embedding", "position_embedding", "final_norm", "w_out"):
        if name not in weights:
            raise ValueError(f"Missing required OPT weight: {name}")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_tp_decoder_engine(
    config,
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    parallel_config=None,
) -> bytes:
    """Build one rank-local OPT engine with prefill/decode profiles."""
    _supports_config(config, weights)
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires parallel.mode=tensor_parallel and tp_size > 1"
        )
    _validate_tp_quantization(quant_ctx)
    weights = shard_standard_decoder_weights(config, weights, parallel)
    if max_prefill_length is None:
        max_prefill_length = max_cache_length
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))
    multi_bucket_decode = bool(dynamic_kv_profile_rows)
    if multi_bucket_decode:
        decode_buckets: list[int] = []
        seen = set()
        for raw in dynamic_kv_profile_rows or []:
            clamped = max(1, min(int(raw), max_cache_length))
            if clamped not in seen:
                seen.add(clamped)
                decode_buckets.append(clamped)
        decode_buckets.sort()
        if not decode_buckets:
            decode_buckets = [max_cache_length]
            multi_bucket_decode = False
    attention_size = weights.get("_attention_size", config.attention_size)
    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = attention_size
    builder_context = create_builder_context(verbose=verbose, workspace_bytes=1 << 30)
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.float16)
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.bfloat16)
    else:
        work_np_dtype, work_trt_dtype = (np.float32, trt.float32)
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    cache_shape: tuple[int, int]
    if multi_bucket_decode:
        cache_shape = (-1, kv_attention_size)
    else:
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
        cache_rows_min: int | None = None,
        cache_rows_opt: int | None = None,
        cache_rows_max: int | None = None,
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
        if multi_bucket_decode:
            cmn = cache_rows_min if cache_rows_min is not None else 1
            cop = cache_rows_opt if cache_rows_opt is not None else max_cache_length
            cmx = cache_rows_max if cache_rows_max is not None else max_cache_length
            for i in range(num_layers):
                for name in (
                    graph_ops.layer_tensor_name("cache_k", i),
                    graph_ops.layer_tensor_name("cache_v", i),
                ):
                    prof.set_shape(
                        name,
                        (cmn, kv_attention_size),
                        (cop, kv_attention_size),
                        (cmx, kv_attention_size),
                    )
        trt_config.add_optimization_profile(prof)

    _add_profile(
        opt_prefill_length,
        max_prefill_length,
        fixed=False,
        cache_rows_min=1,
        cache_rows_opt=max_cache_length,
        cache_rows_max=max_cache_length,
    )
    if multi_bucket_decode:
        for bucket in decode_buckets:
            _add_profile(
                1, 1, fixed=True, cache_rows_min=1, cache_rows_opt=bucket, cache_rows_max=bucket
            )
    else:
        _add_profile(1, 1, fixed=True)
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"], work_np_dtype, work_trt_dtype
    )
    pos_embed_np = weights["position_embedding"]
    position_embed_table = _const_in_work_dtype(
        network, pos_embed_np.shape, pos_embed_np, work_np_dtype, work_trt_dtype
    )
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    attn_scale = 1.0 / np.sqrt(head_dim)
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    pos_gather = network.add_gather(position_embed_table, position_id, 0)
    pos_add = network.add_elementwise(
        hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
    )
    hidden_state = pos_add.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask_work)
    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        normed = _norm_multi(
            network,
            hidden_state,
            hidden,
            weights[f"{prefix}.input_norm"],
            weights[f"{prefix}.input_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        q = matmul(normed, hidden, attention_size, weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size, weights[f"{prefix}.w_v"], f"{prefix}.w_v")
        q = graph_ops.add_bias_sum(
            network, q, attention_size, weights[f"{prefix}.q_bias"], dtype=work_np_dtype
        )
        k = graph_ops.add_bias_sum(
            network, k, kv_attention_size, weights[f"{prefix}.k_bias"], dtype=work_np_dtype
        )
        v = graph_ops.add_bias_sum(
            network, v, kv_attention_size, weights[f"{prefix}.v_bias"], dtype=work_np_dtype
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
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden, weights[f"{prefix}.o_bias"], dtype=work_np_dtype
        )
        residual1 = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _norm_multi(
            network,
            residual1.get_output(0),
            hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights[f"{prefix}.post_attn_norm_beta"],
            eps_tensor,
            work_np_dtype,
        )
        mlp_out = _relu_mlp(
            network,
            norm2,
            matmul=matmul,
            weights=weights,
            prefix=prefix,
            hidden=hidden,
            mlp_size=mlp_size,
            work_np_dtype=work_np_dtype,
        )
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)
        mlp_out = graph_ops.add_bias_sum(
            network, mlp_out, hidden, weights[f"{prefix}.fc2_bias"], dtype=work_np_dtype
        )
        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
        )
        hidden_state = residual2.get_output(0)
    hidden_state = _norm_multi(
        network,
        hidden_state,
        hidden,
        weights["final_norm"],
        weights["final_norm_beta"],
        eps_tensor,
        work_np_dtype,
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
    out_vocab = weights["w_out"].shape[1]
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
            f"[trtmc-build] Building tensor-parallel OPT engine (layers={num_layers}, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Tensor-parallel decoder engine build failed")
    return bytes(plan)
