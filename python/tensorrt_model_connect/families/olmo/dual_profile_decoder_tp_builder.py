# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dual-profile decoder engine builder.

Produces one rank-local TensorRT engine that handles both prefill
(multi-token) and decode (single-token) phases by switching between two
optimization profiles at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt=opt_prefill_length, max=max_prefill_length].
    TensorRT picks batched MHA kernels (e.g. ``_gemm_mha_v2``) at opt Sq.
  * Profile 1 (decode): Sq fixed to 1. TensorRT picks the GEMV fast-path
    (``_gemv_mha_v1``).

The build path intentionally duplicates most of the single-device
dual-profile graph so tensor-parallel shape decisions stay explicit here:
Q/K/V and MLP expansion projections are column-sharded, output/down
projections are row-sharded, and each row-parallel join inserts a TensorRT
distributed ALL_REDUCE collective.

Scope: covers the same architectural variants the legacy
``standard_decoder_builder`` supports - RMSNorm or LayerNorm; SwiGLU or
GeluFC MLP; RoPE (full / partial / interleaved), learned absolute, or
ALiBi position; sequential or parallel residual; optional q/k_norm,
QKV/output/MLP biases, and a Bloom-style embedding LayerNorm. Quantized
builds (fp8 / int8 ``quant_ctx``) thread Q/DQ insertion through every
projection matmul via ``QuantContext.maybe_quantized_matmul``. Per-layer
debug outputs, hidden-state outputs, and the VL ``embed_input`` path stay
on ``standard_decoder_builder`` for now and are dispatched there from
inside ``build_standard_decoder_engine``.

Tensor contract (matches the C++ runtime KvCache naming):
  Inputs (dynamic shapes - Sq varies by profile)
    token_id        int32   (-1,)
    position_id     int32   (-1,)
    attention_mask  float32 (-1, -1)                 # (Sq, max_cache + Sq)
    cache_k_i       fp16/f32 (max_cache, kv_size)    # static
    cache_v_i       fp16/f32 (max_cache, kv_size)    # static
  Outputs
    logits          float32 (1, vocab)               # last-row sliced inside the engine
    present_k_i     fp16/f32 (-1, kv_size)           # (Sq, kv_size)
    present_v_i     fp16/f32 (-1, kv_size)           # (Sq, kv_size)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from . import graph_blocks
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


def _const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in work_np_dtype storage and cast it to work_trt_dtype.

    Needed for bf16 builds: the dual-profile builder stores bf16 weights
    on disk as fp16 (work_np_dtype = np.float16), but the runtime tensor
    must be bfloat16 to match the rest of the graph. ``add_constant``
    alone produces an fp16 constant - we need an explicit cast to
    bfloat16 so layers like IRotaryEmbeddingLayer (which require all
    inputs to share a dtype) accept it. fp16 / fp32 builds are no-ops
    because work_np_dtype maps directly to work_trt_dtype.
    """
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _make_matmul_fn(
    network: trt.INetworkDefinition,
    dtype: np.dtype,
    quant_ctx: "QuantContext | None",
):
    """Mirror of ``graph_blocks._make_matmul_fn`` for the dual-profile path.

    Returns a callable ``(lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor``
    that routes through ``QuantContext.maybe_quantized_matmul`` when present
    and falls back to a plain ``add_matmul_rhs_constant`` otherwise. The
    ``weight_name`` is the dotted weight key (e.g. ``layer.0.w_q``) used by
    the quantization profile to look up scales and the per-layer exclude
    pattern.
    """
    if quant_ctx is None:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype
            )

        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
        )

    return matmul


def _validate_tp_quantization(quant_ctx: "QuantContext | None") -> None:
    if quant_ctx is None:
        return
    format_name = getattr(getattr(quant_ctx, "profile", None), "format", None)
    if getattr(format_name, "name", None) != "fp8":
        raise ValueError("Tensor-parallel decoder quantization currently supports fp8 only")


def _norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return graph_ops.add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(network, inp, hidden, gamma, eps_tensor, dtype=dtype)


# ---------------------------------------------------------------------------
# MLP helpers.
# ---------------------------------------------------------------------------


def _swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(
        gated.get_output(0), mlp_size, hidden, weights[f"{prefix}.w_down"], f"{prefix}.w_down"
    )
    return mlp_out


def _gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    matmul,
    weights: "WeightDict",
    prefix: str,
    hidden: int,
    mlp_size: int,
    activation: str,
    work_np_dtype: np.dtype,
) -> trt.ITensor:
    fc1 = matmul(inp, hidden, mlp_size, weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = graph_ops.add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden, weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden, fc2_bias, dtype=work_np_dtype)
    return fc2


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the dual-profile builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"dual_profile_decoder_builder does not support model_type={model_type!r}"
        )
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    parallel_config=None,
) -> bytes:
    """Build a rank-local tensor-parallel engine with prefill/decode profiles.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    The current TP implementation supports tp_size 2, 4, and 8. The caller
    invokes this builder once per rank and packages the rank-local plans into
    one bundle.

    When ``dynamic_kv_profile_rows`` is provided, the engine carries one
    prefill profile (profile 0) followed by N decode profiles (profile 1..N),
    one per bucket - letting TriAttention pick the smallest active KV cache
    bucket at runtime while still benefitting from batched prefill on the
    prompt. The cache_k/cache_v inputs are declared dynamic so each profile
    can constrain their row count independently.
    """
    _supports_config(config, weights)
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires parallel.mode=tensor_parallel and tp_size > 1"
        )
    _validate_tp_quantization(quant_ctx)
    if mlp_type != "swiglu":
        raise NotImplementedError(
            "Tensor-parallel decoder builds currently support SwiGLU MLPs only"
        )
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
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
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
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    if position_type == "rope":
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
        cos_half_table = _const_in_work_dtype(
            network, cos_half_np.shape, cos_half_np, work_np_dtype, work_trt_dtype
        )
        sin_half_table = _const_in_work_dtype(
            network, sin_half_np.shape, sin_half_np, work_np_dtype, work_trt_dtype
        )
    position_embed_table: trt.ITensor | None = None
    if position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = _const_in_work_dtype(
            network, pos_embed_np.shape, pos_embed_np, work_np_dtype, work_trt_dtype
        )
    alibi_slopes_tensor: trt.ITensor | None = None
    alibi_cache_positions_fp32: trt.ITensor | None = None
    if position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads)
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1), alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32
        )
        alibi_cache_positions_fp32 = graph_ops.add_constant(
            network,
            (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32),
            dtype=np.float32,
        )
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([[config.rms_norm_eps]], dtype=np.float32), dtype=np.float32
    )
    eps_tensor_per_head = graph_ops.add_constant(
        network, (1, 1, 1), np.array([[[config.rms_norm_eps]]], dtype=np.float32), dtype=np.float32
    )
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1)) if scale_attn_weights else 1.0
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)
    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
        )
        hidden_state = pos_add.get_output(0)
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get("embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network,
            hidden_state,
            hidden,
            embed_norm,
            embed_norm_beta,
            eps_tensor,
            "layernorm",
            work_np_dtype,
        )
    if position_type == "alibi":
        mask_4d = graph_ops.add_alibi_mask_4d(
            network,
            attention_mask_work,
            position_id,
            alibi_slopes_tensor,
            alibi_cache_positions_fp32,
            num_heads,
            target_dtype=work_trt_dtype,
        )
    else:
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
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor,
            "layernorm",
            work_np_dtype,
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
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm_per_head(
                network,
                q,
                num_heads,
                head_dim,
                q_norm,
                eps_tensor_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm_per_head(
                network,
                k,
                num_kv_heads,
                head_dim,
                k_norm,
                eps_tensor_per_head,
                dtype=work_np_dtype,
                sequence_length=None,
            )
        if position_type == "rope":
            q = graph_ops.add_apply_rope_native(
                network,
                q,
                num_heads,
                head_dim,
                cos_half_table,
                sin_half_table,
                position_id,
                rotary_embedding_dim,
                interleaved_rope,
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
                interleaved_rope,
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
            causal=False,
            mask=mask_4d,
            scale=attn_scale,
            tag=f"{prefix}.attn",
        )
        attn_out = matmul(
            context, attention_size, hidden, weights[f"{prefix}.w_o"], f"{prefix}.w_o"
        )
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype
            )
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network,
                    hidden_state,
                    hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor,
                    "layernorm",
                    work_np_dtype,
                )
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM
            )
            norm2 = _norm_multi(
                network,
                residual1.get_output(0),
                hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor,
                "layernorm",
                work_np_dtype,
            )
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network,
                norm2,
                matmul=matmul,
                weights=weights,
                prefix=prefix,
                hidden=hidden,
                mlp_size=mlp_size,
                activation=activation,
                work_np_dtype=work_np_dtype,
            )
        else:
            mlp_out = _swiglu_mlp(
                network,
                norm2,
                matmul=matmul,
                weights=weights,
                prefix=prefix,
                hidden=hidden,
                mlp_size=mlp_size,
            )
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)
        if parallel_residual:
            sum_attn = network.add_elementwise(hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
            )
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM
            )
        hidden_state = residual2.get_output(0)
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network,
            hidden_state,
            hidden,
            final_norm,
            weights.get("final_norm_beta"),
            eps_tensor,
            "layernorm",
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
    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype
    )
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, zero_bias, dtype=work_np_dtype)
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
            f"[trtmc-build] Building tensor-parallel decoder engine (layers={num_layers}, hidden={hidden}, attn={attention_size}, kv={kv_attention_size}, mlp={mlp_size}, cache={max_cache_length}, opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, norm={'layernorm'}, mlp_type={mlp_type}, pos={position_type}, precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Tensor-parallel decoder engine build failed")
    return bytes(plan)
