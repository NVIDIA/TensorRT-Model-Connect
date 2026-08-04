# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one LocateAnything native-KV TensorRT engine for a TP rank.

Prefill and decode are separate single-profile engines. Each rank owns only
its sharded KV heads, while TensorRT ``IKVCacheUpdate`` mutates the caller-
provided full-capacity BF16 cache and non-decomposable ``IAttention`` consumes
only ``key_value_lengths`` active rows.
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

_NATIVE_PREFILL_CHUNK_TOKENS = 32768

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

    Needed for bf16 builds: the builder stores bf16 weights
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
    """Mirror of ``graph_blocks._make_matmul_fn`` for the TP path.

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
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul

    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        return quant_ctx.maybe_quantized_matmul(
            network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
            dtype=dtype)
    return matmul


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
        return graph_ops.add_layer_norm(
            network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(
        network, inp, hidden, gamma, eps_tensor, dtype=dtype)


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
    gate = matmul(inp, hidden, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size,
                weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    mlp_out = matmul(gated.get_output(0), mlp_size, hidden,
                     weights[f"{prefix}.w_down"], f"{prefix}.w_down")
    return mlp_out


# ---------------------------------------------------------------------------
# Config guard.
# ---------------------------------------------------------------------------


def _supports_config(config: "ModelConfig", weights: "WeightDict") -> None:
    """Reject configs the TP builder cannot handle."""
    model_type = getattr(config, "model_type", "").lower()
    if "moe" in model_type or "mamba" in model_type or "rwkv" in model_type:
        raise NotImplementedError(
            f"LocateAnything TP native-KV builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_qwen_vl_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp16",
    opt_prefill_length: int = 64,
    max_prefill_length: int | None = None,
    quant_ctx: "QuantContext | None" = None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    interleaved_rope: bool = False,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = False,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    parallel_config=None,
    profile_mode: str = "decode",
) -> bytes:
    """Build one rank-local prefill or decode engine.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    The caller packages a prefill and decode plan for every TP rank. Legacy
    KV buckets and multi-profile engines are rejected.
    """
    _supports_config(config, weights)
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires "
            "parallel.mode=tensor_parallel and tp_size > 1")
    if profile_mode not in ("prefill", "decode"):
        raise ValueError(
            "LocateAnything TP native KV requires profile_mode='prefill' or 'decode'")
    if precision != "bf16":
        raise ValueError("LocateAnything TP native KV cache requires BF16")
    if quant_ctx is not None or dynamic_kv_profile_rows:
        raise ValueError(
            "LocateAnything TP native KV cache does not support quantization or KV buckets")
    if position_type != "rope":
        raise NotImplementedError(
            "LocateAnything TP native KV cache requires rotary position embeddings")
    if norm_type != "rmsnorm" or mlp_type != "swiglu" or activation != "silu":
        raise NotImplementedError(
            "LocateAnything TP native KV requires Qwen2 RMSNorm/SwiGLU/SILU")
    if parallel_residual or interleaved_rope or partial_rotary_factor != 1.0:
        raise NotImplementedError(
            "LocateAnything TP native KV requires sequential residual and full non-interleaved RoPE")
    if not scale_attn_weights:
        raise NotImplementedError(
            "LocateAnything TP native KV requires scaled dot-product attention")
    weights = shard_standard_decoder_weights(config, weights, parallel)

    if max_prefill_length is None:
        max_prefill_length = min(
            max_cache_length, _NATIVE_PREFILL_CHUNK_TOKENS)
    max_prefill_length = max(1, min(max_prefill_length, max_cache_length))
    opt_prefill_length = max(1, min(opt_prefill_length, max_prefill_length))

    mlp_size = weights.get("_mlp_size", config.intermediate_size)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = config.head_dim
    attention_size = num_heads * head_dim
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    native_active_rope_inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim, config.rope_theta, partial_rotary_factor)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    multi_device_preview = getattr(
        trt.PreviewFeature, "MULTIDEVICE_RUNTIME_10_16", None)
    if multi_device_preview is not None:
        trt_config.set_preview_feature(multi_device_preview, True)

    work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input(
        "key_value_lengths", trt.int32, (1,))
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (-1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (-1, 1))

    cache_shape = (1, num_kv_heads, max_cache_length, head_dim)
    cache_k_inputs: list[trt.ITensor] = []
    cache_v_inputs: list[trt.ITensor] = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, cache_shape)
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, cache_shape)
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        if embed_input:
            prof.set_shape(
                "input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
            prof.set_shape(
                "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        trt_config.add_optimization_profile(prof)

    if profile_mode == "prefill":
        _add_profile(opt_prefill_length, max_prefill_length, fixed=False)
    else:
        _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    graph_ops.validate_native_rope_dim(rotary_embedding_dim)
    cos_half_table, sin_half_table = graph_ops.add_active_rope_cache(
        network, position_id, native_active_rope_inv_freq, work_trt_dtype)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32)
    eps_tensor_per_head = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([[[config.rms_norm_eps]]], dtype=np.float32),
        dtype=np.float32)

    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if embed_input and input_embed_tensor is not None and use_input_embed_tensor is not None:
        token_embed = hidden_state
        if token_embed.dtype != work_trt_dtype:
            token_embed = network.add_cast(token_embed, work_trt_dtype).get_output(0)
        input_embed = input_embed_tensor
        if input_embed.dtype != work_trt_dtype:
            input_embed = network.add_cast(input_embed, work_trt_dtype).get_output(0)
        embed_selector = use_input_embed_tensor
        if embed_selector.dtype != work_trt_dtype:
            embed_selector = network.add_cast(embed_selector, work_trt_dtype).get_output(0)
        one_const = _const_in_work_dtype(
            network, (1, 1), np.array([[1.0]], dtype=work_np_dtype),
            work_np_dtype, work_trt_dtype)
        inv_flag = network.add_elementwise(
            one_const, embed_selector, trt.ElementWiseOperation.SUB)
        tok_part = network.add_elementwise(
            inv_flag.get_output(0), token_embed,
            trt.ElementWiseOperation.PROD)
        embed_part = network.add_elementwise(
            embed_selector, input_embed,
            trt.ElementWiseOperation.PROD)
        hidden_state_sum = network.add_elementwise(
            tok_part.get_output(0), embed_part.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = hidden_state_sum.get_output(0)

    # Make sure the main hidden stream is in the requested runtime dtype
    # before entering the layer stack (BF16 mode stores fp16 constants).
    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get(
            "embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            eps_tensor, "layernorm", work_np_dtype)

    present_k_outs: list[trt.ITensor] = []
    present_v_outs: list[trt.ITensor] = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"

        # Pre-attention norm.
        normed = _norm_multi(
            network, hidden_state, hidden,
            weights[f"{prefix}.input_norm"],
            weights.get(f"{prefix}.input_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_v"], f"{prefix}.w_v")

        # Optional QKV biases (Qwen2 / GPT-2 / OPT / Bloom / Falcon / etc.).
        q_bias = weights.get(f"{prefix}.q_bias")
        if q_bias is not None:
            q = graph_ops.add_bias_sum(
                network, q, attention_size, q_bias, dtype=work_np_dtype)
        k_bias = weights.get(f"{prefix}.k_bias")
        if k_bias is not None:
            k = graph_ops.add_bias_sum(
                network, k, kv_attention_size, k_bias, dtype=work_np_dtype)
        v_bias = weights.get(f"{prefix}.v_bias")
        if v_bias is not None:
            v = graph_ops.add_bias_sum(
                network, v, kv_attention_size, v_bias, dtype=work_np_dtype)

        # Optional per-head q/k norm (Qwen3).
        q_norm = weights.get(f"{prefix}.q_norm")
        if q_norm is not None:
            q = graph_ops.add_rms_norm_per_head(
                network, q, num_heads, head_dim, q_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            k = graph_ops.add_rms_norm_per_head(
                network, k, num_kv_heads, head_dim, k_norm,
                eps_tensor_per_head, dtype=work_np_dtype,
                sequence_length=None)

        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_half_table, sin_half_table, None,
            rotary_embedding_dim, interleaved_rope,
            sequence_length=None)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_half_table, sin_half_table, None,
            rotary_embedding_dim, interleaved_rope,
            sequence_length=None)

        native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
            network, q, k, v,
            cache_k_inputs[layer_idx], cache_v_inputs[layer_idx],
            cache_write_indices, key_value_lengths,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, q_seq=None, scale=attn_scale,
            tag=f"{prefix}.attn")
        context = native_attention["context"]
        present_k_outs.append(native_attention["present_k"])
        present_v_outs.append(native_attention["present_v"])

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        attn_out = add_all_reduce_sum(network, attn_out, parallel.tp_size)
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        residual1 = network.add_elementwise(
            hidden_state, attn_out, trt.ElementWiseOperation.SUM)
        norm2 = _norm_multi(
            network, residual1.get_output(0), hidden,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)
        mlp_out = _swiglu_mlp(
            network, norm2,
            matmul=matmul, weights=weights, prefix=prefix,
            hidden=hidden, mlp_size=mlp_size)
        mlp_out = add_all_reduce_sum(network, mlp_out, parallel.tp_size)

        residual2 = network.add_elementwise(
            residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"),
            eps_tensor, norm_type, work_np_dtype)

    # Only the LAST prompt token's logits matter for the next-token sample,
    # so slice hidden_state from (Sq, hidden) to (1, hidden) before the LM
    # head. This keeps the output contract identical to the single-token
    # engine (logits shape = (1, vocab)) under both profiles and avoids
    # computing (Sq - 1) redundant vocab-sized matmul rows during prefill.
    shape_t = network.add_shape(hidden_state).get_output(0)  # [2] int64
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    start_sub = network.add_elementwise(
        shape_t, one_hidden, trt.ElementWiseOperation.SUB)
    start_t = start_sub.get_output(0)  # [Sq - 1, 0]
    size_t = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    slicer = network.add_slice(hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    slicer.set_input(1, start_t)
    slicer.set_input(2, size_t)
    last_hidden = slicer.get_output(0)

    out_vocab = (weights["w_out"].shape[1]
                 if isinstance(weights["w_out"], np.ndarray) else vocab)
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"],
        dtype=work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

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
        print(f"[trtmc-build] Building tensor-parallel {profile_mode} engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"kv={kv_attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
              f"precision={precision}, tp={parallel.tp_size}, rank={parallel.rank}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Tensor-parallel decoder engine build failed")
    return bytes(plan)
