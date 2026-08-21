# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-profile decoder engine builder — single engine, two optimization profiles.

Produces one TensorRT engine that handles both prefill (multi-token) and
decode (single-token) phases by switching between two optimization profiles
at runtime:
  * Profile 0 (prefill): Sq ranges over [1, opt=opt_prefill_length, max=max_prefill_length].
    TensorRT picks batched MHA kernels (e.g. ``_gemm_mha_v2``) at opt Sq.
  * Profile 1 (decode): Sq fixed to 1. TensorRT picks the GEMV fast-path
    (``_gemv_mha_v1``).

Both profiles use the same graph and weights — only the optimization
profile differs, so the engine's weights live once in GPU memory and the
C++ runtime creates two ``IExecutionContext``s (one per profile) that
share the engine.

Scope: covers the same architectural variants the legacy
``standard_decoder_builder`` supports — RMSNorm or LayerNorm; SwiGLU or
GeluFC MLP; RoPE (full / partial / interleaved), learned absolute, or
ALiBi position; sequential or parallel residual; optional q/k_norm,
QKV/output/MLP biases, and a Bloom-style embedding LayerNorm. Quantized
builds (fp8 / int8 ``quant_ctx``) thread Q/DQ insertion through every
projection matmul via ``QuantContext.maybe_quantized_matmul``. Per-layer
debug outputs, hidden-state outputs, and the VL ``embed_input`` path stay
on ``standard_decoder_builder`` for now and are dispatched there from
inside ``build_standard_decoder_engine``.

Tensor contract (matches the C++ runtime KvCache naming):
  Inputs (dynamic shapes — Sq varies by profile)
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
from .utils import (
    const_in_work_dtype as _const_in_work_dtype,
    create_builder_context,
    norm_multi as _norm_multi,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


_make_matmul_fn = graph_blocks.make_matmul_fn


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
    fc1 = matmul(inp, hidden, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{prefix}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=work_np_dtype)
    activated = graph_ops.add_activation(network, fc1, activation, dtype=work_np_dtype)
    fc2 = matmul(activated, mlp_size, hidden,
                 weights[f"{prefix}.w_fc2"], f"{prefix}.w_fc2")
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
            f"dual_profile_decoder_builder does not support model_type={model_type!r}")
    if "embedding" not in weights:
        raise NotImplementedError("missing embedding weight")
    if "final_norm" not in weights:
        raise NotImplementedError("missing final_norm weight")


# ---------------------------------------------------------------------------
# Main builder.
# ---------------------------------------------------------------------------


def build_dual_profile_decoder_engine(
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
    round_rope_inv_freq_to_bf16: bool = False,
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
) -> bytes:
    """Build a prefill/decode-capable dynamic-Sq decoder engine.

    ``norm_type`` / ``mlp_type`` / ``position_type`` / ``activation`` /
    ``partial_rotary_factor`` / ``interleaved_rope`` / ``parallel_residual`` /
    ``scale_attn_weights`` mirror the same parameters on
    ``build_standard_decoder_engine``.

    ``quant_ctx`` (optional) routes every projection matmul through
    ``QuantContext.maybe_quantized_matmul`` for fp8 / int8 Q/DQ insertion;
    when ``None`` the matmuls are plain fp16 / bf16 / fp32.

    ``profile_mode`` controls which optimization profiles are emitted:

    * ``"dual_profile"``: one prefill profile followed by one or more decode
      profiles. When ``dynamic_kv_profile_rows`` is provided, the decode side
      gets one profile per bucket — letting TriAttention pick the smallest
      active KV cache bucket at runtime while still benefitting from batched
      prefill on the prompt.
    * ``"prefill"``: one prefill profile only. This is used by split-engine
      bundles, where decode is served by a separate fixed-Sq=1 engine.

    Dynamic-KV cache bucket profiles are only meaningful in ``dual_profile``
    mode. In either mode, cache_k/cache_v inputs are declared dynamic when
    bucket profiles are requested so each profile can constrain their row count.
    """
    _supports_config(config, weights)
    if profile_mode not in ("dual_profile", "prefill"):
        raise ValueError(
            "profile_mode must be 'dual_profile' or 'prefill', "
            f"got {profile_mode!r}")

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
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    rope_scaling = config.raw.get("rope_scaling") or {}
    raw_mrope_section = (
        rope_scaling.get("mrope_section")
        if isinstance(rope_scaling, dict) else None)
    mrope_section = (
        tuple(int(value) for value in raw_mrope_section)
        if raw_mrope_section is not None else None)
    if mrope_section is not None:
        if position_type != "rope":
            raise ValueError("mRoPE requires position_type='rope'")
        if len(mrope_section) != 3 or any(value <= 0 for value in mrope_section):
            raise ValueError(
                "rope_scaling.mrope_section must contain three positive integers")
        if sum(mrope_section) != rotary_embedding_dim // 2:
            raise ValueError(
                "rope_scaling.mrope_section must sum to half the rotary dimension; "
                f"got {mrope_section} for rotary dimension {rotary_embedding_dim}")

    builder_context = create_builder_context(
        verbose=verbose,
    )
    builder = builder_context.builder
    network = builder_context.network
    trt_config = builder_context.config

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    mrope_position_ids = (
        network.add_input("mrope_position_ids", trt.int32, (3, -1))
        if mrope_section is not None else None)
    attention_mask = network.add_input("attention_mask", trt.float32, (-1, -1))
    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input(
            "input_embed", trt.float32, (-1, hidden))
        use_input_embed_tensor = network.add_input(
            "use_input_embed", trt.float32, (-1, 1))

    cache_shape: tuple[int, int]
    if multi_bucket_decode:
        cache_shape = (-1, kv_attention_size)
    else:
        cache_shape = (max_cache_length, kv_attention_size)
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

    # Cast mask to compute dtype for elementwise broadcast.
    if work_trt_dtype != trt.float32:
        attention_mask_work = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)
    else:
        attention_mask_work = attention_mask

    # Two (or 1+N) optimization profiles — same graph, different Sq / cache.
    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False,
                     cache_rows_min: int | None = None,
                     cache_rows_opt: int | None = None,
                     cache_rows_max: int | None = None):
        prof = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        prof.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        prof.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        if mrope_position_ids is not None:
            prof.set_shape(
                "mrope_position_ids",
                (3, min_sq), (3, opt_sq), (3, max_sq))
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq))
        if embed_input:
            prof.set_shape(
                "input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
            prof.set_shape(
                "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        if multi_bucket_decode:
            cmn = cache_rows_min if cache_rows_min is not None else 1
            cop = cache_rows_opt if cache_rows_opt is not None else max_cache_length
            cmx = cache_rows_max if cache_rows_max is not None else max_cache_length
            for i in range(num_layers):
                for name in (graph_ops.layer_tensor_name("cache_k", i),
                             graph_ops.layer_tensor_name("cache_v", i)):
                    prof.set_shape(
                        name,
                        (cmn, kv_attention_size),
                        (cop, kv_attention_size),
                        (cmx, kv_attention_size))
        trt_config.add_optimization_profile(prof)

    import os as _os_dbg
    if profile_mode == "prefill":
        _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                     cache_rows_min=1, cache_rows_opt=max_cache_length,
                     cache_rows_max=max_cache_length)
    elif _os_dbg.environ.get("TRTMC_DECODE_ONLY_DEBUG") == "1":
        # Diagnostic: build a one-profile engine with dynamic-shape inputs
        # but Sq pinned to 1. Lets us isolate dynamic-shape enqueueV3
        # overhead from per-profile kernel specialisation.
        _add_profile(1, 1, fixed=True)
    else:
        _reverse = _os_dbg.environ.get("TRTMC_REVERSE_PROFILE_ORDER", "0") == "1"
        if _reverse:
            # Decode profile registered first so it commits its preferred
            # weight layout before the prefill profile compiles.
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(1, 1, fixed=True,
                                 cache_rows_min=1, cache_rows_opt=bucket,
                                 cache_rows_max=bucket)
            else:
                _add_profile(1, 1, fixed=True)
            _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                         cache_rows_min=1, cache_rows_opt=max_cache_length,
                         cache_rows_max=max_cache_length)
        else:
            _add_profile(opt_prefill_length, max_prefill_length, fixed=False,
                         cache_rows_min=1, cache_rows_opt=max_cache_length,
                         cache_rows_max=max_cache_length)
            if multi_bucket_decode:
                for bucket in decode_buckets:
                    _add_profile(1, 1, fixed=True,
                                 cache_rows_min=1, cache_rows_opt=bucket,
                                 cache_rows_max=bucket)
            else:
                _add_profile(1, 1, fixed=True)

    # ---- Shared constants ------------------------------------------------
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype)

    # RoPE tables (only when position_type == "rope"). Built for the worst
    # case key length max_cache_length + max_prefill_length, since RoPE is
    # gathered by position_id at runtime. The half-dim tables feed TRT's
    # native IRotaryEmbeddingLayer.
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    if position_type == "rope":
        kmax = max_cache_length + max_prefill_length
        graph_ops.validate_native_rope_dim(rotary_embedding_dim)
        cos_half_np = graph_ops.make_rope_table_half_dim(
            kmax, head_dim, config.rope_theta, True,
            partial_rotary_factor, interleaved=interleaved_rope,
            round_inv_freq_to_bf16=round_rope_inv_freq_to_bf16)
        sin_half_np = graph_ops.make_rope_table_half_dim(
            kmax, head_dim, config.rope_theta, False,
            partial_rotary_factor, interleaved=interleaved_rope,
            round_inv_freq_to_bf16=round_rope_inv_freq_to_bf16)
        if round_rope_inv_freq_to_bf16:
            cos_half_table = graph_ops.add_constant(
                network, cos_half_np.shape, cos_half_np, dtype=np.float32)
            sin_half_table = graph_ops.add_constant(
                network, sin_half_np.shape, sin_half_np, dtype=np.float32)
            if work_trt_dtype != trt.float32:
                cos_half_table = network.add_cast(
                    cos_half_table, work_trt_dtype).get_output(0)
                sin_half_table = network.add_cast(
                    sin_half_table, work_trt_dtype).get_output(0)
        else:
            cos_half_table = _const_in_work_dtype(
                network, cos_half_np.shape, cos_half_np,
                work_np_dtype, work_trt_dtype)
            sin_half_table = _const_in_work_dtype(
                network, sin_half_np.shape, sin_half_np,
                work_np_dtype, work_trt_dtype)

    # Learned position embedding (GPT-2 / OPT / GPT-Neo / XGLM).
    position_embed_table: trt.ITensor | None = None
    if position_type == "learned":
        pos_embed_np = weights["position_embedding"]
        position_embed_table = _const_in_work_dtype(
            network, pos_embed_np.shape, pos_embed_np,
            work_np_dtype, work_trt_dtype)

    # ALiBi slopes + cache-slot positions for multi-row mask augmentation.
    alibi_slopes_tensor: trt.ITensor | None = None
    alibi_cache_positions_fp32: trt.ITensor | None = None
    if position_type == "alibi":
        alibi_slopes_np = graph_ops.compute_alibi_slopes(num_heads)
        # Slopes live as fp32 so the (key_pos - q_pos) math stays in fp32;
        # add_alibi_mask_4d casts the final bias to work_trt_dtype before adding
        # to the additive mask.
        alibi_slopes_tensor = graph_ops.add_constant(
            network, (num_heads, 1, 1),
            alibi_slopes_np.reshape(num_heads, 1, 1), dtype=np.float32)
        # Cache slot k (for k in [0, max_cache_length)) holds the K/V at
        # position k. The current step's K/V live in slots
        # [max_cache_length, max_cache_length + Sq) and their positions come
        # from position_id at runtime, so we only pre-build the cache half.
        alibi_cache_positions_fp32 = graph_ops.add_constant(
            network, (max_cache_length,),
            np.arange(max_cache_length, dtype=np.float32), dtype=np.float32)

    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([[config.rms_norm_eps]], dtype=np.float32),
        dtype=np.float32)
    eps_tensor_per_head = graph_ops.add_constant(
        network, (1, 1, 1),
        np.array([[[config.rms_norm_eps]]], dtype=np.float32),
        dtype=np.float32)

    # Attention scale.
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(network, work_np_dtype, quant_ctx)

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

    if input_embed_tensor is not None and use_input_embed_tensor is not None:
        token_embed = hidden_state
        if token_embed.dtype != work_trt_dtype:
            token_embed = network.add_cast(token_embed, work_trt_dtype).get_output(0)
        input_embed = input_embed_tensor
        if input_embed.dtype != work_trt_dtype:
            input_embed = network.add_cast(input_embed, work_trt_dtype).get_output(0)
        embed_selector = use_input_embed_tensor
        if embed_selector.dtype != work_trt_dtype:
            embed_selector = network.add_cast(embed_selector, work_trt_dtype).get_output(0)
        one = _const_in_work_dtype(
            network, (1, 1), np.array([[1.0]], dtype=work_np_dtype),
            work_np_dtype, work_trt_dtype)
        inverse_selector = network.add_elementwise(
            one, embed_selector, trt.ElementWiseOperation.SUB).get_output(0)
        token_part = network.add_elementwise(
            inverse_selector, token_embed, trt.ElementWiseOperation.PROD).get_output(0)
        embed_part = network.add_elementwise(
            embed_selector, input_embed, trt.ElementWiseOperation.PROD).get_output(0)
        hidden_state = network.add_elementwise(
            token_part, embed_part, trt.ElementWiseOperation.SUM).get_output(0)

    if position_type == "learned" and position_embed_table is not None:
        pos_gather = network.add_gather(position_embed_table, position_id, 0)
        pos_add = network.add_elementwise(
            hidden_state, pos_gather.get_output(0),
            trt.ElementWiseOperation.SUM)
        hidden_state = pos_add.get_output(0)

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

    # Build the 4D additive mask once — shared across layers. ALiBi
    # variants augment the mask with per-head linear bias.
    if position_type == "alibi":
        mask_4d = graph_ops.add_alibi_mask_4d(
            network, attention_mask_work, position_id,
            alibi_slopes_tensor, alibi_cache_positions_fp32,
            num_heads, target_dtype=work_trt_dtype)
    else:
        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask_work)

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

        # Position embedding (RoPE only; learned was applied above and ALiBi
        # is added into the attention mask).
        if position_type == "rope":
            if mrope_position_ids is not None and mrope_section is not None:
                q = graph_ops.add_apply_mrope_native_sequence(
                    network, q, num_heads, head_dim,
                    cos_half_table, sin_half_table, mrope_position_ids,
                    mrope_section, rotary_embedding_dim, interleaved_rope)
                k = graph_ops.add_apply_mrope_native_sequence(
                    network, k, num_kv_heads, head_dim,
                    cos_half_table, sin_half_table, mrope_position_ids,
                    mrope_section, rotary_embedding_dim, interleaved_rope)
            else:
                q = graph_ops.add_apply_rope_native(
                    network, q, num_heads, head_dim,
                    cos_half_table, sin_half_table, position_id,
                    rotary_embedding_dim, interleaved_rope,
                    sequence_length=None)
                k = graph_ops.add_apply_rope_native(
                    network, k, num_kv_heads, head_dim,
                    cos_half_table, sin_half_table, position_id,
                    rotary_embedding_dim, interleaved_rope,
                    sequence_length=None)

        # Present K / V (this step's raw K / V), shape (Sq, attn_size).
        present_k_outs.append(k)
        present_v_outs.append(v)

        # Concatenate cached + current K / V along the sequence dim.
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0

        context = graph_ops.add_attention_from_rows(
            network, q, all_k_cat.get_output(0), all_v_cat.get_output(0),
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=None, kv_seq=None, causal=False, mask=mask_4d,
            scale=attn_scale, tag=f"{prefix}.attn")

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)

        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        if parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network, hidden_state, hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor, norm_type, work_np_dtype)
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, work_np_dtype)

        # MLP — SwiGLU (Llama-style) or GeluFC (GPT-2-style).
        if mlp_type == "gelu_fc":
            mlp_out = _gelu_fc_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size,
                activation=activation, work_np_dtype=work_np_dtype)
        else:
            mlp_out = _swiglu_mlp(
                network, norm2,
                matmul=matmul, weights=weights, prefix=prefix,
                hidden=hidden, mlp_size=mlp_size)

        # Final residual.
        if parallel_residual:
            sum_attn = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        else:
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
        mode_label = "prefill-profile" if profile_mode == "prefill" else "dual-profile"
        print(f"[trtmc build] Building {mode_label} engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"kv={kv_attention_size}, "
              f"mlp={mlp_size}, cache={max_cache_length}, "
              f"opt_prefill={opt_prefill_length}, max_prefill={max_prefill_length}, "
              f"norm={norm_type}, mlp_type={mlp_type}, pos={position_type}, "
              f"precision={precision}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("dual-profile decoder engine build failed")
    return bytes(plan)
