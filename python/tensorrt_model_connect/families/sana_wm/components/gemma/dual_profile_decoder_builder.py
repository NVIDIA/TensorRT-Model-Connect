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
debug outputs and the VL ``embed_input`` path stay on
``standard_decoder_builder`` for now and are dispatched there from inside
``build_standard_decoder_engine``.

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

from ...builder_lifetime import get_process_trt_logger
from . import graph_ops
from . import graph_blocks

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .config import ModelConfig
    from .checkpoint_mapper import WeightDict
    from tensorrt_model_connect.quantization.context import QuantContext


def _const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
    storage_np_dtype: np.dtype | None = None,
) -> trt.ITensor:
    """Create a constant in work_np_dtype storage and cast it to work_trt_dtype.

    Needed for bf16 builds: the dual-profile builder stores bf16 weights
    on disk as fp16 (work_np_dtype = np.float16), but the runtime tensor
    must be bfloat16 to match the rest of the graph. ``add_constant``
    alone produces an fp16 constant — we need an explicit cast to
    bfloat16 so layers like IRotaryEmbeddingLayer (which require all
    inputs to share a dtype) accept it. fp16 / fp32 builds are no-ops
    because work_np_dtype maps directly to work_trt_dtype.
    """
    const = graph_ops.add_constant(
        network,
        shape,
        values,
        dtype=storage_np_dtype if storage_np_dtype is not None else work_np_dtype,
    )
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def _make_matmul_fn(
    network: trt.INetworkDefinition,
    dtype: np.dtype,
    quant_ctx: "QuantContext | None",
    *,
    preserve_bf16_weights: bool = False,
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
                network, lhs, lhs_w, rhs_w, rhs_weights,
                dtype=np.float32 if preserve_bf16_weights else dtype)
        return matmul

    if preserve_bf16_weights:
        raise ValueError("SANA-WM exact Gemma weights do not support quantized matmul")

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
    *,
    exact_sana_wm_gemma: bool = False,
    eps: float = 1.0e-6,
) -> trt.ITensor:
    if exact_sana_wm_gemma and norm_type == "rmsnorm":
        if beta is not None:
            raise ValueError("SANA-WM exact Gemma RMSNorm does not support beta")
        return graph_ops.add_sana_wm_gemma_rms_norm(network, inp, gamma, eps)
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
    activation: str,
    work_np_dtype: np.dtype,
    debug_prefix: str | None = None,
    exact_sana_wm_gemma: bool = False,
) -> trt.ITensor:
    gate = matmul(inp, hidden, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{prefix}.w_gate")
    up = matmul(inp, hidden, mlp_size,
                weights[f"{prefix}.w_up"], f"{prefix}.w_up")
    activated_gate = None
    if exact_sana_wm_gemma:
        if activation != "gelu_pytorch_tanh":
            raise ValueError(
                "SANA-WM exact Gemma gated MLP requires gelu_pytorch_tanh"
            )
        gated = graph_ops.add_sana_wm_gemma_gated_gelu(network, gate, up)
    else:
        activated_gate = graph_ops.add_activation(
            network, gate, activation, dtype=work_np_dtype)
        gated = network.add_elementwise(
            activated_gate, up, trt.ElementWiseOperation.PROD).get_output(0)
    mlp_out = matmul(gated, mlp_size, hidden,
                     weights[f"{prefix}.w_down"], f"{prefix}.w_down")
    if debug_prefix is not None:
        _mark_debug_output(network, gate, f"{debug_prefix}_mlp_gate")
        _mark_debug_output(network, up, f"{debug_prefix}_mlp_up")
        if activated_gate is not None:
            _mark_debug_output(network, activated_gate, f"{debug_prefix}_mlp_activated_gate")
        _mark_debug_output(network, gated, f"{debug_prefix}_mlp_gated")
        _mark_debug_output(network, mlp_out, f"{debug_prefix}_mlp_down")
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


def _uses_gemma_post_norm_residual(config: "ModelConfig") -> bool:
    model_type = str(config.model_type).lower()
    architectures = [str(arch).lower() for arch in config.architectures]
    return (
        model_type.startswith(("gemma2", "gemma3"))
        or any("gemma2" in arch or "gemma3" in arch for arch in architectures)
    )


def _mark_debug_output(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    name: str,
) -> None:
    cast = network.add_cast(tensor, trt.float32)
    output = cast.get_output(0)
    output.name = name
    network.mark_output(output)


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
    verbose: bool = False,
    dynamic_kv_profile_rows: list[int] | None = None,
    profile_mode: str = "dual_profile",
    hidden_state_output: bool = False,
    debug_layer_outputs: bool = False,
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

    gemma_post_norm_residual = _uses_gemma_post_norm_residual(config)
    if gemma_post_norm_residual and activation == "silu":
        activation = config.hidden_act or "gelu_pytorch_tanh"

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

    logger = get_process_trt_logger(trt, verbose=verbose)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # ---- Inputs (dynamic Sq) ---------------------------------------------
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
        prof.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq))
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
    exact_sana_wm_gemma = bool(weights.get("_sana_wm_exact_gemma", False))
    embedding_table = _const_in_work_dtype(
        network, (vocab, hidden), weights["embedding"],
        work_np_dtype, work_trt_dtype,
        storage_np_dtype=np.float32 if exact_sana_wm_gemma else None)

    # RoPE tables (only when position_type == "rope"). Built for the worst
    # case key length max_cache_length + max_prefill_length, since RoPE is
    # gathered by position_id at runtime. The half-dim tables feed TRT's
    # native IRotaryEmbeddingLayer.
    cos_half_table: trt.ITensor | None = None
    sin_half_table: trt.ITensor | None = None
    exact_rope_tables: dict[str, tuple[trt.ITensor, trt.ITensor]] = {}
    exact_layer_types = list(weights.get("_sana_wm_layer_types", []))
    if position_type == "rope":
        kmax = max_cache_length + max_prefill_length
        graph_ops.validate_native_rope_dim(rotary_embedding_dim)
        raw_rope_tables = weights.get("_sana_wm_rope_tables")
        if exact_sana_wm_gemma and isinstance(raw_rope_tables, dict):
            for layer_type, arrays in raw_rope_tables.items():
                cos_np = np.asarray(arrays[0])
                sin_np = np.asarray(arrays[1])
                expected_shape = (kmax, rotary_embedding_dim)
                if cos_np.shape[0] < kmax or cos_np.shape[1:] != expected_shape[1:]:
                    raise ValueError(
                        f"SANA-WM exact {layer_type} cosine table must cover "
                        f"{expected_shape}, got {cos_np.shape}"
                    )
                if sin_np.shape[0] < kmax or sin_np.shape[1:] != expected_shape[1:]:
                    raise ValueError(
                        f"SANA-WM exact {layer_type} sine table must cover "
                        f"{expected_shape}, got {sin_np.shape}"
                    )
                cos_tensor = _const_in_work_dtype(
                    network, expected_shape, cos_np[:kmax], work_np_dtype, work_trt_dtype)
                sin_tensor = _const_in_work_dtype(
                    network, expected_shape, sin_np[:kmax], work_np_dtype, work_trt_dtype)
                exact_rope_tables[str(layer_type)] = (cos_tensor, sin_tensor)
            if len(exact_layer_types) != num_layers:
                raise ValueError(
                    "SANA-WM exact Gemma3 layer_types must contain one entry per layer"
                )
        elif exact_sana_wm_gemma:
            cos_half_np = np.asarray(weights["_sana_wm_rope_cos"])
            sin_half_np = np.asarray(weights["_sana_wm_rope_sin"])
            expected_shape = (kmax, rotary_embedding_dim)
            if cos_half_np.shape[0] < kmax or cos_half_np.shape[1:] != expected_shape[1:]:
                raise ValueError(
                    "SANA-WM exact Gemma cosine table must cover "
                    f"{expected_shape}, got {cos_half_np.shape}"
                )
            if sin_half_np.shape[0] < kmax or sin_half_np.shape[1:] != expected_shape[1:]:
                raise ValueError(
                    "SANA-WM exact Gemma sine table must cover "
                    f"{expected_shape}, got {sin_half_np.shape}"
                )
            cos_half_np = cos_half_np[:kmax]
            sin_half_np = sin_half_np[:kmax]
        else:
            cos_half_np = graph_ops.make_rope_table_half_dim(
                kmax, head_dim, config.rope_theta, True,
                partial_rotary_factor, interleaved=interleaved_rope)
            sin_half_np = graph_ops.make_rope_table_half_dim(
                kmax, head_dim, config.rope_theta, False,
                partial_rotary_factor, interleaved=interleaved_rope)
        if not exact_rope_tables:
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
    attn_logit_softcap = config.raw.get("attn_logit_softcapping")
    final_logit_softcap = config.raw.get("final_logit_softcapping")

    # Quantization-aware matmul (passes weight_name through to QuantContext).
    matmul = _make_matmul_fn(
        network,
        work_np_dtype,
        quant_ctx,
        preserve_bf16_weights=exact_sana_wm_gemma,
    )

    # ---- Embedding -------------------------------------------------------
    emb = network.add_gather(embedding_table, token_id, 0)
    hidden_state = emb.get_output(0)  # (Sq, hidden)

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

    embedding_scale = weights.get("_embedding_scale")
    if embedding_scale is not None:
        scale = _const_in_work_dtype(
            network,
            (1, 1),
            np.array([[embedding_scale]], dtype=np.float32),
            work_np_dtype,
            work_trt_dtype,
        )
        hidden_state = network.add_elementwise(
            hidden_state, scale, trt.ElementWiseOperation.PROD
        ).get_output(0)

    # Optional embedding LayerNorm (Bloom).
    embed_norm = weights.get("embedding_norm")
    if embed_norm is not None:
        embed_norm_beta = weights.get(
            "embedding_norm_beta", np.zeros(hidden, dtype=np.float32))
        hidden_state = _norm_multi(
            network, hidden_state, hidden, embed_norm, embed_norm_beta,
            eps_tensor, "layernorm", work_np_dtype,
            exact_sana_wm_gemma=exact_sana_wm_gemma,
            eps=config.rms_norm_eps)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

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
            eps_tensor, norm_type, work_np_dtype,
            exact_sana_wm_gemma=exact_sana_wm_gemma,
            eps=config.rms_norm_eps)
        debug_layer = debug_layer_outputs and layer_idx == 0
        if debug_layer:
            _mark_debug_output(network, normed, "debug_layer_0_normed")

        # Q / K / V projections.
        q = matmul(normed, hidden, attention_size,
                   weights[f"{prefix}.w_q"], f"{prefix}.w_q")
        k = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_k"], f"{prefix}.w_k")
        v = matmul(normed, hidden, kv_attention_size,
                   weights[f"{prefix}.w_v"], f"{prefix}.w_v")
        if debug_layer:
            _mark_debug_output(network, q, "debug_layer_0_q_projected")
            _mark_debug_output(network, k, "debug_layer_0_k_projected")
            _mark_debug_output(network, v, "debug_layer_0_v_projected")

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
            if exact_sana_wm_gemma:
                q = graph_ops.add_sana_wm_gemma_rms_norm_per_head(
                    network, q, num_heads, head_dim, q_norm,
                    config.rms_norm_eps)
            else:
                q = graph_ops.add_rms_norm_per_head(
                    network, q, num_heads, head_dim, q_norm,
                    eps_tensor_per_head, dtype=work_np_dtype,
                    sequence_length=None)
        k_norm = weights.get(f"{prefix}.k_norm")
        if k_norm is not None:
            if exact_sana_wm_gemma:
                k = graph_ops.add_sana_wm_gemma_rms_norm_per_head(
                    network, k, num_kv_heads, head_dim, k_norm,
                    config.rms_norm_eps)
            else:
                k = graph_ops.add_rms_norm_per_head(
                    network, k, num_kv_heads, head_dim, k_norm,
                    eps_tensor_per_head, dtype=work_np_dtype,
                    sequence_length=None)

        # Position embedding (RoPE only; learned was applied above and ALiBi
        # is added into the attention mask).
        if position_type == "rope":
            if exact_sana_wm_gemma:
                layer_cos = cos_half_table
                layer_sin = sin_half_table
                if exact_rope_tables:
                    layer_type = exact_layer_types[layer_idx]
                    try:
                        layer_cos, layer_sin = exact_rope_tables[layer_type]
                    except KeyError as exc:
                        raise ValueError(
                            f"missing exact SANA-WM RoPE table for {layer_type!r}"
                        ) from exc
                q = graph_ops.add_sana_wm_gemma_rope(
                    network, q, num_heads, head_dim,
                    layer_cos, layer_sin, position_id,
                    rotary_embedding_dim, interleaved_rope)
                k = graph_ops.add_sana_wm_gemma_rope(
                    network, k, num_kv_heads, head_dim,
                    layer_cos, layer_sin, position_id,
                    rotary_embedding_dim, interleaved_rope)
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
        if debug_layer:
            _mark_debug_output(network, q, "debug_layer_0_q_rope")
            _mark_debug_output(network, k, "debug_layer_0_k_rope")

        # Present K / V (this step's raw K / V), shape (Sq, attn_size).
        present_k_outs.append(k)
        present_v_outs.append(v)

        # Concatenate cached + current K / V along the sequence dim.
        all_k_cat = network.add_concatenation([cache_k_inputs[layer_idx], k])
        all_k_cat.axis = 0
        all_v_cat = network.add_concatenation([cache_v_inputs[layer_idx], v])
        all_v_cat.axis = 0

        if exact_sana_wm_gemma:
            context = graph_ops.add_sana_wm_gemma_attention(
                network, q, all_k_cat.get_output(0), all_v_cat.get_output(0), mask_4d,
                num_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, scale=attn_scale)
        else:
            context = graph_ops.add_attention_from_rows(
                network, q, all_k_cat.get_output(0), all_v_cat.get_output(0),
                num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                q_seq=None, kv_seq=None, causal=False, mask=mask_4d,
                scale=attn_scale, logit_softcap=attn_logit_softcap,
                tag=f"{prefix}.attn")
        if debug_layer:
            _mark_debug_output(network, context, "debug_layer_0_context")

        attn_out = matmul(context, attention_size, hidden,
                          weights[f"{prefix}.w_o"], f"{prefix}.w_o")
        o_bias = weights.get(f"{prefix}.o_bias")
        if o_bias is not None:
            attn_out = graph_ops.add_bias_sum(
                network, attn_out, hidden, o_bias, dtype=work_np_dtype)
        if debug_layer:
            _mark_debug_output(network, attn_out, "debug_layer_0_attn_projected")

        gemma2_norms = (
            weights.get(f"{prefix}.pre_ff_norm") is not None
            and weights.get(f"{prefix}.post_ff_norm") is not None
        )

        if gemma2_norms:
            attn_out = _norm_multi(
                network, attn_out, hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, work_np_dtype,
                exact_sana_wm_gemma=exact_sana_wm_gemma,
                eps=config.rms_norm_eps)
            if debug_layer:
                _mark_debug_output(network, attn_out, "debug_layer_0_attn_post_norm")
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.pre_ff_norm"],
                weights.get(f"{prefix}.pre_ff_norm_beta"),
                eps_tensor, norm_type, work_np_dtype,
                exact_sana_wm_gemma=exact_sana_wm_gemma,
                eps=config.rms_norm_eps)
        # Residual structure: parallel (GPT-NeoX / CodeGen / Falcon-3) vs
        # sequential (everything else).
        elif parallel_residual:
            post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
            if post_attn_norm_w is not None:
                norm2 = _norm_multi(
                    network, hidden_state, hidden,
                    post_attn_norm_w,
                    weights.get(f"{prefix}.post_attn_norm_beta"),
                    eps_tensor, norm_type, work_np_dtype,
                    exact_sana_wm_gemma=exact_sana_wm_gemma,
                    eps=config.rms_norm_eps)
            else:
                norm2 = normed
        else:
            residual1 = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            norm2 = _norm_multi(
                network, residual1.get_output(0), hidden,
                weights[f"{prefix}.post_attn_norm"],
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, work_np_dtype,
                exact_sana_wm_gemma=exact_sana_wm_gemma,
                eps=config.rms_norm_eps)

        if debug_layer:
            _mark_debug_output(network, norm2, "debug_layer_0_pre_ff_norm")

        if debug_layer_outputs:
            if parallel_residual and not gemma2_norms:
                debug_post_attn = network.add_elementwise(
                    hidden_state, attn_out, trt.ElementWiseOperation.SUM
                ).get_output(0)
            else:
                debug_post_attn = residual1.get_output(0)
            _mark_debug_output(
                network, debug_post_attn, f"debug_post_attn_{layer_idx}"
            )

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
                hidden=hidden, mlp_size=mlp_size,
                activation=activation, work_np_dtype=work_np_dtype,
                debug_prefix="debug_layer_0" if debug_layer else None,
                exact_sana_wm_gemma=exact_sana_wm_gemma)

        # Final residual.
        if gemma2_norms:
            mlp_out = _norm_multi(
                network, mlp_out, hidden,
                weights[f"{prefix}.post_ff_norm"],
                weights.get(f"{prefix}.post_ff_norm_beta"),
                eps_tensor, norm_type, work_np_dtype,
                exact_sana_wm_gemma=exact_sana_wm_gemma,
                eps=config.rms_norm_eps)
            if debug_layer:
                _mark_debug_output(network, mlp_out, "debug_layer_0_mlp_post_norm")
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        elif parallel_residual:
            sum_attn = network.add_elementwise(
                hidden_state, attn_out, trt.ElementWiseOperation.SUM)
            residual2 = network.add_elementwise(
                sum_attn.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        else:
            residual2 = network.add_elementwise(
                residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
        hidden_state = residual2.get_output(0)
        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # ---- Final norm + LM head -------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _norm_multi(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"),
            eps_tensor, norm_type, work_np_dtype,
            exact_sana_wm_gemma=exact_sana_wm_gemma,
            eps=config.rms_norm_eps)

    if hidden_state_output:
        hs_out = network.add_identity(hidden_state).get_output(0)
        hs_out.name = "hidden_state"
        network.mark_output(hs_out)

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
        dtype=np.float32 if exact_sana_wm_gemma else work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        zero_bias = np.zeros(out_vocab, dtype=work_np_dtype)
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, zero_bias, dtype=work_np_dtype)

    if final_logit_softcap is not None and float(final_logit_softcap) > 0.0:
        logits = graph_ops.add_tanh_softcap(
            network, logits, float(final_logit_softcap), scalar_shape=(1, 1))

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
