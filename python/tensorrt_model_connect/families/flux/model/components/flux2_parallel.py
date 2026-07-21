# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel FLUX.2-dev DiT (Diffusion Transformer) engine builder.

Builds a TensorRT engine for the FLUX.2-dev transformer denoiser,
which has two types of blocks:
  1. Joint transformer blocks (double-stream): image and text attend jointly
  2. Single transformer blocks (single-stream): operate on concatenated tokens

Key differences from FLUX.1:
  - Inner dim = 6144 (48 heads x 128), joint = 8, single = 48
  - MLP ratio = 3.0 (vs 4.0)
  - 4D RoPE (32,32,32,32) vs 3D (16,56,56)
  - Global modulation inputs instead of per-block norm1.linear
  - Fused to_qkv_mlp_proj in single blocks
  - FFN uses .linear_in / .linear_out naming
  - No pooled projections; joint attn dim matches dit_dim

Engine I/O:
    Inputs:
        hidden_states [num_img_tokens, dim] float32
        encoder_hidden_states [text_seq_len, dim] float32
        temb [dim] float32
        rotary_cos [total_seq_len, head_dim] float32
        rotary_sin [total_seq_len, head_dim] float32
    Outputs:
        output [num_img_tokens, out_channels] float32

Global modulation weights (double_stream_modulation_img/txt, single_stream_modulation)
are baked into the engine as constant linear projections from temb.

Preprocessor weights (timestep MLP, x_embedder, context_embedder, RoPE,
global modulation tables) are handled externally by the runtime.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .. import model as graph_ops
from .....parallel_config import (
    ParallelConfig,
    _slice_first_dim,
    _slice_last_dim,
    add_all_reduce_sum,
    normalize_parallel_config,
    validate_dit_tp,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...weights import WeightDict


# --- Helpers for STRONGLY_TYPED reduced-precision networks ---
#
# Strategy: strongly typed reduced-precision network. BF16 can safely keep
# normalization reductions in reduced precision; FP16 must compute those
# reductions in FP32 to avoid overflow in FLUX.2 AdaLN/QK norms.
#
# cast_dtype controls the reduced precision type:
#   trt.float16  — FP16 (10-bit mantissa, max 65504)
#   trt.bfloat16 — BF16 (7-bit mantissa, FP32 dynamic range)

# Module-level settings, configured by build_flux2_dit_engine.
_CAST_DTYPE = trt.float16
_FP8_MODE = False   # When True, uses FP8 Q/DQ with TN layout for matmuls
_FP8_SCALES = {}    # Per-layer FP8 scales: {layer_name: {input_scale, weight_scale}}

# Hold references to weight arrays to prevent GC during engine build
_weight_refs = []

_FP8_WEIGHT_SUFFIX = ".weight.fp8_tn"


def _to_compute_dtype(network, tensor):
    """Cast to compute dtype if not already."""
    if tensor.dtype == _CAST_DTYPE:
        return tensor
    return network.add_cast(tensor, _CAST_DTYPE).get_output(0)


def _to_fp32(network, tensor):
    """Cast to FP32 if not already."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def _fp16_compute() -> bool:
    """True when strongly typed FLUX.2 is using FP16 as its compute dtype."""
    return _CAST_DTYPE == trt.float16


def _make_reduced_weights(data_fp32, shape):
    """Create TRT Weights in the current reduced precision dtype."""
    if _CAST_DTYPE == trt.bfloat16:
        import ml_dtypes
        if (
            isinstance(data_fp32, np.ndarray)
            and data_fp32.dtype == ml_dtypes.bfloat16
            and data_fp32.flags.c_contiguous
        ):
            bf16_arr = data_fp32
        else:
            bf16_arr = np.ascontiguousarray(data_fp32.astype(ml_dtypes.bfloat16))
        return trt.Weights(trt.bfloat16, bf16_arr.ctypes.data, bf16_arr.size), bf16_arr
    else:
        if (
            isinstance(data_fp32, np.ndarray)
            and data_fp32.dtype == np.float16
            and data_fp32.flags.c_contiguous
        ):
            fp16_arr = data_fp32
        else:
            fp16_arr = np.ascontiguousarray(data_fp32, dtype=np.float16)
        return trt.Weights(fp16_arr), fp16_arr


def _add_constant_reduced(network, shape, values_fp32):
    """Add constant in reduced precision (BF16/FP16)."""
    w, arr_ref = _make_reduced_weights(values_fp32, shape)
    _weight_refs.append(arr_ref)
    return network.add_constant(shape, w).get_output(0)


def _fp8_weight_key(prefix: str) -> str:
    return f"{prefix}{_FP8_WEIGHT_SUFFIX}"


def _convert_weight_to_fp8_tn(rhs_weights, wt_scale):
    """Convert [in, out] FP32/BF16 weight storage to FP8 TN [out, in]."""
    import ml_dtypes

    rhs_tn = np.ascontiguousarray(rhs_weights.T.astype(np.float32))
    return np.ascontiguousarray(
        (rhs_tn / wt_scale).astype(ml_dtypes.float8_e4m3fn))


def _matmul_reduced_precision(network, lhs, lhs_width, rhs_width, rhs_weights,
                              inp_scale=None, wt_scale=None,
                              fp8_weight_tn=None):
    """Matmul in reduced precision. Input/output stay in _CAST_DTYPE.

    If inp_scale/wt_scale are provided and _FP8_MODE is True, inserts FP8 Q/DQ
    nodes with TN layout for proper fusion on Hopper+/Blackwell.
    """
    # FP8 path: ONLY use FP8 Q/DQ for layers with calibrated scales.
    # Non-calibrated layers (context_embedder, x_embedder, time_text_embed,
    # norm_out, etc.) fall through to the BF16 path below.
    if _FP8_MODE and inp_scale is not None and wt_scale is not None:
        # FP8 Q/DQ path with TN layout (required for fusion on Blackwell).
        #
        # DQ output type = BF16 (not FP32) so the entire network stays in
        # BF16 as the base type.  This prevents FP32 intermediates from
        # bloating activation memory (12.9 GB → ~1 GB) and avoids creating
        # backend boundaries that fragment TensorRT compiler partitioning.
        #
        # TensorRT fuses: DQ(A8) + DQ(W8) -> MatMul into
        # a single FP8 FC kernel regardless of the DQ output type.
        # Q/DQ on input (activation)
        lhs_ready = _to_compute_dtype(network, lhs)
        s_inp = network.add_constant((), trt.Weights(np.array(inp_scale, dtype=np.float32)))
        q_inp = network.add_quantize(lhs_ready, s_inp.get_output(0), trt.DataType.FP8)
        dq_inp = network.add_dequantize(q_inp.get_output(0), s_inp.get_output(0), _CAST_DTYPE)

        # Weight: FP8 constant + DQ (TN layout)
        # Quantize: fp8_val = round(weight / scale), then DQ recovers: fp8_val * scale ≈ weight
        rhs_fp8 = fp8_weight_tn
        if rhs_fp8 is None:
            rhs_fp8 = _convert_weight_to_fp8_tn(rhs_weights, wt_scale)
        rhs_fp8_const = network.add_constant(
            (rhs_width, lhs_width),
            trt.Weights(trt.DataType.FP8, rhs_fp8.ctypes.data, rhs_fp8.size))
        _weight_refs.append(rhs_fp8)
        s_wt = network.add_constant((), trt.Weights(np.array(wt_scale, dtype=np.float32)))
        dq_wt = network.add_dequantize(rhs_fp8_const.get_output(0), s_wt.get_output(0), _CAST_DTYPE)

        # MatMul with TN layout (opB=TRANSPOSE) — required for FP8 fusion
        mm = network.add_matrix_multiply(
            dq_inp.get_output(0), trt.MatrixOperation.NONE,
            dq_wt.get_output(0), trt.MatrixOperation.TRANSPOSE)
        return mm.get_output(0)

    # BF16/FP16 path, also used for non-quantized matmuls in FP8 mode.
    lhs_cast = _to_compute_dtype(network, lhs)
    w, arr_ref = _make_reduced_weights(rhs_weights, (lhs_width, rhs_width))
    _weight_refs.append(arr_ref)
    rhs = network.add_constant((lhs_width, rhs_width), w)
    mm = network.add_matrix_multiply(
        lhs_cast, trt.MatrixOperation.NONE,
        rhs.get_output(0), trt.MatrixOperation.NONE)
    return mm.get_output(0)


def _bias_sum_reduced(network, inp, width, bias):
    """Bias addition in compute dtype."""
    bias_t = _add_constant_reduced(network, (1, width), bias)
    return network.add_elementwise(
        inp, bias_t, trt.ElementWiseOperation.SUM).get_output(0)


def build_flux2_dit_engine(
    weights: "WeightDict",
    *,
    dim: int = 6144,
    num_heads: int = 48,
    num_layers: int = 8,
    num_single_layers: int = 48,
    num_img_tokens: int,
    text_seq_len: int = 512,
    mlp_ratio: float = 3.0,
    packed_channels: int = 128,
    t5_dim: int = 15360,
    freq_dim: int = 256,
    eps: float = 1e-6,
    verbose: bool = False,
    cast_dtype: str = "fp16",
    fp8_scales: dict | None = None,
    parallel_config: ParallelConfig | None = None,
) -> bytes:
    """Build FLUX.2-dev DiT denoiser TRT engine plan.

    Args:
        cast_dtype: Reduced precision for matmuls — "fp16" or "bf16".
        fp8_scales: Per-layer FP8 scales dict {layer_name: {input_scale, weight_scale}}.
            When provided, uses FP8 Q/DQ with TN layout for matmul fusion.
    """
    global _CAST_DTYPE, _FP8_MODE, _FP8_SCALES, _weight_refs
    _CAST_DTYPE = trt.bfloat16 if cast_dtype == "bf16" else trt.float16
    _FP8_MODE = fp8_scales is not None
    # FP8 mode uses the selected reduced precision as the base type. Linear
    # layers get FP8 Q/DQ and attention uses TRT's native IAttention API.
    # This avoids FP32 intermediates that bloat activation memory 15× and
    # prevents explicit Cast nodes from fragmenting TensorRT compiler partitioning.
    _FP8_SCALES = fp8_scales or {}
    _weight_refs = []  # clear from previous builds

    head_dim = dim // num_heads
    ffn_dim = int(dim * mlp_ratio)
    total_seq = text_seq_len + num_img_tokens
    parallel = normalize_parallel_config(parallel_config)
    validate_dit_tp(
        dim=dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        parallel=parallel,
        feature="Flux.2 tensor parallel",
    )
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    tp_suffix = (
        f", tp={parallel.tp_size}, rank={parallel.rank}"
        if parallel.enabled else "")
    print(f"  [flux2-dit] Network: strongly_typed=True, "
          f"cast_dtype={cast_dtype}, fp8={_FP8_MODE}{tp_suffix}",
          file=sys.stderr)

    # --- Inputs ---
    # All preprocessor ops (x_embedder, context_embedder, temb MLP) are baked
    # into the engine — no CPU/cuBLAS needed at runtime.
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (num_img_tokens, packed_channels))
    encoder_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, t5_dim))
    timestep_inp = network.add_input(
        "timestep", trt.float32, (1,))
    guidance_inp = network.add_input(
        "guidance", trt.float32, (1,))
    rotary_cos = network.add_input(
        "rotary_cos", trt.float32, (total_seq, head_dim))
    rotary_sin = network.add_input(
        "rotary_sin", trt.float32, (total_seq, head_dim))

    eps_t = _add_constant_reduced(network, (1, 1), np.array([eps], dtype=np.float32))

    # Cast FP32 inputs to the reduced compute dtype at the boundary.
    rotary_cos = network.add_cast(rotary_cos, _CAST_DTYPE).get_output(0)
    rotary_sin = network.add_cast(rotary_sin, _CAST_DTYPE).get_output(0)

    # Split RoPE for text and image
    txt_cos = network.add_slice(rotary_cos, (0, 0), (text_seq_len, head_dim), (1, 1)).get_output(0)
    txt_sin = network.add_slice(rotary_sin, (0, 0), (text_seq_len, head_dim), (1, 1)).get_output(0)
    img_cos = network.add_slice(rotary_cos, (text_seq_len, 0), (num_img_tokens, head_dim), (1, 1)).get_output(0)
    img_sin = network.add_slice(rotary_sin, (text_seq_len, 0), (num_img_tokens, head_dim), (1, 1)).get_output(0)

    # Cast FP32 inputs to the reduced compute dtype at the boundary.
    hidden = _to_compute_dtype(network, hidden_inp)
    encoder_hidden = _to_compute_dtype(network, encoder_inp)

    # --- x_embedder: packed latents [num_img_tokens, packed_channels] → [num_img_tokens, dim] ---
    x_emb_w = weights.get("x_embedder.weight")  # [packed_channels, dim]
    if x_emb_w is not None:
        _xe_inp_s = _FP8_SCALES.get("x_embedder", {}).get("input_scale")
        _xe_wt_s = _FP8_SCALES.get("x_embedder", {}).get("weight_scale")
        hidden = _matmul_reduced_precision(network, hidden, packed_channels, dim, x_emb_w,
                                           inp_scale=_xe_inp_s, wt_scale=_xe_wt_s)
        x_emb_b = weights.get("x_embedder.bias")
        if x_emb_b is not None:
            hidden = _bias_sum_reduced(network, hidden, dim, x_emb_b)

    # --- context_embedder: raw T5 [text_seq, t5_dim] → [text_seq, dim] ---
    ctx_w = weights.get("context_embedder.weight")  # [t5_dim, dim]
    if ctx_w is not None:
        encoder_hidden = _matmul_reduced_precision(
            network, encoder_hidden, t5_dim, dim, ctx_w)
        ctx_b = weights.get("context_embedder.bias")
        if ctx_b is not None:
            encoder_hidden = _bias_sum_reduced(network, encoder_hidden, dim, ctx_b)

    # --- temb MLP: sinusoidal(timestep) → Linear → SiLU → Linear + guidance ---
    # Build sinusoidal frequency table as constant: freqs[i] = 1/(10000^(2i/freq_dim))
    half_freq = freq_dim // 2
    freq_np = 1.0 / (10000.0 ** (np.arange(half_freq, dtype=np.float32) / half_freq))
    # freq_const shape [1, half_freq]
    freq_const = _add_constant_reduced(network, (1, half_freq), freq_np)

    def _build_sinusoidal_embedding(scalar_inp):
        """Build sinusoidal embedding: scalar → [1, freq_dim]."""
        # Scale timestep by 1000 (FLUX convention)
        s1000 = graph_ops.add_constant(network, (1,), np.array([1000.0], dtype=np.float32))
        scaled = network.add_elementwise(
            scalar_inp, s1000, trt.ElementWiseOperation.PROD).get_output(0)
        scaled = _to_compute_dtype(network, scaled)
        # Reshape to [1, 1] for broadcast multiply
        scaled_2d = network.add_shuffle(scaled)
        scaled_2d.reshape_dims = (1, 1)
        # args = scaled * freqs → [1, half_freq]
        args = network.add_elementwise(
            scaled_2d.get_output(0), freq_const,
            trt.ElementWiseOperation.PROD).get_output(0)
        # cos and sin — C++ convention is [cos, sin]
        cos_out = network.add_unary(args, trt.UnaryOperation.COS).get_output(0)
        sin_out = network.add_unary(args, trt.UnaryOperation.SIN).get_output(0)
        # Concatenate [cos, sin] → [1, freq_dim]
        cat = network.add_concatenation([cos_out, sin_out])
        cat.axis = 1
        return cat.get_output(0)

    def _build_mlp_2layer(emb, w0_key, w2_key):
        """Linear → SiLU → Linear, returns [1, dim]."""
        w0 = weights.get(f"{w0_key}.weight")
        b0 = weights.get(f"{w0_key}.bias")
        w2 = weights.get(f"{w2_key}.weight")
        b2 = weights.get(f"{w2_key}.bias")
        if w0 is None or w2 is None:
            return None
        in_dim = w0.shape[0]
        out_dim = w0.shape[1]
        x = _matmul_reduced_precision(network, emb, in_dim, out_dim, w0)
        if b0 is not None:
            x = _bias_sum_reduced(network, x, out_dim, b0)
        # SiLU = x * sigmoid(x)
        sig = network.add_activation(x, trt.ActivationType.SIGMOID)
        x = network.add_elementwise(
            x, sig.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
        out2_dim = w2.shape[1]
        x = _matmul_reduced_precision(network, x, out_dim, out2_dim, w2)
        if b2 is not None:
            x = _bias_sum_reduced(network, x, out2_dim, b2)
        return x

    # Timestep embedding: sinusoidal(t) → MLP → [1, dim]
    t_sinusoidal = _build_sinusoidal_embedding(timestep_inp)
    temb_combined = _build_mlp_2layer(
        t_sinusoidal,
        "time_text_embed.timestep_embedder.linear_1",
        "time_text_embed.timestep_embedder.linear_2")

    # Guidance embedding: sinusoidal(g) → MLP → [1, dim], added to temb
    g_sinusoidal = _build_sinusoidal_embedding(guidance_inp)
    g_proj = _build_mlp_2layer(
        g_sinusoidal,
        "time_text_embed.guidance_embedder.linear_1",
        "time_text_embed.guidance_embedder.linear_2")
    if g_proj is not None and temb_combined is not None:
        temb_combined = network.add_elementwise(
            temb_combined, g_proj, trt.ElementWiseOperation.SUM).get_output(0)

    # Reshape temb from [1, dim] to [dim] for downstream modulation
    temb_squeeze = network.add_shuffle(temb_combined)
    temb_squeeze.reshape_dims = (dim,)
    temb_work = temb_squeeze.get_output(0)

    # --- Compute SiLU(temb) once for all modulation ---
    temb_silu = network.add_activation(temb_work, trt.ActivationType.SIGMOID)
    temb_silu_out = network.add_elementwise(
        temb_work, temb_silu.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    # --- Global modulation weights as constants ---
    # These are shared across all blocks: temb @ mod_weight -> [6*dim] or [3*dim]
    mod_img_w = weights.get("double_stream_modulation_img")
    mod_txt_w = weights.get("double_stream_modulation_txt")
    mod_single_w = weights.get("single_stream_modulation")
    # ===================== Joint Transformer Blocks =====================
    for layer_idx in range(num_layers):
        p = f"transformer_blocks.{layer_idx}"
        print(f"  [flux2-dit] Joint block {layer_idx}/{num_layers}", file=sys.stderr)

        # --- Global modulation: SiLU(temb) @ mod_weight ---
        # mod_img: [dim] @ [dim, 6*dim] -> [6*dim]
        mod_img_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 6 * dim, mod_img_w)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_6(
            network, mod_img_proj, dim)

        mod_txt_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 6 * dim, mod_txt_w)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _chunk_6(
            network, mod_txt_proj, dim)

        # --- AdaLN-Zero for image ---
        normed_hidden = _adaln_modulate(
            network, hidden, scale_msa, shift_msa, dim, eps_t,
            num_img_tokens, eps)

        # --- AdaLN-Zero for text ---
        normed_encoder = _adaln_modulate(
            network, encoder_hidden, c_scale_msa, c_shift_msa, dim, eps_t,
            text_seq_len, eps)

        # --- Joint Attention (BF16 projections) ---
        # Image QKV
        q_img = _linear_col_parallel(
            network, normed_hidden, dim, dim, weights, f"{p}.attn.to_q", parallel)
        k_img = _linear_col_parallel(
            network, normed_hidden, dim, dim, weights, f"{p}.attn.to_k", parallel)
        v_img = _linear_col_parallel(
            network, normed_hidden, dim, dim, weights, f"{p}.attn.to_v", parallel)

        # Text QKV (added projections)
        q_txt = _linear_col_parallel(
            network, normed_encoder, dim, dim, weights, f"{p}.attn.add_q_proj", parallel)
        k_txt = _linear_col_parallel(
            network, normed_encoder, dim, dim, weights, f"{p}.attn.add_k_proj", parallel)
        v_txt = _linear_col_parallel(
            network, normed_encoder, dim, dim, weights, f"{p}.attn.add_v_proj", parallel)

        # QK norm
        q_img = _rms_norm_per_head_seq(network, q_img, local_num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, num_img_tokens)
        k_img = _rms_norm_per_head_seq(network, k_img, local_num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, num_img_tokens)
        q_txt = _rms_norm_per_head_seq(network, q_txt, local_num_heads, head_dim, weights[f"{p}.attn.norm_added_q.weight"], eps_t, text_seq_len)
        k_txt = _rms_norm_per_head_seq(network, k_txt, local_num_heads, head_dim, weights[f"{p}.attn.norm_added_k.weight"], eps_t, text_seq_len)

        # Apply RoPE to image Q, K
        q_img = _apply_native_rope_from_full_cache(
            network, q_img, img_cos, img_sin, local_num_heads, head_dim, num_img_tokens)
        k_img = _apply_native_rope_from_full_cache(
            network, k_img, img_cos, img_sin, local_num_heads, head_dim, num_img_tokens)

        # Apply RoPE to text Q, K
        q_txt = _apply_native_rope_from_full_cache(
            network, q_txt, txt_cos, txt_sin, local_num_heads, head_dim, text_seq_len)
        k_txt = _apply_native_rope_from_full_cache(
            network, k_txt, txt_cos, txt_sin, local_num_heads, head_dim, text_seq_len)

        # Concatenate: [text, image] for joint attention
        q_cat = network.add_concatenation([q_txt, q_img])
        q_cat.axis = 0  # [total_seq, dim]
        k_cat = network.add_concatenation([k_txt, k_img])
        k_cat.axis = 0
        v_cat = network.add_concatenation([v_txt, v_img])
        v_cat.axis = 0

        # Multi-head attention via TRT native IAttention.
        attn_out = _mha(network, q_cat.get_output(0), k_cat.get_output(0),
                        v_cat.get_output(0), local_num_heads, head_dim, total_seq)

        # Split attention output back into text and image
        txt_attn = network.add_slice(attn_out, (0, 0), (text_seq_len, local_dim), (1, 1)).get_output(0)
        img_attn = network.add_slice(attn_out, (text_seq_len, 0), (num_img_tokens, local_dim), (1, 1)).get_output(0)

        # Image output projection + gate + residual (BF16 projection)
        img_attn_proj = _linear_row_parallel(
            network, img_attn, local_dim, dim, weights, f"{p}.attn.to_out.0", parallel)
        img_attn_gated = _gate_1d(network, img_attn_proj, gate_msa, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, img_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Text output projection + gate + residual
        txt_attn_proj = _linear_row_parallel(
            network, txt_attn, local_dim, dim, weights, f"{p}.attn.to_add_out", parallel)
        txt_attn_gated = _gate_1d(network, txt_attn_proj, c_gate_msa, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, txt_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Image FFN (linear_in / linear_out naming) ---
        normed_ff = _adaln_modulate(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t,
            num_img_tokens, eps)
        ff_out = _swiglu_ffn_parallel(
            network, normed_ff, dim, weights, f"{p}.ff", parallel)
        ff_gated = _gate_1d(network, ff_out, gate_mlp, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Text FFN (linear_in / linear_out naming) ---
        normed_ctx_ff = _adaln_modulate(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t,
            text_seq_len, eps)
        ctx_ff_out = _swiglu_ffn_parallel(
            network, normed_ctx_ff, dim, weights, f"{p}.ff_context", parallel)
        ctx_ff_gated = _gate_1d(network, ctx_ff_out, c_gate_mlp, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, ctx_ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

    # ===================== Single Transformer Blocks =====================
    for layer_idx in range(num_single_layers):
        p = f"single_transformer_blocks.{layer_idx}"
        if layer_idx % 8 == 0:
            print(f"  [flux2-dit] Single block {layer_idx}/{num_single_layers}", file=sys.stderr)

        # Concatenate text + image
        cat_hidden = network.add_concatenation([encoder_hidden, hidden])
        cat_hidden.axis = 0  # [total_seq, dim]
        residual = cat_hidden.get_output(0)

        # --- Global modulation: SiLU(temb) @ mod_single_weight ---
        mod_single_proj = _matmul_bias_1d_opt(network, temb_silu_out, dim, 3 * dim, mod_single_w)
        shift_msa_s, scale_msa_s, gate_msa_s = _chunk_3(
            network, mod_single_proj, dim)

        # AdaLN-Zero modulation
        normed_cat = _adaln_modulate(
            network, residual, scale_msa_s, shift_msa_s, dim, eps_t,
            total_seq, eps)

        # --- Fused QKV + MLP projection ---
        # to_qkv_mlp_proj: [dim, 3*dim + 2*ffn_dim]  (gated MLP: gate + value)
        fused_out_dim = 3 * local_dim + 2 * local_ffn_dim
        fused_w = _slice_flux2_single_fused_qkv_mlp_weight(
            weights[f"{p}.attn.to_qkv_mlp_proj.weight"],
            dim=dim,
            ffn_dim=ffn_dim,
            parallel=parallel,
        )
        _fused_inp_s = _FP8_SCALES.get(f"{p}.attn.to_qkv_mlp_proj", {}).get("input_scale")
        _fused_wt_s = _FP8_SCALES.get(f"{p}.attn.to_qkv_mlp_proj", {}).get("weight_scale")
        _fused_fp8_weight_tn = (
            None if parallel.enabled
            else weights.get(_fp8_weight_key(f"{p}.attn.to_qkv_mlp_proj"))
        )
        fused = _matmul_reduced_precision(network, normed_cat, dim, fused_out_dim, fused_w,
                                           inp_scale=_fused_inp_s, wt_scale=_fused_wt_s,
                                           fp8_weight_tn=_fused_fp8_weight_tn)
        fused_b = weights.get(f"{p}.attn.to_qkv_mlp_proj.bias")
        if fused_b is not None:
            fused_b = _slice_flux2_single_fused_qkv_mlp_bias(
                fused_b,
                dim=dim,
                ffn_dim=ffn_dim,
                parallel=parallel,
            )
            fused = _bias_sum_reduced(network, fused, fused_out_dim, fused_b)

        # Slice: Q [dim], K [dim], V [dim], MLP_gate [ffn_dim], MLP_value [ffn_dim]
        q_s = network.add_slice(fused, (0, 0), (total_seq, local_dim), (1, 1)).get_output(0)
        k_s = network.add_slice(fused, (0, local_dim), (total_seq, local_dim), (1, 1)).get_output(0)
        v_s = network.add_slice(fused, (0, 2 * local_dim), (total_seq, local_dim), (1, 1)).get_output(0)
        mlp_gate = network.add_slice(fused, (0, 3 * local_dim), (total_seq, local_ffn_dim), (1, 1)).get_output(0)
        mlp_value = network.add_slice(fused, (0, 3 * local_dim + local_ffn_dim), (total_seq, local_ffn_dim), (1, 1)).get_output(0)

        # SwiGLU on MLP branch: silu(x1) * x2
        mlp_gate_act = graph_ops.add_activation(network, mlp_gate, "silu")
        mlp_hidden = network.add_elementwise(
            mlp_gate_act, mlp_value, trt.ElementWiseOperation.PROD).get_output(0)

        # QK norm
        q_s = _rms_norm_per_head_seq(network, q_s, local_num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, total_seq)
        k_s = _rms_norm_per_head_seq(network, k_s, local_num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, total_seq)

        # Apply RoPE (full sequence: text + image cos/sin)
        q_s = _apply_native_rope_from_full_cache(
            network, q_s, rotary_cos, rotary_sin, local_num_heads, head_dim, total_seq)
        k_s = _apply_native_rope_from_full_cache(
            network, k_s, rotary_cos, rotary_sin, local_num_heads, head_dim, total_seq)

        attn_out_s = _mha(network, q_s, k_s, v_s, local_num_heads, head_dim, total_seq)

        # Concatenate attn + mlp -> to_out projection
        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 1  # [total_seq, dim + ffn_dim]

        combined = _linear_flux2_single_block_out_parallel(
            network,
            cat_attn_mlp.get_output(0),
            local_dim=local_dim,
            local_ffn_dim=local_ffn_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            weights=weights,
            prefix=f"{p}.attn.to_out",
            parallel=parallel,
        )

        # Gate + residual
        gated_s = _gate_1d(network, combined, gate_msa_s, total_seq)
        cat_hidden_out = network.add_elementwise(
            residual, gated_s,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Split back
        encoder_hidden = network.add_slice(
            cat_hidden_out, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
        hidden = network.add_slice(
            cat_hidden_out, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)

    # ===================== Final Output =====================
    # AdaLayerNormContinuous: SiLU(temb) -> Linear(dim, 2*dim) -> chunk(2) -> scale, shift
    final_norm_w = weights["norm_out.linear.weight"]
    final_norm_b = weights.get("norm_out.linear.bias")

    # temb_work is already in compute dtype (BF16 or FP32) from the temb MLP above
    temb_final = temb_work
    temb_silu_f = network.add_activation(temb_final, trt.ActivationType.SIGMOID)
    temb_silu_f_out = network.add_elementwise(
        temb_final, temb_silu_f.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    final_proj = _matmul_bias_1d_opt(network, temb_silu_f_out, dim, 2 * dim, final_norm_w, final_norm_b)
    final_scale = network.add_slice(final_proj, (0,), (dim,), (1,)).get_output(0)
    final_shift = network.add_slice(final_proj, (dim,), (dim,), (1,)).get_output(0)

    output = _adaln_modulate(
        network, hidden, final_scale, final_shift, dim, eps_t,
        num_img_tokens, eps)

    # proj_out: [num_img_tokens, dim] -> [num_img_tokens, out_channels]
    proj_out_w = weights["proj_out.weight"]
    out_channels = proj_out_w.shape[1]
    _po_inp_s = _FP8_SCALES.get("proj_out", {}).get("input_scale")
    _po_wt_s = _FP8_SCALES.get("proj_out", {}).get("weight_scale")
    output = _matmul_reduced_precision(network, output, dim, out_channels, proj_out_w,
                                        inp_scale=_po_inp_s, wt_scale=_po_wt_s)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = _bias_sum_reduced(network, output, out_channels, proj_out_b)

    # Cast back to FP32 at output boundary
    output = _to_fp32(network, output)

    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)

    print(f"[flux2-dit] Building TRT engine "
          f"(dim={dim}, joint={num_layers}, single={num_single_layers}, "
          f"img_tokens={num_img_tokens}, text_seq={text_seq_len}{tp_suffix}) ...",
          file=sys.stderr)
    # Use verbose logging for this large model to capture TRT errors
    logger.min_severity = trt.Logger.INFO
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for FLUX.2 DiT")
    return bytes(plan)


def _matmul_bias_1d_opt(network, inp, in_dim, out_dim, weight, bias=None):
    """Matmul + optional bias for 1D input: [in_dim] -> [out_dim]."""
    inp_2d = network.add_shuffle(inp)
    inp_2d.reshape_dims = (1, in_dim)
    out = _matmul_reduced_precision(network, inp_2d.get_output(0), in_dim, out_dim, weight)
    if bias is not None:
        out = _bias_sum_reduced(network, out, out_dim, bias)
    flat = network.add_shuffle(out)
    flat.reshape_dims = (out_dim,)
    return flat.get_output(0)


def _chunk_6(network, tensor, dim):
    """Split [6*dim] into 6 x [dim]."""
    chunks = []
    for i in range(6):
        s = network.add_slice(tensor, (i * dim,), (dim,), (1,))
        chunks.append(s.get_output(0))
    return chunks


def _chunk_3(network, tensor, dim):
    """Split [3*dim] into 3 x [dim]."""
    chunks = []
    for i in range(3):
        s = network.add_slice(tensor, (i * dim,), (dim,), (1,))
        chunks.append(s.get_output(0))
    return chunks


def _adaln_modulate(network, x, scale, shift, dim, eps_t, seq_len, eps=1e-6):
    """AdaLN: LayerNorm(x) * (1 + scale) + shift.
    x: [seq_len, dim], scale/shift: [dim] (1D from chunk).
    In strongly typed FP16, keep LayerNorm and modulation in FP32; pure FP16
    reductions overflow on FLUX.2 and can collapse the decoded image."""
    fp16_norm = _fp16_compute()
    if fp16_norm:
        normed = graph_ops.add_layer_norm_native(
            network,
            x,
            dim,
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32),
            eps,
            dtype=np.float16,
        )
        normed = _to_fp32(network, normed)
    else:
        # BF16 LayerNorm: mean/var/normalize stay in BF16.
        mean = network.add_reduce(x, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
        centered = network.add_elementwise(
            x, mean.get_output(0), trt.ElementWiseOperation.SUB)
        sq = network.add_elementwise(
            centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD)
        var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
        denom = network.add_elementwise(
            var.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
        sqrt_l = network.add_unary(denom.get_output(0), trt.UnaryOperation.SQRT)
        recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
        normed = network.add_elementwise(
            centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    scale_2d = network.add_shuffle(scale)
    scale_2d.reshape_dims = (1, dim)
    shift_2d = network.add_shuffle(shift)
    shift_2d.reshape_dims = (1, dim)
    scale_value = scale_2d.get_output(0)
    shift_value = shift_2d.get_output(0)

    if fp16_norm:
        scale_value = _to_fp32(network, scale_value)
        shift_value = _to_fp32(network, shift_value)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=np.float32))
    else:
        one_const = _add_constant_reduced(network, (1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_1 = network.add_elementwise(
        one_const, scale_value, trt.ElementWiseOperation.SUM).get_output(0)

    scaled = network.add_elementwise(
        normed, scale_plus_1, trt.ElementWiseOperation.PROD)
    shifted = network.add_elementwise(
        scaled.get_output(0), shift_value, trt.ElementWiseOperation.SUM)
    result = shifted.get_output(0)
    if fp16_norm:
        result = network.add_cast(result, _CAST_DTYPE).get_output(0)
    return result


def _linear_col_parallel(
    network,
    inp,
    in_dim: int,
    out_dim: int,
    weights,
    prefix: str,
    parallel: ParallelConfig,
):
    """Column-parallel linear: output features are rank-local."""
    local_out_dim = out_dim // parallel.tp_size
    weight = weights[f"{prefix}.weight"]
    if parallel.enabled:
        weight = _slice_last_dim(weight, parallel.rank, parallel.tp_size)
    fp8_weight_tn = None if parallel.enabled else weights.get(_fp8_weight_key(prefix))
    out = _matmul_reduced_precision(
        network,
        inp,
        in_dim,
        local_out_dim,
        weight,
        inp_scale=_FP8_SCALES.get(prefix, {}).get("input_scale"),
        wt_scale=_FP8_SCALES.get(prefix, {}).get("weight_scale"),
        fp8_weight_tn=fp8_weight_tn,
    )
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        if parallel.enabled:
            bias = _slice_first_dim(bias, parallel.rank, parallel.tp_size)
        out = _bias_sum_reduced(network, out, local_out_dim, bias)
    return out


def _linear_row_parallel(
    network,
    inp,
    in_dim: int,
    out_dim: int,
    weights,
    prefix: str,
    parallel: ParallelConfig,
):
    """Row-parallel linear: partial outputs are all-reduced across TP ranks."""
    weight = weights[f"{prefix}.weight"]
    if parallel.enabled:
        weight = _slice_first_dim(weight, parallel.rank, parallel.tp_size)
    fp8_weight_tn = None if parallel.enabled else weights.get(_fp8_weight_key(prefix))
    out = _matmul_reduced_precision(
        network,
        inp,
        in_dim,
        out_dim,
        weight,
        inp_scale=_FP8_SCALES.get(prefix, {}).get("input_scale"),
        wt_scale=_FP8_SCALES.get(prefix, {}).get("weight_scale"),
        fp8_weight_tn=fp8_weight_tn,
    )
    if parallel.enabled:
        out = add_all_reduce_sum(network, out, parallel.tp_size)
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = _bias_sum_reduced(network, out, out_dim, bias)
    return out


def _slice_swiglu_linear_in_weight(
    weight: np.ndarray,
    *,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard SwiGLU gate/value columns while preserving their two-half layout."""
    if not parallel.enabled:
        return weight
    gate = _slice_last_dim(weight[:, :ffn_dim], parallel.rank, parallel.tp_size)
    value = _slice_last_dim(weight[:, ffn_dim:2 * ffn_dim], parallel.rank, parallel.tp_size)
    return np.ascontiguousarray(np.concatenate([gate, value], axis=1))


def _slice_swiglu_linear_in_bias(
    bias: np.ndarray,
    *,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard SwiGLU gate/value bias segments while preserving layout."""
    if not parallel.enabled:
        return bias
    gate = _slice_first_dim(bias[:ffn_dim], parallel.rank, parallel.tp_size)
    value = _slice_first_dim(bias[ffn_dim:2 * ffn_dim], parallel.rank, parallel.tp_size)
    return np.ascontiguousarray(np.concatenate([gate, value], axis=0))


def _swiglu_ffn_parallel(network, inp, dim, weights, prefix, parallel: ParallelConfig):
    """Tensor-parallel SwiGLU FFN in reduced precision."""
    fc1_w = weights[f"{prefix}.linear_in.weight"]
    double_ffn_dim = fc1_w.shape[1]  # 2 * ffn_dim
    ffn_dim = double_ffn_dim // 2
    local_ffn_dim = ffn_dim // parallel.tp_size
    local_double_ffn_dim = 2 * local_ffn_dim
    fc1_w = _slice_swiglu_linear_in_weight(
        fc1_w, ffn_dim=ffn_dim, parallel=parallel)
    fp8_weight_tn = (
        None if parallel.enabled
        else weights.get(_fp8_weight_key(f"{prefix}.linear_in"))
    )
    fc1 = _matmul_reduced_precision(
        network,
        inp,
        dim,
        local_double_ffn_dim,
        fc1_w,
        inp_scale=_FP8_SCALES.get(f"{prefix}.linear_in", {}).get("input_scale"),
        wt_scale=_FP8_SCALES.get(f"{prefix}.linear_in", {}).get("weight_scale"),
        fp8_weight_tn=fp8_weight_tn,
    )
    fc1_b = weights.get(f"{prefix}.linear_in.bias")
    if fc1_b is not None:
        fc1_b = _slice_swiglu_linear_in_bias(
            fc1_b, ffn_dim=ffn_dim, parallel=parallel)
        fc1 = _bias_sum_reduced(network, fc1, local_double_ffn_dim, fc1_b)

    seq_len = inp.shape[0]
    x1 = network.add_slice(fc1, (0, 0), (seq_len, local_ffn_dim), (1, 1)).get_output(0)
    x2 = network.add_slice(
        fc1, (0, local_ffn_dim), (seq_len, local_ffn_dim), (1, 1)).get_output(0)

    gate_act = graph_ops.add_activation(network, x1, "silu")
    gated = network.add_elementwise(
        gate_act, x2, trt.ElementWiseOperation.PROD).get_output(0)

    return _linear_row_parallel(
        network,
        gated,
        local_ffn_dim,
        dim,
        weights,
        f"{prefix}.linear_out",
        parallel,
    )


def _slice_flux2_single_fused_qkv_mlp_weight(
    weight: np.ndarray,
    *,
    dim: int,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard FLUX.2 fused Q/K/V/gate/value output columns per rank."""
    if not parallel.enabled:
        return weight
    q = _slice_last_dim(weight[:, 0:dim], parallel.rank, parallel.tp_size)
    k = _slice_last_dim(weight[:, dim:2 * dim], parallel.rank, parallel.tp_size)
    v = _slice_last_dim(weight[:, 2 * dim:3 * dim], parallel.rank, parallel.tp_size)
    gate = _slice_last_dim(
        weight[:, 3 * dim:3 * dim + ffn_dim], parallel.rank, parallel.tp_size)
    value = _slice_last_dim(
        weight[:, 3 * dim + ffn_dim:3 * dim + 2 * ffn_dim],
        parallel.rank,
        parallel.tp_size,
    )
    return np.ascontiguousarray(np.concatenate([q, k, v, gate, value], axis=1))


def _slice_flux2_single_fused_qkv_mlp_bias(
    bias: np.ndarray,
    *,
    dim: int,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard FLUX.2 fused Q/K/V/gate/value bias segments per rank."""
    if not parallel.enabled:
        return bias
    q = _slice_first_dim(bias[0:dim], parallel.rank, parallel.tp_size)
    k = _slice_first_dim(bias[dim:2 * dim], parallel.rank, parallel.tp_size)
    v = _slice_first_dim(bias[2 * dim:3 * dim], parallel.rank, parallel.tp_size)
    gate = _slice_first_dim(
        bias[3 * dim:3 * dim + ffn_dim], parallel.rank, parallel.tp_size)
    value = _slice_first_dim(
        bias[3 * dim + ffn_dim:3 * dim + 2 * ffn_dim],
        parallel.rank,
        parallel.tp_size,
    )
    return np.ascontiguousarray(np.concatenate([q, k, v, gate, value], axis=0))


def _slice_flux2_single_block_out_weight(
    weight: np.ndarray,
    *,
    dim: int,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard rows for FLUX.2 single-block output projection."""
    if not parallel.enabled:
        return weight
    attn_weight = _slice_first_dim(weight[:dim, :], parallel.rank, parallel.tp_size)
    mlp_weight = _slice_first_dim(
        weight[dim:dim + ffn_dim, :], parallel.rank, parallel.tp_size)
    return np.ascontiguousarray(np.concatenate([attn_weight, mlp_weight], axis=0))


def _linear_flux2_single_block_out_parallel(
    network,
    inp,
    *,
    local_dim: int,
    local_ffn_dim: int,
    dim: int,
    ffn_dim: int,
    weights,
    prefix: str,
    parallel: ParallelConfig,
):
    """Row-parallel FLUX.2 single-block projection from local attention+MLP."""
    weight = _slice_flux2_single_block_out_weight(
        weights[f"{prefix}.weight"], dim=dim, ffn_dim=ffn_dim, parallel=parallel)
    fp8_weight_tn = None if parallel.enabled else weights.get(_fp8_weight_key(prefix))
    in_features = local_dim + local_ffn_dim
    out = _matmul_reduced_precision(
        network,
        inp,
        in_features,
        dim,
        weight,
        inp_scale=_FP8_SCALES.get(prefix, {}).get("input_scale"),
        wt_scale=_FP8_SCALES.get(prefix, {}).get("weight_scale"),
        fp8_weight_tn=fp8_weight_tn,
    )
    if parallel.enabled:
        out = add_all_reduce_sum(network, out, parallel.tp_size)
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = _bias_sum_reduced(network, out, dim, bias)
    return out


def _rms_norm_per_head_seq(network, x, num_heads, head_dim, weight, eps_t, seq_len):
    """Per-head RMS norm for [seq_len, dim] tensors with [head_dim] weights.

    Reshapes to [seq_len, num_heads, head_dim], applies RMS norm on head_dim axis,
    then reshapes back to [seq_len, dim].
    """
    if _fp16_compute():
        return graph_ops.add_rms_norm_per_head(
            network,
            x,
            num_heads,
            head_dim,
            weight,
            eps_t,
            dtype=np.float16,
            sequence_length=seq_len,
        )

    dim = num_heads * head_dim

    # Reshape [seq_len, dim] -> [seq_len * num_heads, head_dim]
    reshaped = network.add_shuffle(x)
    reshaped.reshape_dims = (seq_len * num_heads, head_dim)

    # RMS norm on last axis (head_dim)
    reshaped_out = reshaped.get_output(0)
    sq = network.add_elementwise(reshaped_out, reshaped_out, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        reshaped_out, recip.get_output(0), trt.ElementWiseOperation.PROD)

    # Apply per-head gamma [1, head_dim]
    gamma_t = _add_constant_reduced(network, (1, head_dim), weight)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD)

    # Reshape back to [seq_len, dim]
    reshape_back = network.add_shuffle(scaled.get_output(0))
    reshape_back.reshape_dims = (seq_len, dim)
    return reshape_back.get_output(0)


def _apply_native_rope_from_full_cache(
    network, x, cos_vals, sin_vals, num_heads, head_dim, seq_len,
):
    """Apply TRT native RoPE using runtime full-dimension cos/sin rows."""
    return graph_ops.add_apply_rope_native_from_full_cache(
        network, x, num_heads, head_dim, cos_vals, sin_vals,
        seq_len, interleaved=True)


def _mha(network, q, k, v, num_heads, head_dim, seq_len):
    """Multi-head attention via TRT native IAttention."""
    return graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=num_heads, head_dim=head_dim,
        q_seq=seq_len, kv_seq=seq_len)


def _gate_1d(network, x, gate, seq_len):
    """Gate: x * gate (broadcast gate [dim] over [seq_len, dim]).
    gate is [dim] (1D), x is [seq_len, dim] (2D). Reshape gate for broadcast."""
    # Reshape gate from [dim] to [1, dim] for TRT broadcast
    gate_2d = network.add_shuffle(gate)
    gate_2d.reshape_dims = (1, -1)
    return network.add_elementwise(
        x, gate_2d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
