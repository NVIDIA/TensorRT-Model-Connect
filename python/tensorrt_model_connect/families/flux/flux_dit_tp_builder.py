# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX DiT (Diffusion Transformer) engine builder.

Builds a TensorRT engine for the FLUX-specific transformer denoiser,
which has two types of blocks:
  1. Joint transformer blocks (double-stream): image and text attend jointly
  2. Single transformer blocks (single-stream): operate on concatenated tokens

Engine I/O:
    Inputs:
        hidden_states [num_img_tokens, dim] float32 (x_embedder output)
        encoder_hidden_states [text_seq_len, dim] float32 (context_embedder output)
        temb [dim] float32  (combined timestep+text+guidance embedding)
        rotary_cos [total_seq_len, head_dim] float32
        rotary_sin [total_seq_len, head_dim] float32
    Outputs:
        output [num_img_tokens, out_channels] float32

Preprocessor weights (timestep MLP, guidance MLP, text embedder, x_embedder,
context_embedder, RoPE) are handled externally by the runtime.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from ...parallel_config import (
    ParallelConfig,
    _slice_first_dim,
    _slice_last_dim,
    add_all_reduce_sum,
    normalize_parallel_config,
    validate_dit_tp,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_flux_dit_engine(
    weights: "WeightDict",
    *,
    dim: int = 3072,
    num_heads: int = 24,
    num_layers: int = 19,
    num_single_layers: int = 38,
    num_img_tokens: int,
    text_seq_len: int = 512,
    mlp_ratio: float = 4.0,
    eps: float = 1e-6,
    verbose: bool = False,
    parallel_config: ParallelConfig | None = None,
) -> bytes:
    """Build FLUX DiT denoiser TRT engine plan."""
    head_dim = dim // num_heads
    ffn_dim = int(dim * mlp_ratio)  # For GELU-approximate FFN
    total_seq = text_seq_len + num_img_tokens
    parallel = normalize_parallel_config(parallel_config)
    validate_dit_tp(
        dim=dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        parallel=parallel,
        feature="FLUX DiT tensor parallel",
    )
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # --- Inputs ---
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (num_img_tokens, dim))
    encoder_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, dim))
    temb_inp = network.add_input(
        "temb", trt.float32, (dim,))
    rotary_cos = network.add_input(
        "rotary_cos", trt.float32, (total_seq, head_dim))
    rotary_sin = network.add_input(
        "rotary_sin", trt.float32, (total_seq, head_dim))

    # Constants
    eps_np = np.array([eps], dtype=np.float32)
    eps_t = graph_ops.add_constant(network, (1, 1), eps_np)

    # Split RoPE for text and image
    txt_cos = network.add_slice(rotary_cos, (0, 0), (text_seq_len, head_dim), (1, 1)).get_output(0)
    txt_sin = network.add_slice(rotary_sin, (0, 0), (text_seq_len, head_dim), (1, 1)).get_output(0)
    img_cos = network.add_slice(rotary_cos, (text_seq_len, 0), (num_img_tokens, head_dim), (1, 1)).get_output(0)
    img_sin = network.add_slice(rotary_sin, (text_seq_len, 0), (num_img_tokens, head_dim), (1, 1)).get_output(0)

    hidden = hidden_inp
    encoder_hidden = encoder_inp

    # ===================== Joint Transformer Blocks =====================
    for layer_idx in range(num_layers):
        p = f"transformer_blocks.{layer_idx}"

        # --- AdaLN-Zero for image (norm1) ---
        # norm1.linear: SiLU(temb) -> Linear(dim, 6*dim) -> chunk(6)
        norm1_w = weights[f"{p}.norm1.linear.weight"]  # [dim, 6*dim]
        norm1_b = weights[f"{p}.norm1.linear.bias"]    # [6*dim]

        temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu_out = network.add_elementwise(
            temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

        norm1_proj = _matmul_bias_1d(network, temb_silu_out, dim, 6 * dim, norm1_w, norm1_b)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_6(
            network, norm1_proj, dim)

        # AdaLN: LayerNorm(x) * (1 + scale) + shift
        normed_hidden = _adaln_modulate(network, hidden, scale_msa, shift_msa, dim, eps_t, num_img_tokens)

        # --- AdaLN-Zero for text (norm1_context) ---
        ctx_norm1_w = weights[f"{p}.norm1_context.linear.weight"]
        ctx_norm1_b = weights[f"{p}.norm1_context.linear.bias"]

        ctx_norm1_proj = _matmul_bias_1d(network, temb_silu_out, dim, 6 * dim, ctx_norm1_w, ctx_norm1_b)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _chunk_6(
            network, ctx_norm1_proj, dim)

        normed_encoder = _adaln_modulate(
            network, encoder_hidden, c_scale_msa, c_shift_msa, dim, eps_t, text_seq_len)

        # --- Joint Attention ---
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

        # Multi-head attention
        attn_out = _mha(network, q_cat.get_output(0), k_cat.get_output(0),
                        v_cat.get_output(0), local_num_heads, head_dim, total_seq)

        # Split attention output back into text and image
        txt_attn = network.add_slice(attn_out, (0, 0), (text_seq_len, local_dim), (1, 1)).get_output(0)
        img_attn = network.add_slice(attn_out, (text_seq_len, 0), (num_img_tokens, local_dim), (1, 1)).get_output(0)

        # Image output projection + gate + residual
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

        # --- Image FFN ---
        normed_ff = _layernorm_modulate(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t, num_img_tokens)
        ff_out = _gelu_ffn(network, normed_ff, dim, weights, f"{p}.ff", parallel)
        ff_gated = _gate_1d(network, ff_out, gate_mlp, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Text FFN ---
        normed_ctx_ff = _layernorm_modulate(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t, text_seq_len)
        ctx_ff_out = _gelu_ffn(network, normed_ctx_ff, dim, weights, f"{p}.ff_context", parallel)
        ctx_ff_gated = _gate_1d(network, ctx_ff_out, c_gate_mlp, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, ctx_ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

    # ===================== Single Transformer Blocks =====================
    for layer_idx in range(num_single_layers):
        p = f"single_transformer_blocks.{layer_idx}"

        # Concatenate text + image
        cat_hidden = network.add_concatenation([encoder_hidden, hidden])
        cat_hidden.axis = 0  # [total_seq, dim]
        residual = cat_hidden.get_output(0)

        # AdaLN-Zero Single: SiLU(temb) -> Linear(dim, 3*dim) -> chunk(3)
        norm_w = weights[f"{p}.norm.linear.weight"]
        norm_b = weights[f"{p}.norm.linear.bias"]

        temb_silu2 = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu2_out = network.add_elementwise(
            temb_inp, temb_silu2.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

        norm_proj = _matmul_bias_1d(network, temb_silu2_out, dim, 3 * dim, norm_w, norm_b)
        shift_msa_s, scale_msa_s, gate_msa_s = _chunk_3(network, norm_proj, dim)

        normed_cat = _adaln_modulate(
            network, residual, scale_msa_s, shift_msa_s, dim, eps_t, total_seq)

        # Parallel MLP: proj_mlp -> GELU_tanh
        mlp_hidden = _linear_col_parallel(
            network, normed_cat, dim, ffn_dim, weights, f"{p}.proj_mlp", parallel)
        mlp_hidden = graph_ops.add_gelu_tanh(network, mlp_hidden)

        # Self-attention on full sequence (text + image)
        q_s = _linear_col_parallel(
            network, normed_cat, dim, dim, weights, f"{p}.attn.to_q", parallel)
        k_s = _linear_col_parallel(
            network, normed_cat, dim, dim, weights, f"{p}.attn.to_k", parallel)
        v_s = _linear_col_parallel(
            network, normed_cat, dim, dim, weights, f"{p}.attn.to_v", parallel)

        # QK norm
        q_s = _rms_norm_per_head_seq(network, q_s, local_num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, total_seq)
        k_s = _rms_norm_per_head_seq(network, k_s, local_num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, total_seq)

        # Apply RoPE (full sequence: text + image cos/sin)
        q_s = _apply_native_rope_from_full_cache(
            network, q_s, rotary_cos, rotary_sin, local_num_heads, head_dim, total_seq)
        k_s = _apply_native_rope_from_full_cache(
            network, k_s, rotary_cos, rotary_sin, local_num_heads, head_dim, total_seq)

        attn_out_s = _mha(network, q_s, k_s, v_s, local_num_heads, head_dim, total_seq)

        # Concatenate attn + mlp -> proj_out
        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 1  # [total_seq, dim + ffn_dim]

        combined = _linear_single_block_out_parallel(
            network, cat_attn_mlp.get_output(0), local_dim, local_ffn_dim,
            dim, ffn_dim, weights, f"{p}.proj_out", parallel)

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
    final_norm_b = weights["norm_out.linear.bias"]

    temb_silu_f = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_f_out = network.add_elementwise(
        temb_inp, temb_silu_f.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    final_proj = _matmul_bias_1d(network, temb_silu_f_out, dim, 2 * dim, final_norm_w, final_norm_b)
    final_scale = network.add_slice(final_proj, (0,), (dim,), (1,)).get_output(0)
    final_shift = network.add_slice(final_proj, (dim,), (dim,), (1,)).get_output(0)

    output = _adaln_modulate(
        network, hidden, final_scale, final_shift, dim, eps_t, num_img_tokens)

    # proj_out: [num_img_tokens, dim] -> [num_img_tokens, out_channels]
    proj_out_w = weights["proj_out.weight"]
    out_channels = proj_out_w.shape[1]
    output = graph_ops.add_matmul_rhs_constant(
        network, output, dim, out_channels, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_channels, proj_out_b)

    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)

    tp_suffix = (
        f", tp={parallel.tp_size}, rank={parallel.rank}"
        if parallel.enabled else "")
    print(f"[flux-dit] Building TRT engine "
          f"(dim={dim}, joint={num_layers}, single={num_single_layers}, "
          f"img_tokens={num_img_tokens}, text_seq={text_seq_len}{tp_suffix}) ...",
          file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for FLUX DiT")
    return bytes(plan)


# ============================================================================
# Helper functions
# ============================================================================

def _matmul_bias_1d(network, inp, in_dim, out_dim, weight, bias):
    """Matmul + bias for 1D input: [in_dim] -> [out_dim]."""
    inp_2d = network.add_shuffle(inp)
    inp_2d.reshape_dims = (1, in_dim)
    out = graph_ops.add_matmul_rhs_constant(
        network, inp_2d.get_output(0), in_dim, out_dim, weight)
    out = graph_ops.add_bias_sum(network, out, out_dim, bias)
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


def _adaln_modulate(network, x, scale, shift, dim, eps_t, seq_len):
    """AdaLN: LayerNorm(x) * (1 + scale) + shift.
    x: [seq_len, dim], scale/shift: [dim] (1D from chunk)."""
    normed = graph_ops.add_layer_norm_no_affine(network, x, dim, eps_t)

    # Reshape scale/shift from [dim] to [1, dim] for broadcast with [seq_len, dim]
    scale_2d = network.add_shuffle(scale)
    scale_2d.reshape_dims = (1, dim)
    shift_2d = network.add_shuffle(shift)
    shift_2d.reshape_dims = (1, dim)

    one_const = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_1 = network.add_elementwise(
        one_const, scale_2d.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    scaled = network.add_elementwise(
        normed, scale_plus_1, trt.ElementWiseOperation.PROD)
    shifted = network.add_elementwise(
        scaled.get_output(0), shift_2d.get_output(0), trt.ElementWiseOperation.SUM)
    return shifted.get_output(0)


def _layernorm_modulate(network, x, scale, shift, dim, eps_t, seq_len):
    """LayerNorm(x) * (1 + scale) + shift."""
    return _adaln_modulate(network, x, scale, shift, dim, eps_t, seq_len)


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
    out = graph_ops.add_matmul_rhs_constant(network, inp, in_dim, local_out_dim, weight)
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        if parallel.enabled:
            bias = _slice_first_dim(bias, parallel.rank, parallel.tp_size)
        out = graph_ops.add_bias_sum(network, out, local_out_dim, bias)
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
    out = graph_ops.add_matmul_rhs_constant(network, inp, in_dim, out_dim, weight)
    if parallel.enabled:
        out = add_all_reduce_sum(network, out, parallel.tp_size)
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias)
    return out


def _slice_single_block_proj_out_weight(
    weight: np.ndarray,
    *,
    dim: int,
    ffn_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Shard the single-block proj_out rows across attention and MLP halves."""
    if not parallel.enabled:
        return weight
    attn_weight = _slice_first_dim(weight[:dim, :], parallel.rank, parallel.tp_size)
    mlp_weight = _slice_first_dim(weight[dim:dim + ffn_dim, :], parallel.rank, parallel.tp_size)
    return np.ascontiguousarray(np.concatenate([attn_weight, mlp_weight], axis=0))


def _linear_single_block_out_parallel(
    network,
    inp,
    local_dim: int,
    local_ffn_dim: int,
    dim: int,
    ffn_dim: int,
    weights,
    prefix: str,
    parallel: ParallelConfig,
):
    """Single-block row-parallel projection from local attention+MLP features."""
    weight = _slice_single_block_proj_out_weight(
        weights[f"{prefix}.weight"], dim=dim, ffn_dim=ffn_dim, parallel=parallel)
    in_features = local_dim + local_ffn_dim
    out = graph_ops.add_matmul_rhs_constant(network, inp, in_features, dim, weight)
    if parallel.enabled:
        out = add_all_reduce_sum(network, out, parallel.tp_size)
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, dim, bias)
    return out


def _rms_norm_per_head_seq(network, x, num_heads, head_dim, weight, eps_t, seq_len):
    """Per-head RMS norm for [seq_len, dim] tensors with [head_dim] weights.

    Reshapes to [seq_len, num_heads, head_dim], applies RMS norm on head_dim axis,
    then reshapes back to [seq_len, dim].
    """
    return graph_ops.add_rms_norm_per_head(
        network, x, num_heads, head_dim, weight, eps_t,
        sequence_length=seq_len)


def _apply_native_rope_from_full_cache(
    network, x, cos_vals, sin_vals, num_heads, head_dim, seq_len,
):
    """Apply TRT native RoPE using runtime full-dimension cos/sin rows."""
    return graph_ops.add_apply_rope_native_from_full_cache(
        network, x, num_heads, head_dim, cos_vals, sin_vals,
        seq_len, interleaved=True)


def _mha(network, q, k, v, num_heads, head_dim, seq_len):
    """Multi-head attention: returns [seq_len, dim]."""
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


def _gelu_ffn(network, inp, dim, weights, prefix, parallel: ParallelConfig):
    """GELU-approximate FFN: Linear(dim, ffn_dim) -> GELU -> Linear(ffn_dim, dim)."""
    fc1_w = weights[f"{prefix}.net.0.proj.weight"]
    ffn_dim = fc1_w.shape[1]
    local_ffn_dim = ffn_dim // parallel.tp_size
    if parallel.enabled:
        fc1_w = _slice_last_dim(fc1_w, parallel.rank, parallel.tp_size)

    fc1 = graph_ops.add_matmul_rhs_constant(network, inp, dim, local_ffn_dim, fc1_w)
    fc1_b = weights.get(f"{prefix}.net.0.proj.bias")
    if fc1_b is not None:
        if parallel.enabled:
            fc1_b = _slice_first_dim(fc1_b, parallel.rank, parallel.tp_size)
        fc1 = graph_ops.add_bias_sum(network, fc1, local_ffn_dim, fc1_b)

    act = graph_ops.add_gelu_tanh(network, fc1)

    fc2_w = weights[f"{prefix}.net.2.weight"]
    if parallel.enabled:
        fc2_w = _slice_first_dim(fc2_w, parallel.rank, parallel.tp_size)
    fc2 = graph_ops.add_matmul_rhs_constant(network, act, local_ffn_dim, dim, fc2_w)
    if parallel.enabled:
        fc2 = add_all_reduce_sum(network, fc2, parallel.tp_size)
    fc2_b = weights.get(f"{prefix}.net.2.bias")
    if fc2_b is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, dim, fc2_b)
    return fc2
