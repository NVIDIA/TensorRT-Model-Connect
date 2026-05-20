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

from ... import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


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
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
) -> bytes:
    """Build FLUX DiT denoiser TRT engine plan."""
    if max_batch_size > 1:
        return _build_flux_dit_engine_batched(
            weights,
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_img_tokens=num_img_tokens,
            text_seq_len=text_seq_len,
            mlp_ratio=mlp_ratio,
            eps=eps,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
            verbose=verbose,
        )

    head_dim = dim // num_heads
    ffn_dim = int(dim * mlp_ratio)  # For GELU-approximate FFN
    total_seq = text_seq_len + num_img_tokens

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
        q_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_q")
        k_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_k")
        v_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_v")

        # Text QKV (added projections)
        q_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_q_proj")
        k_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_k_proj")
        v_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_v_proj")

        # QK norm
        q_img = _rms_norm_per_head_seq(network, q_img, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, num_img_tokens)
        k_img = _rms_norm_per_head_seq(network, k_img, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, num_img_tokens)
        q_txt = _rms_norm_per_head_seq(network, q_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_q.weight"], eps_t, text_seq_len)
        k_txt = _rms_norm_per_head_seq(network, k_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_k.weight"], eps_t, text_seq_len)

        # Apply RoPE to image Q, K
        q_img = _apply_native_rope_from_full_cache(
            network, q_img, img_cos, img_sin, num_heads, head_dim, num_img_tokens)
        k_img = _apply_native_rope_from_full_cache(
            network, k_img, img_cos, img_sin, num_heads, head_dim, num_img_tokens)

        # Apply RoPE to text Q, K
        q_txt = _apply_native_rope_from_full_cache(
            network, q_txt, txt_cos, txt_sin, num_heads, head_dim, text_seq_len)
        k_txt = _apply_native_rope_from_full_cache(
            network, k_txt, txt_cos, txt_sin, num_heads, head_dim, text_seq_len)

        # Concatenate: [text, image] for joint attention
        q_cat = network.add_concatenation([q_txt, q_img])
        q_cat.axis = 0  # [total_seq, dim]
        k_cat = network.add_concatenation([k_txt, k_img])
        k_cat.axis = 0
        v_cat = network.add_concatenation([v_txt, v_img])
        v_cat.axis = 0

        # Multi-head attention
        attn_out = _mha(network, q_cat.get_output(0), k_cat.get_output(0),
                        v_cat.get_output(0), num_heads, head_dim, total_seq)

        # Split attention output back into text and image
        txt_attn = network.add_slice(attn_out, (0, 0), (text_seq_len, dim), (1, 1)).get_output(0)
        img_attn = network.add_slice(attn_out, (text_seq_len, 0), (num_img_tokens, dim), (1, 1)).get_output(0)

        # Image output projection + gate + residual
        img_attn_proj = _linear(network, img_attn, dim, dim, weights, f"{p}.attn.to_out.0")
        img_attn_gated = _gate_1d(network, img_attn_proj, gate_msa, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, img_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Text output projection + gate + residual
        txt_attn_proj = _linear(network, txt_attn, dim, dim, weights, f"{p}.attn.to_add_out")
        txt_attn_gated = _gate_1d(network, txt_attn_proj, c_gate_msa, text_seq_len)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, txt_attn_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Image FFN ---
        normed_ff = _layernorm_modulate(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t, num_img_tokens)
        ff_out = _gelu_ffn(network, normed_ff, dim, weights, f"{p}.ff")
        ff_gated = _gate_1d(network, ff_out, gate_mlp, num_img_tokens)
        hidden = network.add_elementwise(
            hidden, ff_gated,
            trt.ElementWiseOperation.SUM).get_output(0)

        # --- Text FFN ---
        normed_ctx_ff = _layernorm_modulate(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t, text_seq_len)
        ctx_ff_out = _gelu_ffn(network, normed_ctx_ff, dim, weights, f"{p}.ff_context")
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
        mlp_hidden = graph_ops.add_matmul_rhs_constant(
            network, normed_cat, dim, ffn_dim,
            weights[f"{p}.proj_mlp.weight"])
        mlp_b = weights.get(f"{p}.proj_mlp.bias")
        if mlp_b is not None:
            mlp_hidden = graph_ops.add_bias_sum(network, mlp_hidden, ffn_dim, mlp_b)
        mlp_hidden = graph_ops.add_gelu_tanh(network, mlp_hidden)

        # Self-attention on full sequence (text + image)
        q_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_q")
        k_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_k")
        v_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_v")

        # QK norm
        q_s = _rms_norm_per_head_seq(network, q_s, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"], eps_t, total_seq)
        k_s = _rms_norm_per_head_seq(network, k_s, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"], eps_t, total_seq)

        # Apply RoPE (full sequence: text + image cos/sin)
        q_s = _apply_native_rope_from_full_cache(
            network, q_s, rotary_cos, rotary_sin, num_heads, head_dim, total_seq)
        k_s = _apply_native_rope_from_full_cache(
            network, k_s, rotary_cos, rotary_sin, num_heads, head_dim, total_seq)

        attn_out_s = _mha(network, q_s, k_s, v_s, num_heads, head_dim, total_seq)

        # Concatenate attn + mlp -> proj_out
        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 1  # [total_seq, dim + ffn_dim]

        proj_out_w = weights[f"{p}.proj_out.weight"]
        in_features = dim + ffn_dim
        combined = graph_ops.add_matmul_rhs_constant(
            network, cat_attn_mlp.get_output(0), in_features, dim, proj_out_w)
        proj_out_b = weights.get(f"{p}.proj_out.bias")
        if proj_out_b is not None:
            combined = graph_ops.add_bias_sum(network, combined, dim, proj_out_b)

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

    print(f"[flux-dit] Building TRT engine "
          f"(dim={dim}, joint={num_layers}, single={num_single_layers}, "
          f"img_tokens={num_img_tokens}, text_seq={text_seq_len}) ...",
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


def _linear(network, inp, in_dim, out_dim, weights, prefix):
    """Linear projection with optional bias."""
    out = graph_ops.add_matmul_rhs_constant(
        network, inp, in_dim, out_dim, weights[f"{prefix}.weight"])
    b = weights.get(f"{prefix}.bias")
    if b is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, b)
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


def _gelu_ffn(network, inp, dim, weights, prefix):
    """GELU-approximate FFN: Linear(dim, ffn_dim) -> GELU -> Linear(ffn_dim, dim)."""
    fc1_w = weights[f"{prefix}.net.0.proj.weight"]
    ffn_dim = fc1_w.shape[1]

    fc1 = graph_ops.add_matmul_rhs_constant(network, inp, dim, ffn_dim, fc1_w)
    fc1_b = weights.get(f"{prefix}.net.0.proj.bias")
    if fc1_b is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, ffn_dim, fc1_b)

    act = graph_ops.add_gelu_tanh(network, fc1)

    fc2_w = weights[f"{prefix}.net.2.weight"]
    fc2 = graph_ops.add_matmul_rhs_constant(network, act, ffn_dim, dim, fc2_w)
    fc2_b = weights.get(f"{prefix}.net.2.bias")
    if fc2_b is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, dim, fc2_b)
    return fc2


def _dynamic_batch_shape(network, reference, tail: tuple[int, ...]):
    """Shape tensor [B, *tail] using dim 0 from a dynamic-batch tensor."""
    ref_shape = network.add_shape(reference).get_output(0)
    batch = network.add_slice(ref_shape, start=(0,), shape=(1,), stride=(1,))
    tail_t = graph_ops.add_constant(
        network, (len(tail),), np.asarray(tail, dtype=np.int64), dtype=np.int64)
    target = network.add_concatenation([batch.get_output(0), tail_t])
    target.axis = 0
    return target.get_output(0)


def _slice_batched_vector(network, x, start_width: int, width: int):
    """Slice [B, D] along D while preserving runtime-dynamic B."""
    s = network.add_slice(
        x, start=(0, start_width), shape=(0, 0), stride=(1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (width,)))
    return s.get_output(0)


def _slice_batched_sequence(network, x, start_seq: int, length: int, width: int):
    """Slice [B, S, D] along S while preserving runtime-dynamic B."""
    s = network.add_slice(
        x, start=(0, start_seq, 0), shape=(0, 0, 0), stride=(1, 1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (length, width)))
    return s.get_output(0)


def _slice_batched_rope_half(network, rope, seq_len: int, half: int):
    """Slice interleaved full-dim RoPE cache [B, S, D] to [B, S, D/2]."""
    s = network.add_slice(
        rope, start=(0, 0, 0), shape=(0, 0, 0), stride=(1, 1, 2))
    s.set_input(2, _dynamic_batch_shape(network, rope, (seq_len, half)))
    return s.get_output(0)


def _slice_batched_complex_part(network, x, seq_len: int, num_heads: int,
                                half: int, complex_index: int):
    """Slice real/imag part from [B, S, H, D/2, 2] to [B, S, H, D/2]."""
    s = network.add_slice(
        x, start=(0, 0, 0, 0, complex_index), shape=(0, 0, 0, 0, 0),
        stride=(1, 1, 1, 1, 1))
    s.set_input(2, _dynamic_batch_shape(network, x, (seq_len, num_heads, half, 1)))
    r = network.add_shuffle(s.get_output(0))
    r.reshape_dims = (-1, seq_len, num_heads, half)
    return r.get_output(0)


def _reshape_batched_rows_to_heads_4d(network, x, num_heads: int, head_dim: int,
                                      seq_len: int):
    """[B, S, H*D] -> [B, H, S, D]."""
    r = network.add_shuffle(x)
    r.reshape_dims = (-1, seq_len, num_heads, head_dim)
    r.second_transpose = trt.Permutation([0, 2, 1, 3])
    return r.get_output(0)


def _reshape_heads_4d_to_batched_rows(network, x, num_heads: int, head_dim: int,
                                      seq_len: int):
    """[B, H, S, D] -> [B, S, H*D]."""
    r = network.add_shuffle(x)
    r.first_transpose = trt.Permutation([0, 2, 1, 3])
    r.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return r.get_output(0)


def _mha_batched(network, q, k, v, num_heads: int, head_dim: int, seq_len: int):
    """Multi-head attention for [B, S, H*D] tensors."""
    q_4d = _reshape_batched_rows_to_heads_4d(network, q, num_heads, head_dim, seq_len)
    k_4d = _reshape_batched_rows_to_heads_4d(network, k, num_heads, head_dim, seq_len)
    v_4d = _reshape_batched_rows_to_heads_4d(network, v, num_heads, head_dim, seq_len)
    ctx_4d = graph_ops.add_attention_core(
        network, q_4d, k_4d, v_4d, causal=False, mask=None,
        scale=float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0)
    return _reshape_heads_4d_to_batched_rows(network, ctx_4d, num_heads, head_dim, seq_len)


def _apply_rope_batched_from_full_cache(network, x, cos_t, sin_t, num_heads: int,
                                        head_dim: int, seq_len: int):
    """Apply interleaved RoPE to [B, S, H*D] with [B, S, D] caches."""
    half = head_dim // 2
    x_pairs_r = network.add_shuffle(x)
    x_pairs_r.reshape_dims = (-1, seq_len, num_heads, half, 2)
    x_pairs = x_pairs_r.get_output(0)

    x_real = _slice_batched_complex_part(network, x_pairs, seq_len, num_heads, half, 0)
    x_imag = _slice_batched_complex_part(network, x_pairs, seq_len, num_heads, half, 1)

    cos_half = _slice_batched_rope_half(network, cos_t, seq_len, half)
    sin_half = _slice_batched_rope_half(network, sin_t, seq_len, half)
    cos_4d = network.add_shuffle(cos_half)
    cos_4d.reshape_dims = (-1, seq_len, 1, half)
    sin_4d = network.add_shuffle(sin_half)
    sin_4d.reshape_dims = (-1, seq_len, 1, half)

    r_cos = network.add_elementwise(
        x_real, cos_4d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    i_sin = network.add_elementwise(
        x_imag, sin_4d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    new_real = network.add_elementwise(
        r_cos, i_sin, trt.ElementWiseOperation.SUB).get_output(0)

    r_sin = network.add_elementwise(
        x_real, sin_4d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    i_cos = network.add_elementwise(
        x_imag, cos_4d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
    new_imag = network.add_elementwise(
        r_sin, i_cos, trt.ElementWiseOperation.SUM).get_output(0)

    nr = network.add_shuffle(new_real)
    nr.reshape_dims = (-1, seq_len, num_heads, half, 1)
    ni = network.add_shuffle(new_imag)
    ni.reshape_dims = (-1, seq_len, num_heads, half, 1)
    cat = network.add_concatenation([nr.get_output(0), ni.get_output(0)])
    cat.axis = 4

    flat = network.add_shuffle(cat.get_output(0))
    flat.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return flat.get_output(0)


def _layer_norm_last_dim_no_affine_batched(network, x, eps_t):
    """LayerNorm without affine over final dim for [B, S, D]."""
    mean = network.add_reduce(x, trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
    centered = network.add_elementwise(
        x, mean.get_output(0), trt.ElementWiseOperation.SUB).get_output(0)
    sq = network.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD)
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2,
                             keep_dims=True)
    var_eps = network.add_elementwise(var.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    std = network.add_unary(var_eps.get_output(0), trt.UnaryOperation.SQRT)
    inv_std = network.add_unary(std.get_output(0), trt.UnaryOperation.RECIP)
    return network.add_elementwise(
        centered, inv_std.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)


def _matmul_bias_batched(network, inp, in_dim, out_dim, weight, bias):
    """Matmul + bias for [B, in_dim] -> [B, out_dim]."""
    out = graph_ops.add_matmul_rhs_constant(network, inp, in_dim, out_dim, weight)
    return graph_ops.add_bias_sum(network, out, out_dim, bias)


def _chunk_batched(network, tensor, chunks: int, dim: int):
    """Split [B, chunks*dim] into ``chunks`` tensors of [B, dim]."""
    out = []
    for i in range(chunks):
        out.append(_slice_batched_vector(network, tensor, i * dim, dim))
    return out


def _adaln_modulate_batched(network, x, scale, shift, dim, eps_t):
    """AdaLN: LayerNorm(x) * (1 + scale) + shift for [B, S, D]."""
    normed = _layer_norm_last_dim_no_affine_batched(network, x, eps_t)

    scale_3d = network.add_shuffle(scale)
    scale_3d.reshape_dims = (-1, 1, dim)
    shift_3d = network.add_shuffle(shift)
    shift_3d.reshape_dims = (-1, 1, dim)

    one_const = graph_ops.add_constant(network, (1, 1, 1), np.array([1.0], dtype=np.float32))
    scale_plus_1 = network.add_elementwise(
        one_const, scale_3d.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    scaled = network.add_elementwise(normed, scale_plus_1, trt.ElementWiseOperation.PROD)
    shifted = network.add_elementwise(
        scaled.get_output(0), shift_3d.get_output(0), trt.ElementWiseOperation.SUM)
    return shifted.get_output(0)


def _layernorm_modulate_batched(network, x, scale, shift, dim, eps_t):
    """LayerNorm(x) * (1 + scale) + shift for [B, S, D]."""
    return _adaln_modulate_batched(network, x, scale, shift, dim, eps_t)


def _gate_batched(network, x, gate, dim: int):
    """Gate [B, S, D] with per-sample gate [B, D]."""
    gate_3d = network.add_shuffle(gate)
    gate_3d.reshape_dims = (-1, 1, dim)
    return network.add_elementwise(
        x, gate_3d.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)


def _build_flux_dit_engine_batched(
    weights: "WeightDict",
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    num_single_layers: int,
    num_img_tokens: int,
    text_seq_len: int,
    mlp_ratio: float,
    eps: float,
    max_batch_size: int,
    opt_batch_size: int | None,
    verbose: bool,
) -> bytes:
    """Build a dynamic-leading-batch FLUX.1 DiT TRT engine."""
    from ...engine_builder import add_dynamic_batch_profile

    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1 (got {max_batch_size})")
    opt_batch = min(max_batch_size, 4) if opt_batch_size is None else opt_batch_size

    head_dim = dim // num_heads
    ffn_dim = int(dim * mlp_ratio)
    total_seq = text_seq_len + num_img_tokens

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (-1, num_img_tokens, dim))
    encoder_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (-1, text_seq_len, dim))
    temb_inp = network.add_input("temb", trt.float32, (-1, dim))
    rotary_cos = network.add_input("rotary_cos", trt.float32, (-1, total_seq, head_dim))
    rotary_sin = network.add_input("rotary_sin", trt.float32, (-1, total_seq, head_dim))

    add_dynamic_batch_profile(
        builder,
        config,
        network,
        input_names=[
            "hidden_states",
            "encoder_hidden_states",
            "temb",
            "rotary_cos",
            "rotary_sin",
        ],
        max_batch=max_batch_size,
        opt_batch=opt_batch,
        static_shape={
            "hidden_states": (num_img_tokens, dim),
            "encoder_hidden_states": (text_seq_len, dim),
            "temb": (dim,),
            "rotary_cos": (total_seq, head_dim),
            "rotary_sin": (total_seq, head_dim),
        },
    )

    eps_t = graph_ops.add_constant(network, (1, 1, 1), np.array([eps], dtype=np.float32))

    txt_cos = _slice_batched_sequence(network, rotary_cos, 0, text_seq_len, head_dim)
    txt_sin = _slice_batched_sequence(network, rotary_sin, 0, text_seq_len, head_dim)
    img_cos = _slice_batched_sequence(network, rotary_cos, text_seq_len, num_img_tokens, head_dim)
    img_sin = _slice_batched_sequence(network, rotary_sin, text_seq_len, num_img_tokens, head_dim)

    hidden = hidden_inp
    encoder_hidden = encoder_inp

    for layer_idx in range(num_layers):
        p = f"transformer_blocks.{layer_idx}"

        norm1_w = weights[f"{p}.norm1.linear.weight"]
        norm1_b = weights[f"{p}.norm1.linear.bias"]
        temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu_out = network.add_elementwise(
            temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

        norm1_proj = _matmul_bias_batched(network, temb_silu_out, dim, 6 * dim,
                                          norm1_w, norm1_b)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_batched(
            network, norm1_proj, 6, dim)
        normed_hidden = _adaln_modulate_batched(network, hidden, scale_msa, shift_msa,
                                                dim, eps_t)

        ctx_norm1_w = weights[f"{p}.norm1_context.linear.weight"]
        ctx_norm1_b = weights[f"{p}.norm1_context.linear.bias"]
        ctx_norm1_proj = _matmul_bias_batched(
            network, temb_silu_out, dim, 6 * dim, ctx_norm1_w, ctx_norm1_b)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            _chunk_batched(network, ctx_norm1_proj, 6, dim))
        normed_encoder = _adaln_modulate_batched(
            network, encoder_hidden, c_scale_msa, c_shift_msa, dim, eps_t)

        q_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_q")
        k_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_k")
        v_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_v")

        q_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_q_proj")
        k_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_k_proj")
        v_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_v_proj")

        q_img = graph_ops.add_rms_norm_per_head_batched(
            network, q_img, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"],
            eps_t, sequence_length=num_img_tokens)
        k_img = graph_ops.add_rms_norm_per_head_batched(
            network, k_img, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"],
            eps_t, sequence_length=num_img_tokens)
        q_txt = graph_ops.add_rms_norm_per_head_batched(
            network, q_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_q.weight"],
            eps_t, sequence_length=text_seq_len)
        k_txt = graph_ops.add_rms_norm_per_head_batched(
            network, k_txt, num_heads, head_dim, weights[f"{p}.attn.norm_added_k.weight"],
            eps_t, sequence_length=text_seq_len)

        q_img = _apply_rope_batched_from_full_cache(
            network, q_img, img_cos, img_sin, num_heads, head_dim, num_img_tokens)
        k_img = _apply_rope_batched_from_full_cache(
            network, k_img, img_cos, img_sin, num_heads, head_dim, num_img_tokens)
        q_txt = _apply_rope_batched_from_full_cache(
            network, q_txt, txt_cos, txt_sin, num_heads, head_dim, text_seq_len)
        k_txt = _apply_rope_batched_from_full_cache(
            network, k_txt, txt_cos, txt_sin, num_heads, head_dim, text_seq_len)

        q_cat = network.add_concatenation([q_txt, q_img])
        q_cat.axis = 1
        k_cat = network.add_concatenation([k_txt, k_img])
        k_cat.axis = 1
        v_cat = network.add_concatenation([v_txt, v_img])
        v_cat.axis = 1

        attn_out = _mha_batched(
            network, q_cat.get_output(0), k_cat.get_output(0), v_cat.get_output(0),
            num_heads, head_dim, total_seq)

        txt_attn = _slice_batched_sequence(network, attn_out, 0, text_seq_len, dim)
        img_attn = _slice_batched_sequence(network, attn_out, text_seq_len,
                                           num_img_tokens, dim)

        img_attn_proj = _linear(network, img_attn, dim, dim, weights, f"{p}.attn.to_out.0")
        img_attn_gated = _gate_batched(network, img_attn_proj, gate_msa, dim)
        hidden = network.add_elementwise(
            hidden, img_attn_gated, trt.ElementWiseOperation.SUM).get_output(0)

        txt_attn_proj = _linear(network, txt_attn, dim, dim, weights, f"{p}.attn.to_add_out")
        txt_attn_gated = _gate_batched(network, txt_attn_proj, c_gate_msa, dim)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, txt_attn_gated, trt.ElementWiseOperation.SUM).get_output(0)

        normed_ff = _layernorm_modulate_batched(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t)
        ff_out = _gelu_ffn(network, normed_ff, dim, weights, f"{p}.ff")
        ff_gated = _gate_batched(network, ff_out, gate_mlp, dim)
        hidden = network.add_elementwise(
            hidden, ff_gated, trt.ElementWiseOperation.SUM).get_output(0)

        normed_ctx_ff = _layernorm_modulate_batched(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t)
        ctx_ff_out = _gelu_ffn(network, normed_ctx_ff, dim, weights, f"{p}.ff_context")
        ctx_ff_gated = _gate_batched(network, ctx_ff_out, c_gate_mlp, dim)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, ctx_ff_gated, trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(num_single_layers):
        p = f"single_transformer_blocks.{layer_idx}"

        cat_hidden = network.add_concatenation([encoder_hidden, hidden])
        cat_hidden.axis = 1
        residual = cat_hidden.get_output(0)

        norm_w = weights[f"{p}.norm.linear.weight"]
        norm_b = weights[f"{p}.norm.linear.bias"]
        temb_silu2 = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu2_out = network.add_elementwise(
            temb_inp, temb_silu2.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

        norm_proj = _matmul_bias_batched(network, temb_silu2_out, dim, 3 * dim,
                                         norm_w, norm_b)
        shift_msa_s, scale_msa_s, gate_msa_s = _chunk_batched(network, norm_proj, 3, dim)
        normed_cat = _adaln_modulate_batched(
            network, residual, scale_msa_s, shift_msa_s, dim, eps_t)

        mlp_hidden = graph_ops.add_matmul_rhs_constant(
            network, normed_cat, dim, ffn_dim, weights[f"{p}.proj_mlp.weight"])
        mlp_b = weights.get(f"{p}.proj_mlp.bias")
        if mlp_b is not None:
            mlp_hidden = graph_ops.add_bias_sum(network, mlp_hidden, ffn_dim, mlp_b)
        mlp_hidden = graph_ops.add_gelu_tanh(network, mlp_hidden)

        q_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_q")
        k_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_k")
        v_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_v")

        q_s = graph_ops.add_rms_norm_per_head_batched(
            network, q_s, num_heads, head_dim, weights[f"{p}.attn.norm_q.weight"],
            eps_t, sequence_length=total_seq)
        k_s = graph_ops.add_rms_norm_per_head_batched(
            network, k_s, num_heads, head_dim, weights[f"{p}.attn.norm_k.weight"],
            eps_t, sequence_length=total_seq)

        q_s = _apply_rope_batched_from_full_cache(
            network, q_s, rotary_cos, rotary_sin, num_heads, head_dim, total_seq)
        k_s = _apply_rope_batched_from_full_cache(
            network, k_s, rotary_cos, rotary_sin, num_heads, head_dim, total_seq)

        attn_out_s = _mha_batched(network, q_s, k_s, v_s, num_heads, head_dim, total_seq)

        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 2

        proj_out_w = weights[f"{p}.proj_out.weight"]
        in_features = dim + ffn_dim
        combined = graph_ops.add_matmul_rhs_constant(
            network, cat_attn_mlp.get_output(0), in_features, dim, proj_out_w)
        proj_out_b = weights.get(f"{p}.proj_out.bias")
        if proj_out_b is not None:
            combined = graph_ops.add_bias_sum(network, combined, dim, proj_out_b)

        gated_s = _gate_batched(network, combined, gate_msa_s, dim)
        cat_hidden_out = network.add_elementwise(
            residual, gated_s, trt.ElementWiseOperation.SUM).get_output(0)

        encoder_hidden = _slice_batched_sequence(network, cat_hidden_out, 0,
                                                 text_seq_len, dim)
        hidden = _slice_batched_sequence(network, cat_hidden_out, text_seq_len,
                                         num_img_tokens, dim)

    final_norm_w = weights["norm_out.linear.weight"]
    final_norm_b = weights["norm_out.linear.bias"]

    temb_silu_f = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_f_out = network.add_elementwise(
        temb_inp, temb_silu_f.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)

    final_proj = _matmul_bias_batched(network, temb_silu_f_out, dim, 2 * dim,
                                      final_norm_w, final_norm_b)
    final_scale = _slice_batched_vector(network, final_proj, 0, dim)
    final_shift = _slice_batched_vector(network, final_proj, dim, dim)

    output = _adaln_modulate_batched(network, hidden, final_scale, final_shift, dim, eps_t)

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

    print(f"[flux-dit] Building dynamic-batch TRT engine "
          f"(B=1..{max_batch_size}, opt={opt_batch}, dim={dim}, "
          f"joint={num_layers}, single={num_single_layers}, "
          f"img_tokens={num_img_tokens}, text_seq={text_seq_len}) ...",
          file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for dynamic-batch FLUX DiT")
    return bytes(plan)


def load_flux_dit_weights(
    model_dir: str,
    *,
    dim: int = 3072,
    num_heads: int = 24,
    num_layers: int = 19,
    num_single_layers: int = 38,
) -> "WeightDict":
    """Load FLUX DiT weights from diffusers-format transformer directory."""
    from pathlib import Path
    from ...checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor, _has_tensor

    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def _t(name):
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name):
        return _load_tensor(readers, name).astype(np.float32)

    def _maybe_f(name):
        if _has_tensor(readers, name):
            return _f(name)
        return None

    def _maybe_t(name):
        if _has_tensor(readers, name):
            return _t(name)
        return None

    # --- Joint transformer blocks ---
    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        # AdaLN norms
        weights[f"{p}.norm1.linear.weight"] = _t(f"{p}.norm1.linear.weight")
        weights[f"{p}.norm1.linear.bias"] = _f(f"{p}.norm1.linear.bias")
        weights[f"{p}.norm1_context.linear.weight"] = _t(f"{p}.norm1_context.linear.weight")
        weights[f"{p}.norm1_context.linear.bias"] = _f(f"{p}.norm1_context.linear.bias")

        # Attention projections (image)
        for proj in ("to_q", "to_k", "to_v"):
            weights[f"{p}.attn.{proj}.weight"] = _t(f"{p}.attn.{proj}.weight")
            b = _maybe_f(f"{p}.attn.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn.{proj}.bias"] = b
        weights[f"{p}.attn.to_out.0.weight"] = _t(f"{p}.attn.to_out.0.weight")
        b = _maybe_f(f"{p}.attn.to_out.0.bias")
        if b is not None:
            weights[f"{p}.attn.to_out.0.bias"] = b

        # Attention projections (text "added")
        for proj in ("add_q_proj", "add_k_proj", "add_v_proj"):
            weights[f"{p}.attn.{proj}.weight"] = _t(f"{p}.attn.{proj}.weight")
            b = _maybe_f(f"{p}.attn.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn.{proj}.bias"] = b
        weights[f"{p}.attn.to_add_out.weight"] = _t(f"{p}.attn.to_add_out.weight")
        b = _maybe_f(f"{p}.attn.to_add_out.bias")
        if b is not None:
            weights[f"{p}.attn.to_add_out.bias"] = b

        # QK norms
        for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            w = _maybe_f(f"{p}.attn.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn.{norm}.weight"] = w

        # FFN (image)
        weights[f"{p}.ff.net.0.proj.weight"] = _t(f"{p}.ff.net.0.proj.weight")
        b = _maybe_f(f"{p}.ff.net.0.proj.bias")
        if b is not None:
            weights[f"{p}.ff.net.0.proj.bias"] = b
        weights[f"{p}.ff.net.2.weight"] = _t(f"{p}.ff.net.2.weight")
        b = _maybe_f(f"{p}.ff.net.2.bias")
        if b is not None:
            weights[f"{p}.ff.net.2.bias"] = b

        # FFN (text context)
        weights[f"{p}.ff_context.net.0.proj.weight"] = _t(f"{p}.ff_context.net.0.proj.weight")
        b = _maybe_f(f"{p}.ff_context.net.0.proj.bias")
        if b is not None:
            weights[f"{p}.ff_context.net.0.proj.bias"] = b
        weights[f"{p}.ff_context.net.2.weight"] = _t(f"{p}.ff_context.net.2.weight")
        b = _maybe_f(f"{p}.ff_context.net.2.bias")
        if b is not None:
            weights[f"{p}.ff_context.net.2.bias"] = b

    # --- Single transformer blocks ---
    for i in range(num_single_layers):
        p = f"single_transformer_blocks.{i}"
        weights[f"{p}.norm.linear.weight"] = _t(f"{p}.norm.linear.weight")
        weights[f"{p}.norm.linear.bias"] = _f(f"{p}.norm.linear.bias")

        for proj in ("to_q", "to_k", "to_v"):
            weights[f"{p}.attn.{proj}.weight"] = _t(f"{p}.attn.{proj}.weight")
            b = _maybe_f(f"{p}.attn.{proj}.bias")
            if b is not None:
                weights[f"{p}.attn.{proj}.bias"] = b

        for norm in ("norm_q", "norm_k"):
            w = _maybe_f(f"{p}.attn.{norm}.weight")
            if w is not None:
                weights[f"{p}.attn.{norm}.weight"] = w

        weights[f"{p}.proj_mlp.weight"] = _t(f"{p}.proj_mlp.weight")
        b = _maybe_f(f"{p}.proj_mlp.bias")
        if b is not None:
            weights[f"{p}.proj_mlp.bias"] = b

        weights[f"{p}.proj_out.weight"] = _t(f"{p}.proj_out.weight")
        b = _maybe_f(f"{p}.proj_out.bias")
        if b is not None:
            weights[f"{p}.proj_out.bias"] = b

    # --- Global ---
    weights["norm_out.linear.weight"] = _t("norm_out.linear.weight")
    weights["norm_out.linear.bias"] = _f("norm_out.linear.bias")
    weights["proj_out.weight"] = _t("proj_out.weight")
    b = _maybe_f("proj_out.bias")
    if b is not None:
        weights["proj_out.bias"] = b

    # Preprocessor weights (external to TRT engine)
    weights["x_embedder.weight"] = _t("x_embedder.weight")
    weights["x_embedder.bias"] = _f("x_embedder.bias")
    weights["context_embedder.weight"] = _t("context_embedder.weight")
    weights["context_embedder.bias"] = _f("context_embedder.bias")

    # Time-text embedding MLPs
    for comp in ("timestep_embedder", "text_embedder"):
        for layer in ("linear_1", "linear_2"):
            key = f"time_text_embed.{comp}.{layer}"
            if _has_tensor(readers, f"{key}.weight"):
                weights[f"{key}.weight"] = _t(f"{key}.weight")
                b = _maybe_f(f"{key}.bias")
                if b is not None:
                    weights[f"{key}.bias"] = b

    # Guidance embedder (optional, only for FLUX.1-dev / guidance_embeds=True)
    for layer in ("linear_1", "linear_2"):
        key = f"time_text_embed.guidance_embedder.{layer}"
        if _has_tensor(readers, f"{key}.weight"):
            weights[f"{key}.weight"] = _t(f"{key}.weight")
            b = _maybe_f(f"{key}.bias")
            if b is not None:
                weights[f"{key}.bias"] = b

    return weights
