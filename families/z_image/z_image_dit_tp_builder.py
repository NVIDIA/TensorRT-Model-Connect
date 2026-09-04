# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image DiT engine builder.

Builds a TRT engine for the ZImageTransformer2DModel:
  - 30 main DiT layers (unified single-stream attention + SwiGLU FFN + 4-param AdaLN with tanh gating)
  - 2 noise_refiner layers (same structure as main layers, operate on noise tokens only)
  - 2 context_refiner layers (no AdaLN, plain pre-norm attention + SwiGLU FFN, operate on caption tokens)
  - 3-axis RoPE (time, height, width) with complex-number style

Engine I/O:
    Inputs:
        hidden_states [num_patches, dim] float32  (patchified+embedded noise latents)
        encoder_hidden_states [text_seq_len, dim] float32  (projected caption embeddings)
        timestep_embedding [1, adaln_embed_dim] float32  (t_embedder MLP output)
        rotary_cos [total_seq, head_dim] float32  (3-axis RoPE cos)
        rotary_sin [total_seq, head_dim] float32  (3-axis RoPE sin)
    Outputs:
        output [num_patches, out_channels] float32

HF architecture per main layer:
    # AdaLN modulation: Linear(adaln_dim, 4*dim) -- NO SiLU before per-layer
    mod = adaLN_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(mod, 4)
    gate_msa, gate_mlp = tanh(gate_msa), tanh(gate_mlp)
    scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp

    # Pre-norm + self-attention (unified: noise + caption concatenated)
    x_norm = RMSNorm(x) * scale_msa        # attention_norm1 is pre-norm
    attn_out = SelfAttention(x_norm, RoPE)
    x = x + gate_msa * RMSNorm(attn_out)   # attention_norm2 is POST-norm on attn output

    # Pre-norm + SwiGLU FFN
    x_norm = RMSNorm(x) * scale_mlp        # ffn_norm1 is pre-norm
    ffn_out = SwiGLU(x_norm)
    x = x + gate_mlp * RMSNorm(ffn_out)    # ffn_norm2 is POST-norm on ffn output
"""

from __future__ import annotations

import sys

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import WeightDict
from .parallel import (
    ParallelConfig,
    _slice_first_dim,
    _slice_last_dim,
    add_all_reduce_sum,
    normalize_parallel_config,
    validate_dit_tp,
)


def build_z_image_dit_engine(
    weights: WeightDict,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    num_refiner_layers: int,
    ffn_dim: int,
    num_patches: int,
    text_seq_len: int,
    head_dim: int = 128,
    adaln_embed_dim: int = 256,
    eps: float = 1e-5,
    verbose: bool = False,
    parallel_config: ParallelConfig | None = None,
) -> bytes:
    """Build Z-Image DiT TRT engine."""
    parallel = normalize_parallel_config(parallel_config)
    validate_dit_tp(
        dim=dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        parallel=parallel,
        feature="Z-Image tensor parallel",
    )
    total_seq = num_patches + text_seq_len
    attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
    out_channels = weights["final_linear.weight"].shape[1]
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Inputs (no batch dim -- static shapes)
    noise_inp = network.add_input("hidden_states", trt.float32, (num_patches, dim))
    caption_inp = network.add_input("encoder_hidden_states", trt.float32, (text_seq_len, dim))
    temb_inp = network.add_input("timestep_embedding", trt.float32, (1, adaln_embed_dim))
    rope_cos = network.add_input("rotary_cos", trt.float32, (total_seq, head_dim))
    rope_sin = network.add_input("rotary_sin", trt.float32, (total_seq, head_dim))

    # Constants
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    scale_t = graph_ops.add_constant(network, (1, 1, 1), np.array([attn_scale], dtype=np.float32))
    ones_t = graph_ops.add_constant(network, (1, 1), np.array([1.0], dtype=np.float32))

    noise = noise_inp
    caption = caption_inp

    # --- Noise refiner (on noise only, with AdaLN) ---
    noise_cos = _slice_rope(network, rope_cos, 0, num_patches, head_dim)
    noise_sin = _slice_rope(network, rope_sin, 0, num_patches, head_dim)
    for i in range(num_refiner_layers):
        tp = f"noise_refiner.{i}"
        noise = _add_adaln_dit_block(
            network,
            noise,
            weights,
            tp,
            temb_inp,
            dim,
            num_heads,
            head_dim,
            ffn_dim,
            adaln_embed_dim,
            num_patches,
            eps_t,
            scale_t,
            noise_cos,
            noise_sin,
            ones_t,
            parallel,
        )

    # --- Context refiner (on caption only, no AdaLN) ---
    cap_cos = _slice_rope(network, rope_cos, num_patches, text_seq_len, head_dim)
    cap_sin = _slice_rope(network, rope_sin, num_patches, text_seq_len, head_dim)
    for i in range(num_refiner_layers):
        tp = f"context_refiner.{i}"
        caption = _add_plain_dit_block(
            network,
            caption,
            weights,
            tp,
            dim,
            num_heads,
            head_dim,
            ffn_dim,
            text_seq_len,
            eps_t,
            scale_t,
            cap_cos,
            cap_sin,
            parallel,
        )

    # --- Main layers (unified: noise + caption concatenated) ---
    for i in range(num_layers):
        tp = f"main.{i}"

        # AdaLN modulation: Linear(adaln_dim, 4*dim) -- NO SiLU (unlike FinalLayer)
        adaln_w = weights[f"{tp}.adaln.weight"]
        adaln_b = weights[f"{tp}.adaln.bias"]
        modulation = graph_ops.add_matmul_rhs_constant(
            network, temb_inp, adaln_embed_dim, 4 * dim, adaln_w
        )
        modulation = graph_ops.add_bias_sum(network, modulation, 4 * dim, adaln_b)

        # Chunk into 4: scale_msa, gate_msa, scale_mlp, gate_mlp
        chunks = []
        for ci in range(4):
            s = network.add_slice(modulation, start=(0, ci * dim), shape=(1, dim), stride=(1, 1))
            chunks.append(s.get_output(0))
        scale_msa, gate_msa_raw, scale_mlp, gate_mlp_raw = chunks

        # Tanh gating (critical for stability)
        gate_msa_act = network.add_activation(gate_msa_raw, trt.ActivationType.TANH)
        gate_msa = gate_msa_act.get_output(0)
        gate_mlp_act = network.add_activation(gate_mlp_raw, trt.ActivationType.TANH)
        gate_mlp = gate_mlp_act.get_output(0)

        # scale = 1 + scale
        scale_msa_p1 = network.add_elementwise(
            scale_msa, ones_t, trt.ElementWiseOperation.SUM
        ).get_output(0)
        scale_mlp_p1 = network.add_elementwise(
            scale_mlp, ones_t, trt.ElementWiseOperation.SUM
        ).get_output(0)

        # --- Self-attention with AdaLN ---
        # Concatenate noise + caption FIRST, then apply SAME attention_norm1
        unified = network.add_concatenation([noise, caption])
        unified.axis = 0
        unified_t = unified.get_output(0)

        # Pre-norm: attention_norm1 on the UNIFIED sequence (both noise and caption)
        unified_normed = graph_ops.add_rms_norm(
            network, unified_t, dim, weights[f"{tp}.attn_norm1"], eps_t
        )

        # Apply scale_msa to unified normed sequence
        unified_scaled = network.add_elementwise(
            unified_normed, scale_msa_p1, trt.ElementWiseOperation.PROD
        ).get_output(0)

        # QKV on unified scaled sequence
        q = _linear_col_parallel(network, unified_scaled, dim, dim, weights, f"{tp}.to_q", parallel)
        k = _linear_col_parallel(network, unified_scaled, dim, dim, weights, f"{tp}.to_k", parallel)
        v = _linear_col_parallel(network, unified_scaled, dim, dim, weights, f"{tp}.to_v", parallel)

        # QK norm (per-head)
        q_norm_w = weights[f"{tp}.norm_q"]
        k_norm_w = weights[f"{tp}.norm_k"]
        q_norm_tiled = _tp_per_head_norm_weight(q_norm_w, local_num_heads, head_dim, parallel)
        k_norm_tiled = _tp_per_head_norm_weight(k_norm_w, local_num_heads, head_dim, parallel)
        q = _per_head_rms_norm(
            network, q, local_num_heads, head_dim, q_norm_tiled, eps_t, total_seq
        )
        k = _per_head_rms_norm(
            network, k, local_num_heads, head_dim, k_norm_tiled, eps_t, total_seq
        )

        # Apply RoPE (full unified sequence)
        q = _apply_native_rope_from_full_cache(
            network, q, rope_cos, rope_sin, local_num_heads, head_dim, total_seq
        )
        k = _apply_native_rope_from_full_cache(
            network, k, rope_cos, rope_sin, local_num_heads, head_dim, total_seq
        )

        # Multi-head attention
        attn_out = _multi_head_attention(
            network, q, k, v, local_num_heads, head_dim, total_seq, total_seq, scale_t
        )

        # Output projection
        attn_out = _linear_row_parallel(
            network, attn_out, local_dim, dim, weights, f"{tp}.to_out", parallel
        )

        # Post-norm: attention_norm2 on attn output (NOT on input)
        attn_out_normed = graph_ops.add_rms_norm(
            network, attn_out, dim, weights[f"{tp}.attn_norm2"], eps_t
        )

        # Gate + residual
        gated_attn = network.add_elementwise(
            attn_out_normed, gate_msa, trt.ElementWiseOperation.PROD
        )
        unified_t = network.add_elementwise(
            unified_t, gated_attn.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

        # --- SwiGLU FFN with AdaLN ---
        # Pre-norm: ffn_norm1 on unified
        unified_ffn_normed = graph_ops.add_rms_norm(
            network, unified_t, dim, weights[f"{tp}.ffn_norm1"], eps_t
        )
        unified_ffn_scaled = network.add_elementwise(
            unified_ffn_normed, scale_mlp_p1, trt.ElementWiseOperation.PROD
        ).get_output(0)

        # SwiGLU: gate = SiLU(x @ w1), up = x @ w3, out = (gate * up) @ w2
        gate_proj = _linear_col_parallel(
            network, unified_ffn_scaled, dim, ffn_dim, weights, f"{tp}.ff_w1", parallel
        )
        up_proj = _linear_col_parallel(
            network, unified_ffn_scaled, dim, ffn_dim, weights, f"{tp}.ff_w3", parallel
        )

        gate_sigmoid = network.add_activation(gate_proj, trt.ActivationType.SIGMOID)
        gate_silu = network.add_elementwise(
            gate_proj, gate_sigmoid.get_output(0), trt.ElementWiseOperation.PROD
        )
        gated_ffn = network.add_elementwise(
            gate_silu.get_output(0), up_proj, trt.ElementWiseOperation.PROD
        )

        down_proj = _linear_row_parallel(
            network, gated_ffn.get_output(0), local_ffn_dim, dim, weights, f"{tp}.ff_w2", parallel
        )

        # Post-norm: ffn_norm2 on FFN output
        ffn_out_normed = graph_ops.add_rms_norm(
            network, down_proj, dim, weights[f"{tp}.ffn_norm2"], eps_t
        )

        gated_ffn_out = network.add_elementwise(
            ffn_out_normed, gate_mlp, trt.ElementWiseOperation.PROD
        )
        unified_t = network.add_elementwise(
            unified_t, gated_ffn_out.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

        # Split unified back into noise and caption
        noise = network.add_slice(
            unified_t, start=(0, 0), shape=(num_patches, dim), stride=(1, 1)
        ).get_output(0)
        caption = network.add_slice(
            unified_t,
            start=(num_patches, 0),
            shape=(text_seq_len, dim),
            stride=(1, 1),
        ).get_output(0)

    # --- Final layer ---
    # FinalLayer: LayerNorm(dim, elementwise_affine=False) * (1 + SiLU(Linear(adaln_dim, dim)))
    # Then Linear(dim, out_channels)

    # SiLU on temb, then linear
    temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_act = network.add_elementwise(
        temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD
    )
    final_mod = graph_ops.add_matmul_rhs_constant(
        network, temb_silu_act.get_output(0), adaln_embed_dim, dim, weights["final_adaLN.weight"]
    )
    final_mod = graph_ops.add_bias_sum(network, final_mod, dim, weights["final_adaLN.bias"])
    final_scale = network.add_elementwise(
        final_mod, ones_t, trt.ElementWiseOperation.SUM
    ).get_output(0)

    # LayerNorm (elementwise_affine=False, eps=1e-6): mean-center then variance-normalize
    ln_eps = graph_ops.add_constant(network, (1, 1), np.array([1e-6], dtype=np.float32))

    # Compute mean: [num_patches, 1]
    noise_mean = network.add_reduce(noise, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # Subtract mean
    noise_centered = network.add_elementwise(
        noise, noise_mean.get_output(0), trt.ElementWiseOperation.SUB
    ).get_output(0)
    # Compute variance
    noise_sq = network.add_elementwise(
        noise_centered, noise_centered, trt.ElementWiseOperation.PROD
    )
    noise_var = network.add_reduce(
        noise_sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True
    )
    noise_var_eps = network.add_elementwise(
        noise_var.get_output(0), ln_eps, trt.ElementWiseOperation.SUM
    )
    noise_std = network.add_unary(noise_var_eps.get_output(0), trt.UnaryOperation.SQRT)
    noise_std_recip = network.add_unary(noise_std.get_output(0), trt.UnaryOperation.RECIP)
    noise_ln = network.add_elementwise(
        noise_centered, noise_std_recip.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)

    # Apply scale
    noise_final = network.add_elementwise(
        noise_ln, final_scale, trt.ElementWiseOperation.PROD
    ).get_output(0)

    # Final linear projection
    output = graph_ops.add_matmul_rhs_constant(
        network, noise_final, dim, out_channels, weights["final_linear.weight"]
    )
    output = graph_ops.add_bias_sum(network, output, out_channels, weights["final_linear.bias"])

    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)

    tp_suffix = f", tp={parallel.tp_size}, rank={parallel.rank}" if parallel.enabled else ""
    print(
        f"[z-image-dit] Building TRT engine "
        f"(dim={dim}, layers={num_layers}, refiners={num_refiner_layers}, "
        f"patches={num_patches}, text_seq={text_seq_len}, out_ch={out_channels}"
        f"{tp_suffix}) ...",
        file=sys.stderr,
    )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Z-Image DiT TRT engine build failed")
    return bytes(plan)


def _slice_rope(network, rope, start_seq, length, rope_dim):
    """Slice RoPE along sequence dimension."""
    s = network.add_slice(rope, start=(start_seq, 0), shape=(length, rope_dim), stride=(1, 1))
    return s.get_output(0)


def _apply_native_rope_from_full_cache(
    network,
    x,
    cos_t,
    sin_t,
    num_heads,
    head_dim,
    seq_len,
):
    """Apply TRT native RoPE using runtime full-dimension cos/sin rows."""
    return graph_ops.add_apply_rope_native_from_full_cache(
        network, x, num_heads, head_dim, cos_t, sin_t, seq_len, interleaved=True
    )


def _per_head_rms_norm(network, inp, num_heads, head_dim, gamma, eps_t, seq_len):
    """Per-head RMSNorm: [seq, num_heads * head_dim] -> reshape -> norm -> reshape."""
    return graph_ops.add_rms_norm_per_head(
        network, inp, num_heads, head_dim, gamma, eps_t, sequence_length=seq_len
    )


def _linear_col_parallel(network, inp, in_dim, out_dim, weights, key, parallel):
    """Column-parallel linear: output features are rank-local."""
    local_out_dim = out_dim // parallel.tp_size
    weight = weights[key]
    if parallel.enabled:
        weight = _slice_last_dim(weight, parallel.rank, parallel.tp_size)
    return graph_ops.add_matmul_rhs_constant(network, inp, in_dim, local_out_dim, weight)


def _linear_row_parallel(network, inp, in_dim, out_dim, weights, key, parallel):
    """Row-parallel linear: partial outputs are all-reduced across TP ranks."""
    weight = weights[key]
    if parallel.enabled:
        weight = _slice_first_dim(weight, parallel.rank, parallel.tp_size)
    out = graph_ops.add_matmul_rhs_constant(network, inp, in_dim, out_dim, weight)
    if parallel.enabled:
        out = add_all_reduce_sum(network, out, parallel.tp_size)
    return out


def _tp_per_head_norm_weight(weight, local_num_heads, head_dim, parallel):
    if weight.size == head_dim:
        return np.tile(weight.reshape(1, head_dim), (local_num_heads, 1))
    full = weight.reshape(-1, head_dim)
    if parallel.enabled:
        full = _slice_first_dim(full, parallel.rank, parallel.tp_size)
    return np.ascontiguousarray(full)


def _multi_head_attention(network, q, k, v, num_heads, head_dim, q_seq, kv_seq, scale_t):
    """Standard multi-head attention."""
    return graph_ops.add_attention_from_rows(
        network, q, k, v, num_heads=num_heads, head_dim=head_dim, q_seq=q_seq, kv_seq=kv_seq
    )


def _add_plain_dit_block(
    network,
    x,
    weights,
    prefix,
    dim,
    num_heads,
    head_dim,
    ffn_dim,
    seq_len,
    eps_t,
    scale_t,
    cos_t,
    sin_t,
    parallel,
):
    """Plain DiT block (no AdaLN): pre-norm attention + post-norm + SwiGLU FFN.

    HF architecture (modulation=False):
        attn_out = attention(attention_norm1(x))
        x = x + attention_norm2(attn_out)          # norm2 = post-norm
        x = x + ffn_norm2(feed_forward(ffn_norm1(x)))  # norm2 = post-norm
    """
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size

    # Pre-attention norm
    normed = graph_ops.add_rms_norm(network, x, dim, weights[f"{prefix}.attn_norm1"], eps_t)

    # QKV
    q = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_q", parallel)
    k = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_k", parallel)
    v = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_v", parallel)

    # QK norm
    q_norm = _tp_per_head_norm_weight(
        weights[f"{prefix}.norm_q"], local_num_heads, head_dim, parallel
    )
    k_norm = _tp_per_head_norm_weight(
        weights[f"{prefix}.norm_k"], local_num_heads, head_dim, parallel
    )
    q = _per_head_rms_norm(network, q, local_num_heads, head_dim, q_norm, eps_t, seq_len)
    k = _per_head_rms_norm(network, k, local_num_heads, head_dim, k_norm, eps_t, seq_len)

    # RoPE
    q = _apply_native_rope_from_full_cache(
        network, q, cos_t, sin_t, local_num_heads, head_dim, seq_len
    )
    k = _apply_native_rope_from_full_cache(
        network, k, cos_t, sin_t, local_num_heads, head_dim, seq_len
    )

    # Attention
    attn_out = _multi_head_attention(
        network, q, k, v, local_num_heads, head_dim, seq_len, seq_len, scale_t
    )
    attn_out = _linear_row_parallel(
        network, attn_out, local_dim, dim, weights, f"{prefix}.to_out", parallel
    )

    # Post-norm on attn output
    attn_out_normed = graph_ops.add_rms_norm(
        network, attn_out, dim, weights[f"{prefix}.attn_norm2"], eps_t
    )

    # Residual
    x = network.add_elementwise(x, attn_out_normed, trt.ElementWiseOperation.SUM).get_output(0)

    # Pre-FFN norm
    ffn_normed = graph_ops.add_rms_norm(network, x, dim, weights[f"{prefix}.ffn_norm1"], eps_t)

    # SwiGLU FFN
    gate_proj = _linear_col_parallel(
        network, ffn_normed, dim, ffn_dim, weights, f"{prefix}.ff_w1", parallel
    )
    up_proj = _linear_col_parallel(
        network, ffn_normed, dim, ffn_dim, weights, f"{prefix}.ff_w3", parallel
    )
    gate_sigmoid = network.add_activation(gate_proj, trt.ActivationType.SIGMOID)
    gate_silu = network.add_elementwise(
        gate_proj, gate_sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    )
    gated = network.add_elementwise(gate_silu.get_output(0), up_proj, trt.ElementWiseOperation.PROD)
    down_proj = _linear_row_parallel(
        network, gated.get_output(0), local_ffn_dim, dim, weights, f"{prefix}.ff_w2", parallel
    )

    # Post-norm on FFN output
    ffn_out_normed = graph_ops.add_rms_norm(
        network, down_proj, dim, weights[f"{prefix}.ffn_norm2"], eps_t
    )

    x = network.add_elementwise(x, ffn_out_normed, trt.ElementWiseOperation.SUM).get_output(0)
    return x


def _add_adaln_dit_block(
    network,
    x,
    weights,
    prefix,
    temb,
    dim,
    num_heads,
    head_dim,
    ffn_dim,
    adaln_embed_dim,
    seq_len,
    eps_t,
    scale_t,
    cos_t,
    sin_t,
    ones_t,
    parallel,
):
    """AdaLN DiT block (noise_refiner): 4-chunk modulation + tanh gating + attention + SwiGLU.

    HF architecture (modulation=True):
        mod = adaLN_modulation(adaln_input)  # NO SiLU -- just Linear
        scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(mod, 4)
        gate_msa, gate_mlp = tanh(gate_msa), tanh(gate_mlp)
        scale_msa, scale_mlp = 1 + scale_msa, 1 + scale_mlp

        attn_out = attention(attention_norm1(x) * scale_msa)
        x = x + gate_msa * attention_norm2(attn_out)

        ffn_out = feed_forward(ffn_norm1(x) * scale_mlp)
        x = x + gate_mlp * ffn_norm2(ffn_out)
    """
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size

    # AdaLN modulation: just Linear, no SiLU
    adaln_w = weights[f"{prefix}.adaln.weight"]
    adaln_b = weights[f"{prefix}.adaln.bias"]
    mod = graph_ops.add_matmul_rhs_constant(network, temb, adaln_embed_dim, 4 * dim, adaln_w)
    mod = graph_ops.add_bias_sum(network, mod, 4 * dim, adaln_b)

    chunks = []
    for ci in range(4):
        s = network.add_slice(mod, start=(0, ci * dim), shape=(1, dim), stride=(1, 1))
        chunks.append(s.get_output(0))
    scale_msa, gate_msa_raw, scale_mlp, gate_mlp_raw = chunks

    # Tanh gating
    gate_msa = network.add_activation(gate_msa_raw, trt.ActivationType.TANH).get_output(0)
    gate_mlp = network.add_activation(gate_mlp_raw, trt.ActivationType.TANH).get_output(0)

    scale_msa_p1 = network.add_elementwise(
        scale_msa, ones_t, trt.ElementWiseOperation.SUM
    ).get_output(0)
    scale_mlp_p1 = network.add_elementwise(
        scale_mlp, ones_t, trt.ElementWiseOperation.SUM
    ).get_output(0)

    # Pre-attention norm + AdaLN scale
    normed = graph_ops.add_rms_norm(network, x, dim, weights[f"{prefix}.attn_norm1"], eps_t)
    normed = network.add_elementwise(
        normed, scale_msa_p1, trt.ElementWiseOperation.PROD
    ).get_output(0)

    # QKV
    q = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_q", parallel)
    k = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_k", parallel)
    v = _linear_col_parallel(network, normed, dim, dim, weights, f"{prefix}.to_v", parallel)

    q_norm = _tp_per_head_norm_weight(
        weights[f"{prefix}.norm_q"], local_num_heads, head_dim, parallel
    )
    k_norm = _tp_per_head_norm_weight(
        weights[f"{prefix}.norm_k"], local_num_heads, head_dim, parallel
    )
    q = _per_head_rms_norm(network, q, local_num_heads, head_dim, q_norm, eps_t, seq_len)
    k = _per_head_rms_norm(network, k, local_num_heads, head_dim, k_norm, eps_t, seq_len)

    q = _apply_native_rope_from_full_cache(
        network, q, cos_t, sin_t, local_num_heads, head_dim, seq_len
    )
    k = _apply_native_rope_from_full_cache(
        network, k, cos_t, sin_t, local_num_heads, head_dim, seq_len
    )

    attn_out = _multi_head_attention(
        network, q, k, v, local_num_heads, head_dim, seq_len, seq_len, scale_t
    )
    attn_out = _linear_row_parallel(
        network, attn_out, local_dim, dim, weights, f"{prefix}.to_out", parallel
    )

    # Post-norm on attn output
    attn_out_normed = graph_ops.add_rms_norm(
        network, attn_out, dim, weights[f"{prefix}.attn_norm2"], eps_t
    )

    gated_attn = network.add_elementwise(attn_out_normed, gate_msa, trt.ElementWiseOperation.PROD)
    x = network.add_elementwise(
        x, gated_attn.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)

    # FFN
    ffn_normed = graph_ops.add_rms_norm(network, x, dim, weights[f"{prefix}.ffn_norm1"], eps_t)
    ffn_normed = network.add_elementwise(
        ffn_normed, scale_mlp_p1, trt.ElementWiseOperation.PROD
    ).get_output(0)

    gate_proj = _linear_col_parallel(
        network, ffn_normed, dim, ffn_dim, weights, f"{prefix}.ff_w1", parallel
    )
    up_proj = _linear_col_parallel(
        network, ffn_normed, dim, ffn_dim, weights, f"{prefix}.ff_w3", parallel
    )
    gate_sigmoid = network.add_activation(gate_proj, trt.ActivationType.SIGMOID)
    gate_silu = network.add_elementwise(
        gate_proj, gate_sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    )
    gated = network.add_elementwise(gate_silu.get_output(0), up_proj, trt.ElementWiseOperation.PROD)
    down_proj = _linear_row_parallel(
        network, gated.get_output(0), local_ffn_dim, dim, weights, f"{prefix}.ff_w2", parallel
    )

    # Post-norm on FFN output
    ffn_out_normed = graph_ops.add_rms_norm(
        network, down_proj, dim, weights[f"{prefix}.ffn_norm2"], eps_t
    )

    gated_ffn = network.add_elementwise(ffn_out_normed, gate_mlp, trt.ElementWiseOperation.PROD)
    x = network.add_elementwise(
        x, gated_ffn.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    return x
