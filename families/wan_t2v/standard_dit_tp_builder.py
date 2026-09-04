# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared DiT (Diffusion Transformer) engine builder.

Builds a TensorRT engine for a DiT-style denoiser. Parameterized for
variant selection, mirroring how standard_decoder_builder.py works.

Reusable by: Wan2.1, Hunyuan Video, CogVideoX, FLUX (with conditioning_type).

Engine I/O:
    Inputs:
        hidden_states [1, num_patches, dim] float32 (patchified latent)
        timestep_embedding [1, dim * 6] float32 (from external timestep MLP)
        encoder_hidden_states [1, text_seq_len, context_dim] float32
        rotary_cos [1, num_patches, 1, head_dim] float32 (precomputed)
        rotary_sin [1, num_patches, 1, head_dim] float32 (precomputed)
    Outputs:
        output [1, num_patches, dim] float32

The timestep embedding, patch embedding, and RoPE are computed externally
(in the DiffusionRunner / C++ backend) and passed as inputs. This keeps
the TRT engine focused on the core transformer computation.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from .parallel import (
    ParallelConfig,
    _slice_first_dim,
    _slice_last_dim,
    add_all_reduce_sum,
    normalize_parallel_config,
    validate_dit_tp,
)

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_standard_dit_engine(
    weights: WeightDict,
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    context_dim: int,
    num_patches: int,
    text_seq_len: int = 512,
    use_rope: bool = True,
    eps: float = 1e-06,
    verbose: bool = False,
    parallel_config: ParallelConfig | None = None,
) -> bytes:
    """Build DiT denoiser TRT engine plan.

    Args:
        weights: Weight dict with DiT weights. Expected keys per layer:
            - blocks.{i}.attn1.to_q/to_k/to_v.weight/bias (self-attn)
            - blocks.{i}.attn1.to_out.0.weight/bias
            - blocks.{i}.attn1.norm_q/norm_k.weight (QK norm)
            - blocks.{i}.norm1 (no weight — elementwise_affine=False)
            - blocks.{i}.attn2.to_q/to_k/to_v.weight/bias (cross-attn)
            - blocks.{i}.attn2.to_out.0.weight/bias
            - blocks.{i}.attn2.norm_q/norm_k.weight
            - blocks.{i}.attn2.add_k_proj/add_v_proj.weight/bias (if context needs projection)
            - blocks.{i}.norm2.weight/bias (cross-attn norm, if enabled)
            - blocks.{i}.ffn.net.0.proj.weight/bias (GELU)
            - blocks.{i}.ffn.net.2.weight/bias (output proj)
            - blocks.{i}.norm3 (no weight — elementwise_affine=False)
            - blocks.{i}.scale_shift_table [1, 6, dim]
            Global:
            - norm_out (no weight — elementwise_affine=False)
            - proj_out.weight/bias
            - scale_shift_table [1, 2, dim]
        dim: Hidden dimension of the DiT.
        num_heads: Number of attention heads.
        num_layers: Number of DiT blocks.
        ffn_dim: Feed-forward inner dimension.
        context_dim: Text encoder output dimension (before projection).
        num_patches: Total number of patches (T/pt * H/ph * W/pw).
        text_seq_len: Maximum text sequence length.
        qk_norm: Apply RMSNorm to Q and K.
        cross_attn_norm: Apply LayerNorm before cross-attention.
        ffn_activation: Activation for FFN.
        use_rope: Apply RoPE to self-attention Q/K. When False, the engine
            omits rotary_cos/rotary_sin inputs (suitable for models that use
            fixed position embeddings, e.g. PixArt).
        eps: LayerNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    parallel = normalize_parallel_config(parallel_config)
    validate_dit_tp(
        dim=dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        parallel=parallel,
        feature="Standard DiT tensor parallel",
    )
    head_dim = dim // num_heads
    local_num_heads = num_heads // parallel.tp_size
    local_dim = dim // parallel.tp_size
    local_ffn_dim = ffn_dim // parallel.tp_size
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    hidden_inp = network.add_input("hidden_states", trt.float32, (num_patches, dim))
    temb_inp = network.add_input("timestep_embedding", trt.float32, (1, 6 * dim))
    time_embed_inp = network.add_input("time_embed", trt.float32, (1, dim))
    encoder_hidden = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, context_dim)
    )
    cross_attn_mask = None
    if not use_rope:
        cross_attn_mask = network.add_input(
            "encoder_attention_mask", trt.float32, (1, 1, text_seq_len)
        )
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))
    rotary_cos = rotary_sin = None
    if use_rope:
        rotary_cos = network.add_input("rotary_cos", trt.float32, (num_patches, head_dim))
        rotary_sin = network.add_input("rotary_sin", trt.float32, (num_patches, head_dim))
    hidden = hidden_inp
    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        sst = weights[f"{prefix}.scale_shift_table"]
        sst_const = graph_ops.add_constant(network, (1, 6 * dim), sst.reshape(1, 6 * dim))
        modulation = network.add_elementwise(sst_const, temb_inp, trt.ElementWiseOperation.SUM)
        chunks = []
        for i in range(6):
            s = network.add_slice(
                modulation.get_output(0), start=(0, i * dim), shape=(1, dim), stride=(1, 1)
            )
            chunks.append(s.get_output(0))
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = chunks
        normed = graph_ops.add_adaptive_layernorm(network, hidden, scale_sa, shift_sa, dim, eps)
        q = _linear_col_parallel(
            network, normed, dim, dim, weights, f"{prefix}.attn1.to_q", parallel
        )
        k = _linear_col_parallel(
            network, normed, dim, dim, weights, f"{prefix}.attn1.to_k", parallel
        )
        v = _linear_col_parallel(
            network, normed, dim, dim, weights, f"{prefix}.attn1.to_v", parallel
        )
        q_norm_w = weights.get(f"{prefix}.attn1.norm_q.weight")
        k_norm_w = weights.get(f"{prefix}.attn1.norm_k.weight")
        if q_norm_w is not None:
            q = graph_ops.add_rms_norm(
                network,
                q,
                local_dim,
                _tp_norm_weight(q_norm_w, local_num_heads, head_dim, parallel),
                eps_t,
            )
        if k_norm_w is not None:
            k = graph_ops.add_rms_norm(
                network,
                k,
                local_dim,
                _tp_norm_weight(k_norm_w, local_num_heads, head_dim, parallel),
                eps_t,
            )
        if use_rope:
            q = graph_ops.add_apply_rope_native_from_full_cache(
                network,
                q,
                local_num_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                num_patches,
                interleaved=True,
            )
            k = graph_ops.add_apply_rope_native_from_full_cache(
                network,
                k,
                local_num_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                num_patches,
                interleaved=True,
            )
        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=local_num_heads,
            head_dim=head_dim,
            q_seq=num_patches,
            kv_seq=num_patches,
            tag=f"{prefix}.attn1",
        )
        attn_out = _linear_row_parallel(
            network, context_flat, local_dim, dim, weights, f"{prefix}.attn1.to_out.0", parallel
        )
        gated = network.add_elementwise(attn_out, gate_sa, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)
        cross_norm_w = weights.get(f"{prefix}.norm2.weight")
        cross_norm_b = weights.get(f"{prefix}.norm2.bias")
        if cross_norm_w is not None:
            cross_normed = graph_ops.add_layer_norm(
                network,
                hidden,
                dim,
                cross_norm_w,
                cross_norm_b if cross_norm_b is not None else np.zeros(dim, dtype=np.float32),
                eps_t,
            )
        else:
            cross_normed = hidden
        cross_q = _linear_col_parallel(
            network, cross_normed, dim, dim, weights, f"{prefix}.attn2.to_q", parallel
        )
        add_k_proj_w = weights.get(f"{prefix}.attn2.add_k_proj.weight")
        if add_k_proj_w is not None:
            cross_k = _linear_col_parallel(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.add_k_proj",
                parallel,
            )
            cross_v = _linear_col_parallel(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.add_v_proj",
                parallel,
            )
        else:
            cross_k = _linear_col_parallel(
                network, encoder_hidden, context_dim, dim, weights, f"{prefix}.attn2.to_k", parallel
            )
            cross_v = _linear_col_parallel(
                network, encoder_hidden, context_dim, dim, weights, f"{prefix}.attn2.to_v", parallel
            )
        cq_norm = weights.get(f"{prefix}.attn2.norm_q.weight")
        ck_norm = weights.get(f"{prefix}.attn2.norm_k.weight")
        if cq_norm is not None:
            cross_q = graph_ops.add_rms_norm(
                network,
                cross_q,
                local_dim,
                _tp_norm_weight(cq_norm, local_num_heads, head_dim, parallel),
                eps_t,
            )
        ck_added_norm = weights.get(f"{prefix}.attn2.norm_added_k.weight")
        if ck_added_norm is not None:
            cross_k = graph_ops.add_rms_norm(
                network,
                cross_k,
                local_dim,
                _tp_norm_weight(ck_added_norm, local_num_heads, head_dim, parallel),
                eps_t,
            )
        elif ck_norm is not None:
            cross_k = graph_ops.add_rms_norm(
                network,
                cross_k,
                local_dim,
                _tp_norm_weight(ck_norm, local_num_heads, head_dim, parallel),
                eps_t,
            )
        cross_mask_4d = None
        if cross_attn_mask is not None:
            cross_mask = network.add_shuffle(cross_attn_mask)
            cross_mask.reshape_dims = (1, 1, 1, text_seq_len)
            cross_mask_4d = cross_mask.get_output(0)
        c_context_flat = graph_ops.add_attention_from_rows(
            network,
            cross_q,
            cross_k,
            cross_v,
            num_heads=local_num_heads,
            head_dim=head_dim,
            q_seq=num_patches,
            kv_seq=text_seq_len,
            mask=cross_mask_4d,
            tag=f"{prefix}.attn2",
        )
        cross_out = _linear_row_parallel(
            network, c_context_flat, local_dim, dim, weights, f"{prefix}.attn2.to_out.0", parallel
        )
        hidden = network.add_elementwise(
            hidden, cross_out, trt.ElementWiseOperation.SUM
        ).get_output(0)
        ffn_normed = graph_ops.add_adaptive_layernorm(network, hidden, scale_ff, shift_ff, dim, eps)
        ffn_fc1 = _linear_col_parallel(
            network, ffn_normed, dim, ffn_dim, weights, f"{prefix}.ffn.net.0.proj", parallel
        )
        ffn_act = graph_ops.add_gelu_new(network, ffn_fc1)
        ffn_fc2 = _linear_row_parallel(
            network, ffn_act, local_ffn_dim, dim, weights, f"{prefix}.ffn.net.2", parallel
        )
        gated_ff = network.add_elementwise(ffn_fc2, gate_ff, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated_ff.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)
    final_sst = weights["scale_shift_table"]
    final_sst_const = graph_ops.add_constant(network, (1, 2 * dim), final_sst.reshape(1, 2 * dim))
    time_embed_tiled = network.add_concatenation([time_embed_inp, time_embed_inp])
    time_embed_tiled.axis = 1
    final_modulation = network.add_elementwise(
        final_sst_const, time_embed_tiled.get_output(0), trt.ElementWiseOperation.SUM
    )
    final_shift = network.add_slice(
        final_modulation.get_output(0), start=(0, 0), shape=(1, dim), stride=(1, 1)
    )
    final_scale = network.add_slice(
        final_modulation.get_output(0), start=(0, dim), shape=(1, dim), stride=(1, 1)
    )
    hidden = graph_ops.add_adaptive_layernorm(
        network, hidden, final_scale.get_output(0), final_shift.get_output(0), dim, eps
    )
    proj_out_w = weights["proj_out.weight"]
    out_dim = proj_out_w.shape[1]
    output = graph_ops.add_matmul_rhs_constant(network, hidden, dim, out_dim, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_dim, proj_out_b)
    cast_output = network.add_cast(output, trt.float32)
    output_final = cast_output.get_output(0)
    output_final.name = "output"
    network.mark_output(output_final)
    tp_suffix = f", tp={parallel.tp_size}, rank={parallel.rank}" if parallel.enabled else ""
    print(
        f"[dit-builder] Building TRT engine (dim={dim}, layers={num_layers}, patches={num_patches}{tp_suffix}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for DiT")
    return bytes(plan)


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


def _tp_norm_weight(
    weight: np.ndarray,
    local_num_heads: int,
    head_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Return the rank-local norm vector for sharded attention projections."""
    if not parallel.enabled:
        return weight
    if weight.size == head_dim:
        return weight
    return _slice_first_dim(weight.reshape(-1, head_dim), parallel.rank, parallel.tp_size).reshape(
        local_num_heads * head_dim
    )
