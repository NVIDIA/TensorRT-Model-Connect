# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1 DiT Ulysses context-parallel engine builder.

Weights stay replicated. Activations are sharded over the text and image
sequence dimensions, while attention temporarily exchanges sequence shards
for head shards with ALL_TO_ALL:

    [S / CP, H, D] -> [S, H / CP, D] -> attention
                    -> [S / CP, H, D]

The ordering matches AITune's ``FluxUlyssesAttn``: every rank owns
``text_rank || image_rank`` and rotary caches are sliced from the text and
image regions independently before being concatenated in that same order.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from .flux_dit_builder import (
    _adaln_modulate,
    _apply_native_rope_from_full_cache,
    _chunk_3,
    _chunk_6,
    _configure_compute_precision,
    _gate_1d,
    _gelu_ffn,
    _layernorm_modulate,
    _linear,
    _matmul_bias_1d,
    _rms_norm_per_head_seq,
)
from .parallel import ParallelConfig, normalize_parallel_config

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
    """Build one rank-dynamic FLUX.1 Ulysses context-parallel plan."""
    parallel = normalize_parallel_config(parallel_config)
    _validate_context_parallel(
        parallel,
        num_heads=num_heads,
        num_img_tokens=num_img_tokens,
        text_seq_len=text_seq_len,
    )

    # The upstream CP4 graph is FP32. Set the family-local helper state
    # explicitly so a preceding FLUX build in this process cannot affect it.
    _configure_compute_precision("fp32")
    cp_size = parallel.cp_size
    head_dim = dim // num_heads
    ffn_dim = int(dim * mlp_ratio)
    local_img_tokens = num_img_tokens // cp_size
    local_text_seq = text_seq_len // cp_size
    local_seq = local_text_seq + local_img_tokens
    total_seq = text_seq_len + num_img_tokens

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # Every rank receives identical full inputs. REDUCE_SCATTER selects
    # rank-local rows dynamically, so all CP ranks share one serialized plan.
    hidden_inp = network.add_input("hidden_states", trt.float32, (num_img_tokens, dim))
    encoder_inp = network.add_input("encoder_hidden_states", trt.float32, (text_seq_len, dim))
    temb_inp = network.add_input("temb", trt.float32, (dim,))
    rotary_cos = network.add_input("rotary_cos", trt.float32, (total_seq, head_dim))
    rotary_sin = network.add_input("rotary_sin", trt.float32, (total_seq, head_dim))

    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))

    hidden = _reduce_scatter_replicated(network, hidden_inp, cp_size)
    encoder_hidden = _reduce_scatter_replicated(network, encoder_inp, cp_size)

    # The full rotary is [all text | all image]. Slice both regions
    # independently so local Q/K [text_rank | image_rank] gets matching phases.
    txt_cos_full = _slice_rows(network, rotary_cos, 0, text_seq_len, head_dim)
    txt_sin_full = _slice_rows(network, rotary_sin, 0, text_seq_len, head_dim)
    img_cos_full = _slice_rows(network, rotary_cos, text_seq_len, num_img_tokens, head_dim)
    img_sin_full = _slice_rows(network, rotary_sin, text_seq_len, num_img_tokens, head_dim)
    txt_cos = _reduce_scatter_replicated(network, txt_cos_full, cp_size)
    txt_sin = _reduce_scatter_replicated(network, txt_sin_full, cp_size)
    img_cos = _reduce_scatter_replicated(network, img_cos_full, cp_size)
    img_sin = _reduce_scatter_replicated(network, img_sin_full, cp_size)
    local_cos = _concat_rows(network, txt_cos, img_cos)
    local_sin = _concat_rows(network, txt_sin, img_sin)

    # ===================== Joint Transformer Blocks =====================
    for layer_idx in range(num_layers):
        p = f"transformer_blocks.{layer_idx}"

        temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu_out = network.add_elementwise(
            temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)

        norm1_proj = _matmul_bias_1d(
            network,
            temb_silu_out,
            dim,
            6 * dim,
            weights[f"{p}.norm1.linear.weight"],
            weights[f"{p}.norm1.linear.bias"],
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chunk_6(
            network, norm1_proj, dim
        )
        normed_hidden = _adaln_modulate(
            network, hidden, scale_msa, shift_msa, dim, eps_t, local_img_tokens
        )

        ctx_norm1_proj = _matmul_bias_1d(
            network,
            temb_silu_out,
            dim,
            6 * dim,
            weights[f"{p}.norm1_context.linear.weight"],
            weights[f"{p}.norm1_context.linear.bias"],
        )
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _chunk_6(
            network, ctx_norm1_proj, dim
        )
        normed_encoder = _adaln_modulate(
            network, encoder_hidden, c_scale_msa, c_shift_msa, dim, eps_t, local_text_seq
        )

        q_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_q")
        k_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_k")
        v_img = _linear(network, normed_hidden, dim, dim, weights, f"{p}.attn.to_v")
        q_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_q_proj")
        k_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_k_proj")
        v_txt = _linear(network, normed_encoder, dim, dim, weights, f"{p}.attn.add_v_proj")

        q_img = _rms_norm_per_head_seq(
            network,
            q_img,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_q.weight"],
            eps_t,
            local_img_tokens,
        )
        k_img = _rms_norm_per_head_seq(
            network,
            k_img,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_k.weight"],
            eps_t,
            local_img_tokens,
        )
        q_txt = _rms_norm_per_head_seq(
            network,
            q_txt,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_added_q.weight"],
            eps_t,
            local_text_seq,
        )
        k_txt = _rms_norm_per_head_seq(
            network,
            k_txt,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_added_k.weight"],
            eps_t,
            local_text_seq,
        )

        q_img = _apply_native_rope_from_full_cache(
            network, q_img, img_cos, img_sin, num_heads, head_dim, local_img_tokens
        )
        k_img = _apply_native_rope_from_full_cache(
            network, k_img, img_cos, img_sin, num_heads, head_dim, local_img_tokens
        )
        q_txt = _apply_native_rope_from_full_cache(
            network, q_txt, txt_cos, txt_sin, num_heads, head_dim, local_text_seq
        )
        k_txt = _apply_native_rope_from_full_cache(
            network, k_txt, txt_cos, txt_sin, num_heads, head_dim, local_text_seq
        )

        q_cat = _concat_rows(network, q_txt, q_img)
        k_cat = _concat_rows(network, k_txt, k_img)
        v_cat = _concat_rows(network, v_txt, v_img)
        attn_out = _ulysses_attention_from_rows(
            network,
            q_cat,
            k_cat,
            v_cat,
            local_seq=local_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            cp_size=cp_size,
        )

        txt_attn = _slice_rows(network, attn_out, 0, local_text_seq, dim)
        img_attn = _slice_rows(network, attn_out, local_text_seq, local_img_tokens, dim)

        img_attn_proj = _linear(network, img_attn, dim, dim, weights, f"{p}.attn.to_out.0")
        img_attn_gated = _gate_1d(network, img_attn_proj, gate_msa, local_img_tokens)
        hidden = network.add_elementwise(
            hidden, img_attn_gated, trt.ElementWiseOperation.SUM
        ).get_output(0)

        txt_attn_proj = _linear(network, txt_attn, dim, dim, weights, f"{p}.attn.to_add_out")
        txt_attn_gated = _gate_1d(network, txt_attn_proj, c_gate_msa, local_text_seq)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, txt_attn_gated, trt.ElementWiseOperation.SUM
        ).get_output(0)

        normed_ff = _layernorm_modulate(
            network, hidden, scale_mlp, shift_mlp, dim, eps_t, local_img_tokens
        )
        ff_out = _gelu_ffn(network, normed_ff, dim, weights, f"{p}.ff")
        ff_gated = _gate_1d(network, ff_out, gate_mlp, local_img_tokens)
        hidden = network.add_elementwise(hidden, ff_gated, trt.ElementWiseOperation.SUM).get_output(
            0
        )

        normed_ctx_ff = _layernorm_modulate(
            network, encoder_hidden, c_scale_mlp, c_shift_mlp, dim, eps_t, local_text_seq
        )
        ctx_ff_out = _gelu_ffn(network, normed_ctx_ff, dim, weights, f"{p}.ff_context")
        ctx_ff_gated = _gate_1d(network, ctx_ff_out, c_gate_mlp, local_text_seq)
        encoder_hidden = network.add_elementwise(
            encoder_hidden, ctx_ff_gated, trt.ElementWiseOperation.SUM
        ).get_output(0)

    # ===================== Single Transformer Blocks =====================
    for layer_idx in range(num_single_layers):
        p = f"single_transformer_blocks.{layer_idx}"
        residual = _concat_rows(network, encoder_hidden, hidden)

        temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
        temb_silu_out = network.add_elementwise(
            temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)
        norm_proj = _matmul_bias_1d(
            network,
            temb_silu_out,
            dim,
            3 * dim,
            weights[f"{p}.norm.linear.weight"],
            weights[f"{p}.norm.linear.bias"],
        )
        shift_msa_s, scale_msa_s, gate_msa_s = _chunk_3(network, norm_proj, dim)
        normed_cat = _adaln_modulate(
            network, residual, scale_msa_s, shift_msa_s, dim, eps_t, local_seq
        )

        mlp_hidden = graph_ops.add_matmul_rhs_constant(
            network, normed_cat, dim, ffn_dim, weights[f"{p}.proj_mlp.weight"]
        )
        mlp_bias = weights.get(f"{p}.proj_mlp.bias")
        if mlp_bias is not None:
            mlp_hidden = graph_ops.add_bias_sum(network, mlp_hidden, ffn_dim, mlp_bias)
        mlp_hidden = graph_ops.add_gelu_new(network, mlp_hidden)

        q_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_q")
        k_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_k")
        v_s = _linear(network, normed_cat, dim, dim, weights, f"{p}.attn.to_v")
        q_s = _rms_norm_per_head_seq(
            network,
            q_s,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_q.weight"],
            eps_t,
            local_seq,
        )
        k_s = _rms_norm_per_head_seq(
            network,
            k_s,
            num_heads,
            head_dim,
            weights[f"{p}.attn.norm_k.weight"],
            eps_t,
            local_seq,
        )
        q_s = _apply_native_rope_from_full_cache(
            network, q_s, local_cos, local_sin, num_heads, head_dim, local_seq
        )
        k_s = _apply_native_rope_from_full_cache(
            network, k_s, local_cos, local_sin, num_heads, head_dim, local_seq
        )
        attn_out_s = _ulysses_attention_from_rows(
            network,
            q_s,
            k_s,
            v_s,
            local_seq=local_seq,
            num_heads=num_heads,
            head_dim=head_dim,
            cp_size=cp_size,
        )

        cat_attn_mlp = network.add_concatenation([attn_out_s, mlp_hidden])
        cat_attn_mlp.axis = 1
        in_features = dim + ffn_dim
        combined = graph_ops.add_matmul_rhs_constant(
            network,
            cat_attn_mlp.get_output(0),
            in_features,
            dim,
            weights[f"{p}.proj_out.weight"],
        )
        proj_out_bias = weights.get(f"{p}.proj_out.bias")
        if proj_out_bias is not None:
            combined = graph_ops.add_bias_sum(network, combined, dim, proj_out_bias)

        gated = _gate_1d(network, combined, gate_msa_s, local_seq)
        cat_hidden_out = network.add_elementwise(
            residual, gated, trt.ElementWiseOperation.SUM
        ).get_output(0)
        encoder_hidden = _slice_rows(network, cat_hidden_out, 0, local_text_seq, dim)
        hidden = _slice_rows(network, cat_hidden_out, local_text_seq, local_img_tokens, dim)

    # Pointwise final projection stays sequence-local. Gather image rows only
    # at the engine boundary to preserve the existing full-output ABI.
    temb_silu = network.add_activation(temb_inp, trt.ActivationType.SIGMOID)
    temb_silu_out = network.add_elementwise(
        temb_inp, temb_silu.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    final_proj = _matmul_bias_1d(
        network,
        temb_silu_out,
        dim,
        2 * dim,
        weights["norm_out.linear.weight"],
        weights["norm_out.linear.bias"],
    )
    final_scale = network.add_slice(final_proj, (0,), (dim,), (1,)).get_output(0)
    final_shift = network.add_slice(final_proj, (dim,), (dim,), (1,)).get_output(0)
    output = _adaln_modulate(
        network, hidden, final_scale, final_shift, dim, eps_t, local_img_tokens
    )

    proj_out_w = weights["proj_out.weight"]
    out_channels = proj_out_w.shape[1]
    output = graph_ops.add_matmul_rhs_constant(network, output, dim, out_channels, proj_out_w)
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_channels, proj_out_b)
    output = network.add_cast(output, trt.float32).get_output(0)
    output = _add_collective(network, output, trt.CollectiveOperation.ALL_GATHER, cp_size)
    output.name = "output"
    network.mark_output(output)

    print(
        f"[flux-dit] Building TRT Ulysses CP engine "
        f"(dim={dim}, joint={num_layers}, single={num_single_layers}, "
        f"img_tokens={num_img_tokens}, text_seq={text_seq_len}, cp={cp_size}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for FLUX DiT Ulysses CP")
    return bytes(plan)


def _validate_context_parallel(
    parallel: ParallelConfig,
    *,
    num_heads: int,
    num_img_tokens: int,
    text_seq_len: int,
) -> None:
    if not parallel.cp_enabled:
        raise ValueError("FLUX Ulysses requires context_parallel_size > 1")
    cp_size = parallel.cp_size
    for name, value in (
        ("num_attention_heads", num_heads),
        ("num_img_tokens", num_img_tokens),
        ("text_seq_len", text_seq_len),
    ):
        if value % cp_size != 0:
            raise ValueError(
                f"FLUX Ulysses requires {name} divisible by context_parallel_size "
                f"({value} vs {cp_size})"
            )


def _slice_rows(network, tensor, start: int, rows: int, width: int):
    return network.add_slice(tensor, (start, 0), (rows, width), (1, 1)).get_output(0)


def _concat_rows(network, first, second):
    cat = network.add_concatenation([first, second])
    cat.axis = 0
    return cat.get_output(0)


def _add_collective(
    network,
    tensor,
    operation,
    cp_size: int,
    *,
    reduce_operation=None,
):
    if reduce_operation is None:
        reduce_operation = trt.ReduceOperation.NONE
    layer = network.add_dist_collective(tensor, operation, reduce_operation, -1, [])
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add {operation} collective")
    layer.num_ranks = int(cp_size)
    return layer.get_output(0)


def _reduce_scatter_replicated(network, tensor, cp_size: int):
    """Shard identical full inputs without multiplying their values by CP."""
    inv_cp = graph_ops.add_constant(network, (1, 1), np.array([1.0 / cp_size], dtype=np.float32))
    scaled = network.add_elementwise(tensor, inv_cp, trt.ElementWiseOperation.PROD).get_output(0)
    return _add_collective(
        network,
        scaled,
        trt.CollectiveOperation.REDUCE_SCATTER,
        cp_size,
        reduce_operation=trt.ReduceOperation.SUM,
    )


def _ulysses_seq_to_head(
    network,
    tensor,
    *,
    local_seq: int,
    num_heads: int,
    head_dim: int,
    cp_size: int,
):
    """[S/CP, H*D] -> [1, H/CP, S, D], matching AITune."""
    local_heads = num_heads // cp_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_seq, cp_size, local_heads, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = _add_collective(
        network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, cp_size
    )
    full_seq = network.add_shuffle(exchanged)
    full_seq.first_transpose = trt.Permutation([2, 0, 1, 3])
    full_seq.reshape_dims = (1, local_heads, local_seq * cp_size, head_dim)
    return full_seq.get_output(0)


def _ulysses_head_to_seq(
    network,
    tensor,
    *,
    local_seq: int,
    num_heads: int,
    head_dim: int,
    cp_size: int,
):
    """[1, H/CP, S, D] -> [S/CP, H*D], inverse of seq-to-head."""
    local_heads = num_heads // cp_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_heads, cp_size, local_seq, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = _add_collective(
        network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, cp_size
    )
    local_rows = network.add_shuffle(exchanged)
    local_rows.first_transpose = trt.Permutation([2, 0, 1, 3])
    local_rows.reshape_dims = (local_seq, num_heads * head_dim)
    return local_rows.get_output(0)


def _ulysses_attention_from_rows(
    network,
    q,
    k,
    v,
    *,
    local_seq: int,
    num_heads: int,
    head_dim: int,
    cp_size: int,
):
    q_full = _ulysses_seq_to_head(
        network,
        q,
        local_seq=local_seq,
        num_heads=num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
    )
    k_full = _ulysses_seq_to_head(
        network,
        k,
        local_seq=local_seq,
        num_heads=num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
    )
    v_full = _ulysses_seq_to_head(
        network,
        v,
        local_seq=local_seq,
        num_heads=num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
    )
    context = graph_ops.add_attention_core(
        network, q_full, k_full, v_full, scale=float(1.0 / np.sqrt(head_dim))
    )
    return _ulysses_head_to_seq(
        network,
        context,
        local_seq=local_seq,
        num_heads=num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
    )
