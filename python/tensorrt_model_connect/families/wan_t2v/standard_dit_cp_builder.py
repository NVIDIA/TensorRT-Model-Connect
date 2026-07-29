# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 DiT Ulysses context-parallel engine builder.

Weights and request inputs stay replicated. The patch-token sequence is
sharded dynamically with REDUCE_SCATTER. Self-attention exchanges sequence
shards for head shards with ALL_TO_ALL, and the final patch rows are restored
with ALL_GATHER. Cross-attention keeps the text context replicated and computes
only rank-local patch queries.

Wan2.1-1.3B has 12 attention heads. CP8 pads Q/K/V to 16 routed heads, exchanges
two heads per rank, and removes the four zero-only heads after attention. This
preserves the 12-head model semantics without requiring the checkpoint head
count to be divisible by the context-parallel world size.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from ...parallel_config import ParallelConfig, normalize_parallel_config
from . import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_standard_dit_engine(
    weights: "WeightDict",
    *,
    dim: int,
    num_heads: int,
    num_layers: int,
    ffn_dim: int,
    context_dim: int,
    num_patches: int,
    text_seq_len: int = 512,
    use_rope: bool = True,
    eps: float = 1e-6,
    precision: str = "fp32",
    verbose: bool = False,
    parallel_config: ParallelConfig | None = None,
) -> bytes:
    """Build one rank-dynamic Wan DiT context-parallel plan."""
    parallel = normalize_parallel_config(parallel_config)
    _validate_context_parallel(parallel, num_patches=num_patches)

    cp_size = parallel.cp_size
    local_patches = num_patches // cp_size
    head_dim = dim // num_heads
    routed_num_heads = _round_up_to_multiple(num_heads, cp_size)
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = (np.float16, trt.float16)
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = (np.float32, trt.float32)
    else:
        raise ValueError(f"Unsupported DiT precision {precision!r}; expected fp32 or fp16")

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

    if work_trt_dtype != trt.float32:
        hidden_inp = network.add_cast(hidden_inp, work_trt_dtype).get_output(0)
        temb_inp = network.add_cast(temb_inp, work_trt_dtype).get_output(0)
        time_embed_inp = network.add_cast(time_embed_inp, work_trt_dtype).get_output(0)
        encoder_hidden = network.add_cast(encoder_hidden, work_trt_dtype).get_output(0)
        if cross_attn_mask is not None:
            cross_attn_mask = network.add_cast(cross_attn_mask, work_trt_dtype).get_output(0)

    eps_t = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([eps], dtype=work_np_dtype),
        dtype=work_np_dtype,
    )
    hidden = _reduce_scatter_replicated(network, hidden_inp, cp_size, work_np_dtype)

    rotary_cos = rotary_sin = None
    if use_rope:
        rotary_cos_full = network.add_input("rotary_cos", trt.float32, (num_patches, head_dim))
        rotary_sin_full = network.add_input("rotary_sin", trt.float32, (num_patches, head_dim))
        if work_trt_dtype != trt.float32:
            rotary_cos_full = network.add_cast(rotary_cos_full, work_trt_dtype).get_output(0)
            rotary_sin_full = network.add_cast(rotary_sin_full, work_trt_dtype).get_output(0)
        rotary_cos = _reduce_scatter_replicated(network, rotary_cos_full, cp_size, work_np_dtype)
        rotary_sin = _reduce_scatter_replicated(network, rotary_sin_full, cp_size, work_np_dtype)

    for layer_idx in range(num_layers):
        prefix = f"blocks.{layer_idx}"
        sst = weights[f"{prefix}.scale_shift_table"]
        sst_const = graph_ops.add_constant(
            network,
            (1, 6 * dim),
            sst.reshape(1, 6 * dim),
            dtype=work_np_dtype,
        )
        modulation = network.add_elementwise(sst_const, temb_inp, trt.ElementWiseOperation.SUM)
        chunks = []
        for index in range(6):
            part = network.add_slice(
                modulation.get_output(0),
                start=(0, index * dim),
                shape=(1, dim),
                stride=(1, 1),
            )
            chunks.append(part.get_output(0))
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = chunks

        normed = graph_ops.add_adaptive_layernorm(
            network,
            hidden,
            scale_sa,
            shift_sa,
            dim,
            eps,
            dtype=work_np_dtype,
        )
        q = _linear(network, normed, dim, dim, weights, f"{prefix}.attn1.to_q", work_np_dtype)
        k = _linear(network, normed, dim, dim, weights, f"{prefix}.attn1.to_k", work_np_dtype)
        v = _linear(network, normed, dim, dim, weights, f"{prefix}.attn1.to_v", work_np_dtype)

        q_norm_w = weights.get(f"{prefix}.attn1.norm_q.weight")
        k_norm_w = weights.get(f"{prefix}.attn1.norm_k.weight")
        if q_norm_w is not None:
            q = graph_ops.add_rms_norm(network, q, dim, q_norm_w, eps_t, dtype=work_np_dtype)
        if k_norm_w is not None:
            k = graph_ops.add_rms_norm(network, k, dim, k_norm_w, eps_t, dtype=work_np_dtype)
        if use_rope:
            q = graph_ops.add_apply_rope_native_from_full_cache(
                network,
                q,
                num_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                local_patches,
                interleaved=True,
            )
            k = graph_ops.add_apply_rope_native_from_full_cache(
                network,
                k,
                num_heads,
                head_dim,
                rotary_cos,
                rotary_sin,
                local_patches,
                interleaved=True,
            )

        context_flat = _ulysses_attention_from_rows(
            network,
            q,
            k,
            v,
            local_seq=local_patches,
            num_heads=num_heads,
            head_dim=head_dim,
            routed_num_heads=routed_num_heads,
            cp_size=cp_size,
            dtype=work_np_dtype,
        )
        attn_out = _linear(
            network,
            context_flat,
            dim,
            dim,
            weights,
            f"{prefix}.attn1.to_out.0",
            work_np_dtype,
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
                cross_norm_b if cross_norm_b is not None else np.zeros(dim, dtype=work_np_dtype),
                eps_t,
                dtype=work_np_dtype,
            )
        else:
            cross_normed = hidden

        cross_q = _linear(
            network,
            cross_normed,
            dim,
            dim,
            weights,
            f"{prefix}.attn2.to_q",
            work_np_dtype,
        )
        add_k_proj_w = weights.get(f"{prefix}.attn2.add_k_proj.weight")
        if add_k_proj_w is not None:
            cross_k = _linear(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.add_k_proj",
                work_np_dtype,
            )
            cross_v = _linear(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.add_v_proj",
                work_np_dtype,
            )
        else:
            cross_k = _linear(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.to_k",
                work_np_dtype,
            )
            cross_v = _linear(
                network,
                encoder_hidden,
                context_dim,
                dim,
                weights,
                f"{prefix}.attn2.to_v",
                work_np_dtype,
            )

        cq_norm = weights.get(f"{prefix}.attn2.norm_q.weight")
        ck_norm = weights.get(f"{prefix}.attn2.norm_k.weight")
        if cq_norm is not None:
            cross_q = graph_ops.add_rms_norm(
                network, cross_q, dim, cq_norm, eps_t, dtype=work_np_dtype
            )
        ck_added_norm = weights.get(f"{prefix}.attn2.norm_added_k.weight")
        if ck_added_norm is not None:
            cross_k = graph_ops.add_rms_norm(
                network, cross_k, dim, ck_added_norm, eps_t, dtype=work_np_dtype
            )
        elif ck_norm is not None:
            cross_k = graph_ops.add_rms_norm(
                network, cross_k, dim, ck_norm, eps_t, dtype=work_np_dtype
            )

        cross_mask_4d = None
        if cross_attn_mask is not None:
            cross_mask = network.add_shuffle(cross_attn_mask)
            cross_mask.reshape_dims = (1, 1, 1, text_seq_len)
            cross_mask_4d = cross_mask.get_output(0)
        cross_context = graph_ops.add_attention_from_rows(
            network,
            cross_q,
            cross_k,
            cross_v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=local_patches,
            kv_seq=text_seq_len,
            mask=cross_mask_4d,
            tag=f"{prefix}.attn2",
        )
        cross_out = _linear(
            network,
            cross_context,
            dim,
            dim,
            weights,
            f"{prefix}.attn2.to_out.0",
            work_np_dtype,
        )
        hidden = network.add_elementwise(
            hidden, cross_out, trt.ElementWiseOperation.SUM
        ).get_output(0)

        ffn_normed = graph_ops.add_adaptive_layernorm(
            network,
            hidden,
            scale_ff,
            shift_ff,
            dim,
            eps,
            dtype=work_np_dtype,
        )
        ffn_fc1 = _linear(
            network,
            ffn_normed,
            dim,
            ffn_dim,
            weights,
            f"{prefix}.ffn.net.0.proj",
            work_np_dtype,
        )
        ffn_act = graph_ops.add_gelu_new(network, ffn_fc1, dtype=work_np_dtype)
        ffn_fc2 = _linear(
            network,
            ffn_act,
            ffn_dim,
            dim,
            weights,
            f"{prefix}.ffn.net.2",
            work_np_dtype,
        )
        gated_ff = network.add_elementwise(ffn_fc2, gate_ff, trt.ElementWiseOperation.PROD)
        hidden = network.add_elementwise(
            hidden, gated_ff.get_output(0), trt.ElementWiseOperation.SUM
        ).get_output(0)

    final_sst = weights["scale_shift_table"]
    final_sst_const = graph_ops.add_constant(
        network,
        (1, 2 * dim),
        final_sst.reshape(1, 2 * dim),
        dtype=work_np_dtype,
    )
    time_embed_tiled = network.add_concatenation([time_embed_inp, time_embed_inp])
    time_embed_tiled.axis = 1
    final_modulation = network.add_elementwise(
        final_sst_const,
        time_embed_tiled.get_output(0),
        trt.ElementWiseOperation.SUM,
    )
    final_shift = network.add_slice(
        final_modulation.get_output(0),
        start=(0, 0),
        shape=(1, dim),
        stride=(1, 1),
    )
    final_scale = network.add_slice(
        final_modulation.get_output(0),
        start=(0, dim),
        shape=(1, dim),
        stride=(1, 1),
    )
    hidden = graph_ops.add_adaptive_layernorm(
        network,
        hidden,
        final_scale.get_output(0),
        final_shift.get_output(0),
        dim,
        eps,
        dtype=work_np_dtype,
    )
    proj_out_w = weights["proj_out.weight"]
    out_dim = proj_out_w.shape[1]
    output = graph_ops.add_matmul_rhs_constant(
        network, hidden, dim, out_dim, proj_out_w, dtype=work_np_dtype
    )
    proj_out_b = weights.get("proj_out.bias")
    if proj_out_b is not None:
        output = graph_ops.add_bias_sum(network, output, out_dim, proj_out_b, dtype=work_np_dtype)
    output = network.add_cast(output, trt.float32).get_output(0)
    output = _add_collective(network, output, trt.CollectiveOperation.ALL_GATHER, cp_size)
    output.name = "output"
    network.mark_output(output)

    print(
        f"[wan-dit] Building TRT context-parallel engine "
        f"(dim={dim}, layers={num_layers}, patches={num_patches}, "
        f"cp={cp_size}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for Wan DiT context parallel")
    return bytes(plan)


def _validate_context_parallel(
    parallel: ParallelConfig,
    *,
    num_patches: int,
) -> None:
    if not parallel.cp_enabled:
        raise ValueError(
            "Wan context parallel requires parallel.mode=context_parallel and cp_size > 1"
        )
    if num_patches % parallel.cp_size != 0:
        raise ValueError(
            "Wan context parallel requires num_patches divisible by cp_size "
            f"({num_patches} vs {parallel.cp_size})"
        )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _linear(
    network,
    inp,
    in_dim: int,
    out_dim: int,
    weights,
    prefix: str,
    dtype,
):
    out = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        in_dim,
        out_dim,
        weights[f"{prefix}.weight"],
        dtype=dtype,
    )
    bias = weights.get(f"{prefix}.bias")
    if bias is not None:
        out = graph_ops.add_bias_sum(network, out, out_dim, bias, dtype=dtype)
    return out


def _add_collective(
    network,
    tensor,
    operation,
    cp_size: int,
    *,
    reduce_operation=None,
):
    add_collective = getattr(network, "add_dist_collective", None)
    if add_collective is None:
        raise RuntimeError(
            "Wan context parallel requires TensorRT 11.0+ Python bindings "
            "with INetworkDefinition.add_dist_collective"
        )
    if reduce_operation is None:
        reduce_operation = getattr(trt.ReduceOperation, "NONE", trt.ReduceOperation.SUM)
    layer = add_collective(tensor, operation, reduce_operation, -1, [])
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add {operation} collective")
    if not hasattr(layer, "num_ranks"):
        raise RuntimeError(
            "Wan context parallel requires TensorRT 11.0+ Python bindings "
            "with IDistCollectiveLayer.num_ranks"
        )
    layer.num_ranks = int(cp_size)
    return layer.get_output(0)


def _pad_attention_heads(
    network,
    tensor,
    *,
    local_seq: int,
    num_heads: int,
    routed_num_heads: int,
    head_dim: int,
    dtype,
):
    """Append zero-only heads so the routed count is divisible by CP."""
    pad_features = (routed_num_heads - num_heads) * head_dim
    if pad_features == 0:
        return tensor
    zero_source = network.add_slice(
        tensor,
        start=(0, 0),
        shape=(local_seq, pad_features),
        stride=(1, 1),
    ).get_output(0)
    zero = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([0.0], dtype=dtype),
        dtype=dtype,
    )
    padded_rows = network.add_elementwise(
        zero_source,
        zero,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    padded = network.add_concatenation([tensor, padded_rows])
    padded.axis = 1
    return padded.get_output(0)


def _ulysses_seq_to_head(
    network,
    tensor,
    *,
    local_seq: int,
    routed_num_heads: int,
    head_dim: int,
    cp_size: int,
):
    """Exchange [S/CP, H*D] sequence shards for [1, H/CP, S, D]."""
    local_heads = routed_num_heads // cp_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_seq, cp_size, local_heads, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = _add_collective(
        network,
        routed.get_output(0),
        trt.CollectiveOperation.ALL_TO_ALL,
        cp_size,
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
    routed_num_heads: int,
    head_dim: int,
    cp_size: int,
):
    """Invert the Ulysses exchange back to local sequence rows."""
    local_heads = routed_num_heads // cp_size
    routed = network.add_shuffle(tensor)
    routed.reshape_dims = (local_heads, cp_size, local_seq, head_dim)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = _add_collective(
        network,
        routed.get_output(0),
        trt.CollectiveOperation.ALL_TO_ALL,
        cp_size,
    )
    local_rows = network.add_shuffle(exchanged)
    local_rows.first_transpose = trt.Permutation([2, 0, 1, 3])
    local_rows.reshape_dims = (local_seq, routed_num_heads * head_dim)
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
    routed_num_heads: int,
    cp_size: int,
    dtype,
):
    """Run full-sequence attention over routed real and zero-only heads."""
    routed_inputs = [
        _pad_attention_heads(
            network,
            tensor,
            local_seq=local_seq,
            num_heads=num_heads,
            routed_num_heads=routed_num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        for tensor in (q, k, v)
    ]
    q_full, k_full, v_full = [
        _ulysses_seq_to_head(
            network,
            tensor,
            local_seq=local_seq,
            routed_num_heads=routed_num_heads,
            head_dim=head_dim,
            cp_size=cp_size,
        )
        for tensor in routed_inputs
    ]
    context = graph_ops.add_attention_core(
        network,
        q_full,
        k_full,
        v_full,
        scale=float(1.0 / np.sqrt(head_dim)),
    )
    local_rows = _ulysses_head_to_seq(
        network,
        context,
        local_seq=local_seq,
        routed_num_heads=routed_num_heads,
        head_dim=head_dim,
        cp_size=cp_size,
    )
    if routed_num_heads == num_heads:
        return local_rows
    return network.add_slice(
        local_rows,
        start=(0, 0),
        shape=(local_seq, num_heads * head_dim),
        stride=(1, 1),
    ).get_output(0)


def _reduce_scatter_replicated(
    network,
    tensor,
    cp_size: int,
    dtype,
):
    """Select rank-local rows from identical full inputs without scaling them."""
    inv_cp = graph_ops.add_constant(
        network,
        (1, 1),
        np.array([1.0 / cp_size], dtype=dtype),
        dtype=dtype,
    )
    scaled = network.add_elementwise(tensor, inv_cp, trt.ElementWiseOperation.PROD).get_output(0)
    return _add_collective(
        network,
        scaled,
        trt.CollectiveOperation.REDUCE_SCATTER,
        cp_size,
        reduce_operation=trt.ReduceOperation.SUM,
    )
