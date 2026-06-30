"""Shared T5 encoder engine builder.

Builds a TensorRT engine for a T5-style text encoder (UMT5, mT5, T5).
Reusable by: Wan2.1, FLUX, SD3, Hunyuan Video, CogVideoX.

Engine I/O:
    Input:  input_ids [1, max_seq_len] int32
    Output: text_embeddings [1, max_seq_len, d_model] float32

Single forward pass (no cache), like vision encoder.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from .. import model as graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...weights import WeightDict


def build_t5_encoder_engine(
    weights: WeightDict,
    *,
    d_model: int = 4096,
    num_heads: int = 64,
    d_kv: int = 64,
    d_ff: int = 10240,
    num_layers: int = 24,
    vocab_size: int = 250112,
    max_seq_len: int = 512,
    relative_attention_num_buckets: int = 32,
    relative_attention_max_distance: int = 128,
    eps: float = 1e-6,
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
) -> bytes:
    """Build T5 encoder TRT engine plan.

    Args:
        weights: Weight dict with T5 encoder weights. Expected keys:
            - shared.weight: [vocab_size, d_model] embedding
            - encoder.block.{i}.layer.0.SelfAttention.q/k/v/o.weight
            - encoder.block.{i}.layer.0.layer_norm.weight
            - encoder.block.{i}.layer.1.DenseReluDense.wi_0/wi_1/wo.weight
            - encoder.block.{i}.layer.1.layer_norm.weight
            - encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
            - encoder.final_layer_norm.weight
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        d_kv: Key/value dimension per head.
        d_ff: Feed-forward inner dimension.
        num_layers: Number of encoder layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        relative_attention_num_buckets: T5 relative position bias buckets.
        relative_attention_max_distance: Max distance for relative position.
        eps: RMSNorm epsilon.
        max_batch_size: Maximum runtime batch size. ``1`` (default) preserves
            the legacy static-shape engine byte-for-byte. ``>1`` enables a
            dynamic leading batch dim via a single TRT optimization profile
            covering ``[1, max_batch_size]`` (design Decision C, default
            cap is 8 for text encoders).
        opt_batch_size: Profile ``kOPT``. Defaults to ``min(max_batch_size,
            4)``; only consulted when ``max_batch_size > 1``.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1 (got {max_batch_size})")
    if max_batch_size > 1:
        return _build_t5_encoder_engine_batched(
            weights,
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            eps=eps,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
            verbose=verbose,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs ---
    input_ids = network.add_input("input_ids", trt.int32, (1, max_seq_len))
    # Attention mask: [1, max_seq_len] float32, 0.0 for valid, -1e9 for padding
    attention_mask_input = network.add_input("attention_mask", trt.float32, (1, max_seq_len))

    # --- Constants ---
    eps_t = graph_ops.add_constant(network, (1, 1), np.array([eps], dtype=np.float32))

    # Embedding table [vocab_size, d_model]
    embed_table = graph_ops.add_constant(network, (vocab_size, d_model), weights["shared.weight"])

    # --- Embedding lookup ---
    # Flatten input_ids to [max_seq_len] for gather
    flatten_ids = network.add_shuffle(input_ids)
    flatten_ids.reshape_dims = (max_seq_len,)

    gather = network.add_gather(embed_table, flatten_ids.get_output(0), 0)
    hidden = gather.get_output(0)  # [max_seq_len, d_model]

    # --- Relative position bias ---
    # Precompute bucket indices [max_seq_len, max_seq_len]
    bucket_indices = graph_ops.make_t5_relative_position_bias(
        num_heads,
        max_seq_len,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    # Reshape attention mask: [1, max_seq_len] -> [1, 1, max_seq_len]
    attn_mask_3d = network.add_shuffle(attention_mask_input)
    attn_mask_3d.reshape_dims = (1, 1, max_seq_len)

    # UMT5: each layer has its own relative_attention_bias (unlike T5 which shares layer 0's)
    # Precompute per-layer bias tables as constants
    per_layer_bias = []
    for layer_idx in range(num_layers):
        bias_key = f"encoder.block.{layer_idx}.layer.0.SelfAttention.relative_attention_bias.weight"
        if bias_key in weights:
            rel_bias_weight = weights[bias_key]
        else:
            # Fallback: use layer 0's bias (vanilla T5 behavior)
            rel_bias_weight = weights[
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ]

        bias_table = np.zeros((num_heads, max_seq_len, max_seq_len), dtype=np.float32)
        for q_pos in range(max_seq_len):
            for k_pos in range(max_seq_len):
                bucket = bucket_indices[q_pos, k_pos]
                for h in range(num_heads):
                    bias_table[h, q_pos, k_pos] = rel_bias_weight[bucket, h]

        rel_bias_const = graph_ops.add_constant(
            network, (num_heads, max_seq_len, max_seq_len), bias_table
        )

        # Combine position bias + attention mask
        position_bias_masked = network.add_elementwise(
            rel_bias_const, attn_mask_3d.get_output(0), trt.ElementWiseOperation.SUM
        )
        per_layer_bias.append(position_bias_masked.get_output(0))

    # --- Encoder layers ---
    attention_size = num_heads * d_kv

    for layer_idx in range(num_layers):
        prefix = f"encoder.block.{layer_idx}"

        # === Self-attention sub-layer ===
        # Pre-norm (RMSNorm)
        norm1_gamma = weights[f"{prefix}.layer.0.layer_norm.weight"]
        normed = graph_ops.add_rms_norm(network, hidden, d_model, norm1_gamma, eps_t)

        # Q, K, V projections
        # T5 uses no bias on Q/K/V/O projections
        w_q = weights[f"{prefix}.layer.0.SelfAttention.q.weight"]
        w_k = weights[f"{prefix}.layer.0.SelfAttention.k.weight"]
        w_v = weights[f"{prefix}.layer.0.SelfAttention.v.weight"]
        w_o = weights[f"{prefix}.layer.0.SelfAttention.o.weight"]

        # Self-attention with relative position bias
        q = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_q)
        k = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_k)
        v = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_v)

        # T5 attention is intentionally unscaled; relative position bias and
        # padding mask are folded into the native IAttention additive mask.
        mask_4d = network.add_shuffle(per_layer_bias[layer_idx])
        mask_4d.reshape_dims = (1, num_heads, max_seq_len, max_seq_len)
        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=d_kv,
            q_seq=max_seq_len,
            kv_seq=max_seq_len,
            mask=mask_4d.get_output(0),
            scale=1.0,
        )

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, attention_size, d_model, w_o
        )

        # Residual
        hidden = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(
            0
        )

        # === FFN sub-layer ===
        # Pre-norm (RMSNorm)
        norm2_gamma = weights[f"{prefix}.layer.1.layer_norm.weight"]
        ffn_normed = graph_ops.add_rms_norm(network, hidden, d_model, norm2_gamma, eps_t)

        # T5 gated GELU FFN: gelu(wi_0(x)) * wi_1(x), then wo
        w_fc1 = weights[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"]
        w_fc1_gate = weights[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"]
        w_fc2 = weights[f"{prefix}.layer.1.DenseReluDense.wo.weight"]

        fc1 = graph_ops.add_matmul_rhs_constant(network, ffn_normed, d_model, d_ff, w_fc1)
        fc1_gate = graph_ops.add_matmul_rhs_constant(network, ffn_normed, d_model, d_ff, w_fc1_gate)

        # GELU activation on fc1, multiply with gate
        activated = graph_ops.add_gelu_new(network, fc1)
        gated = network.add_elementwise(activated, fc1_gate, trt.ElementWiseOperation.PROD)

        # Output projection
        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), d_ff, d_model, w_fc2
        )

        # Residual
        hidden = network.add_elementwise(hidden, ffn_out, trt.ElementWiseOperation.SUM).get_output(
            0
        )

    # --- Final norm ---
    final_norm_gamma = weights["encoder.final_layer_norm.weight"]
    hidden = graph_ops.add_rms_norm(network, hidden, d_model, final_norm_gamma, eps_t)

    # --- Output ---
    # Reshape to [1, max_seq_len, d_model]
    out_reshape = network.add_shuffle(hidden)
    out_reshape.reshape_dims = (1, max_seq_len, d_model)
    out_tensor = out_reshape.get_output(0)
    cast_out = network.add_cast(out_tensor, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    # --- Build ---
    print(
        f"[t5-encoder] Building TRT engine "
        f"(d_model={d_model}, layers={num_layers}, seq={max_seq_len}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for T5 encoder")
    return bytes(plan)


# ============================================================================
# Dynamic-batch path
# ============================================================================
#
# When ``max_batch_size > 1`` we build a separate engine where every input
# carries a leading dynamic batch dim and a single
# :class:`IOptimizationProfile` spans ``[1, max_batch_size]`` with
# ``kOPT = min(max_batch_size, 4)`` (design Decisions A and C). The legacy
# static-shape path above is left untouched so ``max_batch_size == 1`` stays
# byte-for-byte identical.


def _build_t5_encoder_engine_batched(
    weights: WeightDict,
    *,
    d_model: int,
    num_heads: int,
    d_kv: int,
    d_ff: int,
    num_layers: int,
    vocab_size: int,
    max_seq_len: int,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
    eps: float,
    max_batch_size: int,
    opt_batch_size: int | None,
    verbose: bool,
) -> bytes:
    """Build a dynamic-leading-batch T5 encoder TRT engine."""
    from .....engine_builder import add_dynamic_batch_profile

    opt_batch = min(max_batch_size, 4) if opt_batch_size is None else opt_batch_size

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # --- Inputs (dynamic leading batch dim) ---
    input_ids = network.add_input("input_ids", trt.int32, (-1, max_seq_len))
    attention_mask_input = network.add_input("attention_mask", trt.float32, (-1, max_seq_len))

    add_dynamic_batch_profile(
        builder,
        config,
        network,
        input_names=["input_ids", "attention_mask"],
        max_batch=max_batch_size,
        opt_batch=opt_batch,
        static_shape={
            "input_ids": (max_seq_len,),
            "attention_mask": (max_seq_len,),
        },
    )

    # --- Constants ---
    eps_t = graph_ops.add_constant(network, (1, 1, 1), np.array([eps], dtype=np.float32))

    # Embedding table [vocab_size, d_model]; gather over [B, S] -> [B, S, d_model]
    embed_table = graph_ops.add_constant(network, (vocab_size, d_model), weights["shared.weight"])

    gather = network.add_gather(embed_table, input_ids, 0)
    hidden = gather.get_output(0)  # [B, S, d_model]

    # --- Relative position bias (constant across batch) ---
    bucket_indices = graph_ops.make_t5_relative_position_bias(
        num_heads,
        max_seq_len,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    # Attention mask: [B, S] -> [B, 1, 1, S] for broadcast over heads & queries.
    attn_mask_4d = network.add_shuffle(attention_mask_input)
    attn_mask_4d.reshape_dims = (-1, 1, 1, max_seq_len)
    attn_mask = attn_mask_4d.get_output(0)

    per_layer_bias = []
    for layer_idx in range(num_layers):
        bias_key = f"encoder.block.{layer_idx}.layer.0.SelfAttention.relative_attention_bias.weight"
        if bias_key in weights:
            rel_bias_weight = weights[bias_key]
        else:
            rel_bias_weight = weights[
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
            ]

        bias_table = np.zeros((num_heads, max_seq_len, max_seq_len), dtype=np.float32)
        for q_pos in range(max_seq_len):
            for k_pos in range(max_seq_len):
                bucket = bucket_indices[q_pos, k_pos]
                for h in range(num_heads):
                    bias_table[h, q_pos, k_pos] = rel_bias_weight[bucket, h]

        # Constant [1, num_heads, S, S] broadcasts cleanly across batch.
        rel_bias_const = graph_ops.add_constant(
            network,
            (1, num_heads, max_seq_len, max_seq_len),
            bias_table.reshape(1, num_heads, max_seq_len, max_seq_len),
        )
        # Combine constant position bias with per-sample padding mask.
        position_bias_masked = network.add_elementwise(
            rel_bias_const, attn_mask, trt.ElementWiseOperation.SUM
        )
        per_layer_bias.append(position_bias_masked.get_output(0))

    attention_size = num_heads * d_kv

    for layer_idx in range(num_layers):
        prefix = f"encoder.block.{layer_idx}"

        # === Self-attention ===
        norm1_gamma = weights[f"{prefix}.layer.0.layer_norm.weight"]
        normed = graph_ops.add_rms_norm_last_dim(network, hidden, d_model, norm1_gamma, eps_t)

        w_q = weights[f"{prefix}.layer.0.SelfAttention.q.weight"]
        w_k = weights[f"{prefix}.layer.0.SelfAttention.k.weight"]
        w_v = weights[f"{prefix}.layer.0.SelfAttention.v.weight"]
        w_o = weights[f"{prefix}.layer.0.SelfAttention.o.weight"]

        q = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_q)
        k = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_k)
        v = graph_ops.add_matmul_rhs_constant(network, normed, d_model, attention_size, w_v)

        # [B, S, H*D] -> [B, H, S, D] for IAttention
        q_4d = _to_heads_4d_batched(network, q, num_heads, d_kv, max_seq_len)
        k_4d = _to_heads_4d_batched(network, k, num_heads, d_kv, max_seq_len)
        v_4d = _to_heads_4d_batched(network, v, num_heads, d_kv, max_seq_len)

        # T5 attention is unscaled; mask carries the only score adjustments.
        ctx_4d = graph_ops.add_attention_core(
            network, q_4d, k_4d, v_4d, causal=False, mask=per_layer_bias[layer_idx], scale=1.0
        )
        context_flat = _to_rows_3d_batched(network, ctx_4d, num_heads, d_kv, max_seq_len)

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, attention_size, d_model, w_o
        )

        hidden = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(
            0
        )

        # === FFN ===
        norm2_gamma = weights[f"{prefix}.layer.1.layer_norm.weight"]
        ffn_normed = graph_ops.add_rms_norm_last_dim(network, hidden, d_model, norm2_gamma, eps_t)

        w_fc1 = weights[f"{prefix}.layer.1.DenseReluDense.wi_0.weight"]
        w_fc1_gate = weights[f"{prefix}.layer.1.DenseReluDense.wi_1.weight"]
        w_fc2 = weights[f"{prefix}.layer.1.DenseReluDense.wo.weight"]

        fc1 = graph_ops.add_matmul_rhs_constant(network, ffn_normed, d_model, d_ff, w_fc1)
        fc1_gate = graph_ops.add_matmul_rhs_constant(network, ffn_normed, d_model, d_ff, w_fc1_gate)

        activated = graph_ops.add_gelu_new(network, fc1)
        gated = network.add_elementwise(activated, fc1_gate, trt.ElementWiseOperation.PROD)

        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0), d_ff, d_model, w_fc2
        )

        hidden = network.add_elementwise(hidden, ffn_out, trt.ElementWiseOperation.SUM).get_output(
            0
        )

    # --- Final norm ---
    final_norm_gamma = weights["encoder.final_layer_norm.weight"]
    hidden = graph_ops.add_rms_norm_last_dim(network, hidden, d_model, final_norm_gamma, eps_t)

    # --- Output ---
    cast_out = network.add_cast(hidden, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    print(
        f"[t5-encoder] Building dynamic-batch TRT engine "
        f"(B=1..{max_batch_size}, opt={opt_batch}, d_model={d_model}, "
        f"layers={num_layers}, seq={max_seq_len}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for dynamic-batch T5 encoder")
    return bytes(plan)


def _to_heads_4d_batched(network, x, num_heads: int, head_dim: int, seq_len: int):
    """``[B, S, H*D]`` -> ``[B, H, S, D]``."""
    r = network.add_shuffle(x)
    r.reshape_dims = (-1, seq_len, num_heads, head_dim)
    r.second_transpose = trt.Permutation([0, 2, 1, 3])
    return r.get_output(0)


def _to_rows_3d_batched(network, x, num_heads: int, head_dim: int, seq_len: int):
    """``[B, H, S, D]`` -> ``[B, S, H*D]``."""
    r = network.add_shuffle(x)
    r.first_transpose = trt.Permutation([0, 2, 1, 3])
    r.reshape_dims = (-1, seq_len, num_heads * head_dim)
    return r.get_output(0)


def load_t5_weights(
    model_dir: str,
    *,
    precision: str = "fp32",
    d_model: int = 4096,
    num_heads: int = 64,
    d_kv: int = 64,
    d_ff: int = 10240,
    num_layers: int = 24,
    vocab_size: int = 250112,
) -> WeightDict:
    """Load T5 encoder weights from a diffusers-format text_encoder directory.

    Expects: model_dir/model.safetensors (or sharded) with HF T5 weight keys.
    Returns WeightDict with transposed projections for TRT matmul.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from ...weights import (
        WeightDict,
        _has_tensor,
        _load_tensor,
        _open_safetensors,
        _target_np_dtype,
    )

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()

    # Embedding
    embed = _load_tensor(readers, "shared.weight")
    weights["shared.weight"] = embed.astype(target_dtype)

    # Relative attention bias — UMT5 has per-layer bias, T5 has only layer 0
    for i in range(num_layers):
        bias_key = f"encoder.block.{i}.layer.0.SelfAttention.relative_attention_bias.weight"
        if _has_tensor(readers, bias_key):
            weights[bias_key] = _load_tensor(readers, bias_key).astype(np.float32)

    def _load_layer(i: int) -> tuple[int, WeightDict]:
        prefix = f"encoder.block.{i}"
        layer = WeightDict()

        # Self-attention weights (transpose for TRT matmul)
        for proj in ("q", "k", "v", "o"):
            key = f"{prefix}.layer.0.SelfAttention.{proj}.weight"
            w = _load_tensor(readers, key)
            # HF shape: [out, in] -> transpose to [in, out]
            layer[key] = np.ascontiguousarray(w.T, dtype=target_dtype)

        # Self-attention layer norm
        norm_key = f"{prefix}.layer.0.layer_norm.weight"
        layer[norm_key] = _load_tensor(readers, norm_key).astype(np.float32)

        # FFN weights (transpose for TRT matmul)
        for proj in ("wi_0", "wi_1", "wo"):
            key = f"{prefix}.layer.1.DenseReluDense.{proj}.weight"
            w = _load_tensor(readers, key)
            layer[key] = np.ascontiguousarray(w.T, dtype=target_dtype)

        # FFN layer norm
        norm_key = f"{prefix}.layer.1.layer_norm.weight"
        layer[norm_key] = _load_tensor(readers, norm_key).astype(np.float32)

        return i, layer

    layer_results: list[tuple[int, WeightDict] | None] = [None] * num_layers
    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
        for future in as_completed(futures):
            i, layer = future.result()
            layer_results[i] = (i, layer)

    for result in layer_results:
        if result is not None:
            weights.update(result[1])

    # Final layer norm
    weights["encoder.final_layer_norm.weight"] = _load_tensor(
        readers, "encoder.final_layer_norm.weight"
    ).astype(np.float32)

    return weights
