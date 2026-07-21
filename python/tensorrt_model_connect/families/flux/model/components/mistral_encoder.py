# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mistral 3 text encoder engine builder for FLUX.2-dev.

Builds a TensorRT engine for a Mistral-3/LLaMA-style decoder architecture
run as an encoder-only model (single forward pass, no KV cache, no
autoregressive generation).  Multi-layer hidden state extraction produces
a concatenated output suitable for conditioning diffusion models.

Engine I/O:
    Inputs:
        input_ids [1, max_seq_len] int32
        attention_mask [1, max_seq_len] float32 (0.0 for valid, -1e9 for padding)
    Outputs:
        text_embeddings [1, max_seq_len, concat_dim] float32
            where concat_dim = len(extract_layers) * hidden_size
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


def load_mistral_encoder_weights(
    model_dir: str,
    *,
    precision: str = "fp32",
    hidden_size: int = 5120,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate_size: int = 32768,
    num_layers: int = 40,
    vocab_size: int = 131072,
) -> WeightDict:
    """Load Mistral 3 encoder weights from a diffusers-format text_encoder directory.

    Expects: model_dir/model.safetensors (or sharded) with HF Mistral weight keys.
    Returns WeightDict with transposed projections for TRT matmul.

    Auto-detects the weight prefix: either ``model.`` (standalone Mistral) or
    ``language_model.model.`` (Mistral3ForConditionalGeneration multimodal wrapper
    as used in FLUX.2-dev).  Weights are stored under the canonical ``model.``
    prefix regardless of the source prefix.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from ...weights import (
        WeightDict, _open_safetensors, _load_tensor, _has_tensor,
        _target_np_dtype, _transpose_2d)

    model_path = Path(model_dir)
    readers = _open_safetensors(model_path)
    weight_dtype = _target_np_dtype(precision)

    # Auto-detect weight prefix
    if _has_tensor(readers, "model.embed_tokens.weight"):
        src_prefix = "model."
    elif _has_tensor(readers, "language_model.model.embed_tokens.weight"):
        src_prefix = "language_model.model."
    else:
        raise KeyError(
            "Cannot find embedding weights under 'model.' or "
            "'language_model.model.' prefix")

    dst_prefix = "model."

    print(f"[mistral-encoder] Weight prefix: {src_prefix!r}", file=sys.stderr)

    def _src(canonical: str) -> str:
        """Map canonical 'model.X' key to the actual source key."""
        if canonical.startswith(dst_prefix):
            return src_prefix + canonical[len(dst_prefix):]
        return canonical

    weights = WeightDict()
    max_workers = min(8, max(1, os.cpu_count() or 1))

    # Embedding table
    embed = _load_tensor(readers, _src("model.embed_tokens.weight"))
    weights["model.embed_tokens.weight"] = embed.astype(weight_dtype)

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    def _load_layer(i):
        layer = WeightDict()
        prefix = f"model.layers.{i}"

        # RMSNorm weights (no transpose needed — 1-D)
        layer[f"{prefix}.input_layernorm.weight"] = _load_tensor(
            readers, _src(f"{prefix}.input_layernorm.weight")).astype(np.float32)
        layer[f"{prefix}.post_attention_layernorm.weight"] = _load_tensor(
            readers, _src(f"{prefix}.post_attention_layernorm.weight")).astype(np.float32)

        # Self-attention projections — transpose [out, in] -> [in, out]
        for proj, out_size in [
            ("q_proj", q_size),
            ("k_proj", kv_size),
            ("v_proj", kv_size),
        ]:
            key = f"{prefix}.self_attn.{proj}.weight"
            w = _load_tensor(readers, _src(key))
            layer[key] = _transpose_2d(w, key, precision=precision)

        o_key = f"{prefix}.self_attn.o_proj.weight"
        w_o = _load_tensor(readers, _src(o_key))
        layer[o_key] = _transpose_2d(w_o, o_key, precision=precision)

        # MLP projections — transpose [out, in] -> [in, out]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"{prefix}.mlp.{proj}.weight"
            w = _load_tensor(readers, _src(key))
            layer[key] = _transpose_2d(w, key, precision=precision)

        return layer

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
        for future in as_completed(futures):
            weights.update(future.result())

    # Final RMSNorm
    weights["model.norm.weight"] = _load_tensor(
        readers, _src("model.norm.weight")).astype(np.float32)

    return weights


def build_mistral_encoder_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    hidden_size: int = 5120,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate_size: int = 32768,
    num_layers: int = 40,
    vocab_size: int = 131072,
    max_seq_len: int = 512,
    extract_layers: list[int] | tuple[int, ...] = (10, 20, 30),
    eps: float = 1e-5,
    rope_theta: float = 1000000000.0,
    verbose: bool = False,
) -> bytes:
    """Build Mistral 3 encoder TRT engine plan.

    Runs all layers as a single forward pass with bidirectional attention
    (no causal mask).  Extracts hidden states from specified layers and
    concatenates them along the feature dimension.

    Args:
        weights: Weight dict from load_mistral_encoder_weights.
        hidden_size: Model hidden dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (GQA).
        head_dim: Dimension per attention head.
        intermediate_size: SwiGLU FFN inner dimension.
        num_layers: Number of transformer layers.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum input sequence length.
        extract_layers: Layer indices (0-based) whose hidden states are
            extracted and concatenated for the output.
        eps: RMSNorm epsilon.
        verbose: Enable TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes.
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    concat_dim = len(extract_layers) * hidden_size

    rope_cos_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, True, interleaved=True)
    rope_sin_half_np = graph_ops.make_rope_table_half_dim(
        max_seq_len, head_dim, rope_theta, False, interleaved=True)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 128 << 30)

    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        # Use ml_dtypes.bfloat16 so weights are stored in bf16; pairing fp16
        # numpy arrays with bf16 TRT layers triggers strongly-typed mismatch
        # errors (ElementWise SUM requires same input types).
        import ml_dtypes
        work_np_dtype = ml_dtypes.bfloat16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    # --- Inputs ---
    input_ids = network.add_input(
        "input_ids", trt.int32, (1, max_seq_len))
    attention_mask_input = network.add_input(
        "attention_mask", trt.float32, (1, max_seq_len))
    if work_trt_dtype != trt.float32:
        attention_mask_input = network.add_cast(
            attention_mask_input, work_trt_dtype).get_output(0)

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=work_np_dtype),
        dtype=work_np_dtype)
    attn_scale_val = 1.0 / np.sqrt(head_dim)

    # Embedding table [vocab_size, hidden_size]
    embed_table = graph_ops.add_constant(
        network, (vocab_size, hidden_size),
        weights["model.embed_tokens.weight"], dtype=work_np_dtype)

    # --- Embedding lookup ---
    flatten_ids = network.add_shuffle(input_ids)
    flatten_ids.reshape_dims = (max_seq_len,)

    gather = network.add_gather(embed_table, flatten_ids.get_output(0), 0)
    hidden = gather.get_output(0)  # [max_seq_len, hidden_size]

    # --- Attention mask: causal + padding ---
    # Build causal mask: upper triangular with -1e9 above diagonal
    # np.finfo doesn't accept ml_dtypes.bfloat16 ("not inexact"); ml_dtypes
    # ships its own finfo for that case.
    if work_np_dtype == np.float32:
        mask_value = -1e9
    else:
        import ml_dtypes
        finfo = (ml_dtypes.finfo
                 if work_np_dtype == ml_dtypes.bfloat16 else np.finfo)
        mask_value = float(finfo(work_np_dtype).min)
    causal_mask_np = np.zeros((max_seq_len, max_seq_len), dtype=work_np_dtype)
    for i in range(max_seq_len):
        for j in range(i + 1, max_seq_len):
            causal_mask_np[i, j] = mask_value
    causal_mask = graph_ops.add_constant(
        network, (1, max_seq_len, max_seq_len), causal_mask_np,
        dtype=work_np_dtype)

    # Combine with padding mask: reshape [1, max_seq_len] -> [1, 1, max_seq_len]
    attn_mask_3d = network.add_shuffle(attention_mask_input)
    attn_mask_3d.reshape_dims = (1, 1, max_seq_len)

    # Combined mask = causal_mask + padding_mask (both add -1e9 for masked positions)
    combined_mask = network.add_elementwise(
        causal_mask, attn_mask_3d.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- RoPE constants ---
    rope_cos_half = graph_ops.add_constant(
        network, (max_seq_len, head_dim // 2), rope_cos_half_np,
        dtype=work_np_dtype)
    rope_sin_half = graph_ops.add_constant(
        network, (max_seq_len, head_dim // 2), rope_sin_half_np,
        dtype=work_np_dtype)
    rope_position_ids = graph_ops.add_constant(
        network, (max_seq_len,), np.arange(max_seq_len, dtype=np.int32),
        dtype=np.int32)
    combined_mask_4d = network.add_shuffle(combined_mask)
    combined_mask_4d.reshape_dims = (1, 1, max_seq_len, max_seq_len)

    # Collected hidden states for extraction
    extracted = []

    # --- Encoder layers ---
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"

        # === Self-attention sub-layer ===

        # Pre-norm (RMSNorm)
        norm1_gamma = weights[f"{prefix}.input_layernorm.weight"]
        normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size, norm1_gamma, eps_t,
            dtype=work_np_dtype)

        # Q/K/V projections
        w_q = weights[f"{prefix}.self_attn.q_proj.weight"]
        w_k = weights[f"{prefix}.self_attn.k_proj.weight"]
        w_v = weights[f"{prefix}.self_attn.v_proj.weight"]
        w_o = weights[f"{prefix}.self_attn.o_proj.weight"]

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, q_size, w_q, dtype=work_np_dtype)
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_size, w_k, dtype=work_np_dtype)
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, kv_size, w_v, dtype=work_np_dtype)

        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            rope_cos_half, rope_sin_half, rope_position_ids,
            head_dim, interleaved=True, sequence_length=max_seq_len)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            rope_cos_half, rope_sin_half, rope_position_ids,
            head_dim, interleaved=True, sequence_length=max_seq_len)

        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_seq=max_seq_len,
            kv_seq=max_seq_len,
            mask=combined_mask_4d.get_output(0),
            scale=attn_scale_val,
            tag=f"layer.{layer_idx}.attn")

        # O projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context_flat, q_size, hidden_size, w_o,
            dtype=work_np_dtype)

        # Residual
        hidden = network.add_elementwise(
            hidden, attn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # === SwiGLU FFN sub-layer ===

        # Pre-norm (RMSNorm)
        norm2_gamma = weights[f"{prefix}.post_attention_layernorm.weight"]
        ffn_normed = graph_ops.add_rms_norm(
            network, hidden, hidden_size, norm2_gamma, eps_t,
            dtype=work_np_dtype)

        # SwiGLU: silu(gate_proj(x)) * up_proj(x), then down_proj
        w_gate = weights[f"{prefix}.mlp.gate_proj.weight"]
        w_up = weights[f"{prefix}.mlp.up_proj.weight"]
        w_down = weights[f"{prefix}.mlp.down_proj.weight"]

        gate = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, hidden_size, intermediate_size, w_gate,
            dtype=work_np_dtype)
        up = graph_ops.add_matmul_rhs_constant(
            network, ffn_normed, hidden_size, intermediate_size, w_up,
            dtype=work_np_dtype)

        # SiLU activation on gate
        gate_activated = graph_ops.add_activation(
            network, gate, "silu", dtype=work_np_dtype)

        # gate * up
        gated = network.add_elementwise(
            gate_activated, up,
            trt.ElementWiseOperation.PROD)

        # down_proj
        ffn_out = graph_ops.add_matmul_rhs_constant(
            network, gated.get_output(0),
            intermediate_size, hidden_size, w_down, dtype=work_np_dtype)

        # Residual
        hidden = network.add_elementwise(
            hidden, ffn_out,
            trt.ElementWiseOperation.SUM).get_output(0)

        # Extract hidden state if this layer is in the extraction list
        if layer_idx in extract_layers:
            extracted.append(hidden)

    # --- Final RMSNorm ---
    final_norm_gamma = weights["model.norm.weight"]
    hidden = graph_ops.add_rms_norm(
        network, hidden, hidden_size, final_norm_gamma, eps_t,
        dtype=work_np_dtype)

    # --- Multi-layer concatenation ---
    # Concatenate extracted hidden states along the feature dimension:
    # each is [max_seq_len, hidden_size] -> result is [max_seq_len, concat_dim]
    if len(extracted) == 0:
        raise ValueError(
            f"No layers matched extract_layers={extract_layers} "
            f"(num_layers={num_layers})")

    if len(extracted) == 1:
        concat_out = extracted[0]
    else:
        concat_layer = network.add_concatenation(extracted)
        concat_layer.axis = 1  # feature dimension
        concat_out = concat_layer.get_output(0)

    # --- Output ---
    # Reshape to [1, max_seq_len, concat_dim]
    out_reshape = network.add_shuffle(concat_out)
    out_reshape.reshape_dims = (1, max_seq_len, concat_dim)
    out_tensor = out_reshape.get_output(0)
    cast_out = network.add_cast(out_tensor, trt.float32)
    out_final = cast_out.get_output(0)
    out_final.name = "text_embeddings"
    network.mark_output(out_final)

    # --- Build ---
    print(
        f"[mistral-encoder] Building TRT engine "
        f"(hidden={hidden_size}, layers={num_layers}, "
        f"heads={num_heads}/{num_kv_heads}, head_dim={head_dim}, "
        f"seq={max_seq_len}, extract={list(extract_layers)}, "
        f"concat_dim={concat_dim}, precision={precision}) ...",
        file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            "TRT engine serialization failed for Mistral encoder")
    return bytes(plan)
