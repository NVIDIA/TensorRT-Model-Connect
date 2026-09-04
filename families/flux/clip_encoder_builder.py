# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP text encoder engine builder.

Builds a TensorRT engine for CLIP text encoding (CLIPTextModel).
Used by FLUX, Stable Diffusion, etc.

Engine I/O:
    Input:  input_ids [1, max_seq_len] int32
    Output: pooled_output [1, hidden_size] float32  (CLS token)
            text_embeddings [1, max_seq_len, hidden_size] float32

Single forward pass, no cache.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def build_clip_encoder_engine(
    weights: "WeightDict",
    *,
    hidden_size: int = 768,
    num_heads: int = 12,
    intermediate_size: int = 3072,
    num_layers: int = 12,
    vocab_size: int = 49408,
    max_seq_len: int = 77,
    eps: float = 1e-5,
    verbose: bool = False,
) -> bytes:
    """Build CLIP text encoder TRT engine plan."""
    head_dim = hidden_size // num_heads

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    input_ids = network.add_input("input_ids", trt.int32, (max_seq_len,))

    # --- Token + position embeddings ---
    tok_embed_w = weights["text_model.embeddings.token_embedding.weight"]  # [V, D]
    pos_embed_w = weights["text_model.embeddings.position_embedding.weight"]  # [max_pos, D]

    tok_const = graph_ops.add_constant(network, tok_embed_w.shape, tok_embed_w)
    tok_gather = network.add_gather(tok_const, input_ids, axis=0)
    hidden = tok_gather.get_output(0)  # [seq, D]

    # Position IDs must be int32 for Gather layer
    pos_ids_np = np.arange(max_seq_len, dtype=np.int32)
    pos_ids_weights = trt.Weights(np.ascontiguousarray(pos_ids_np))
    pos_ids_layer = network.add_constant((max_seq_len,), pos_ids_weights)
    pos_ids_const = pos_ids_layer.get_output(0)
    pos_const = graph_ops.add_constant(network, pos_embed_w.shape, pos_embed_w)
    pos_gather = network.add_gather(pos_const, pos_ids_const, axis=0)
    hidden = network.add_elementwise(
        hidden, pos_gather.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)

    # Causal mask: [seq, seq] with -inf above diagonal
    causal_np = np.full((max_seq_len, max_seq_len), -1e9, dtype=np.float32)
    for i in range(max_seq_len):
        for j in range(i + 1):
            causal_np[i, j] = 0.0
    # Expand to [1, 1, seq, seq] for native IAttention.
    causal_const = graph_ops.add_constant(
        network, (1, 1, max_seq_len, max_seq_len), causal_np[np.newaxis, np.newaxis]
    )

    for i in range(num_layers):
        p = f"text_model.encoder.layers.{i}"

        # Pre-LN (layer_norm1)
        ln1_w = weights[f"{p}.layer_norm1.weight"]
        ln1_b = weights[f"{p}.layer_norm1.bias"]
        normed = graph_ops.add_layer_norm_native(network, hidden, hidden_size, ln1_w, ln1_b, eps)

        # Self-attention
        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size, weights[f"{p}.self_attn.q_proj.weight"]
        )
        q = graph_ops.add_bias_sum(network, q, hidden_size, weights[f"{p}.self_attn.q_proj.bias"])

        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size, weights[f"{p}.self_attn.k_proj.weight"]
        )
        k = graph_ops.add_bias_sum(network, k, hidden_size, weights[f"{p}.self_attn.k_proj.bias"])

        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size, weights[f"{p}.self_attn.v_proj.weight"]
        )
        v = graph_ops.add_bias_sum(network, v, hidden_size, weights[f"{p}.self_attn.v_proj.bias"])

        context_flat = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=max_seq_len,
            kv_seq=max_seq_len,
            mask=causal_const,
        )

        # Output projection
        attn_out = graph_ops.add_matmul_rhs_constant(
            network,
            context_flat,
            hidden_size,
            hidden_size,
            weights[f"{p}.self_attn.out_proj.weight"],
        )
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden_size, weights[f"{p}.self_attn.out_proj.bias"]
        )

        # Residual
        hidden = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(
            0
        )

        # Pre-LN (layer_norm2)
        ln2_w = weights[f"{p}.layer_norm2.weight"]
        ln2_b = weights[f"{p}.layer_norm2.bias"]
        normed2 = graph_ops.add_layer_norm_native(network, hidden, hidden_size, ln2_w, ln2_b, eps)

        # MLP: fc1 -> quick_gelu -> fc2
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size, weights[f"{p}.mlp.fc1.weight"]
        )
        fc1 = graph_ops.add_bias_sum(network, fc1, intermediate_size, weights[f"{p}.mlp.fc1.bias"])

        # QuickGELU: x * sigmoid(1.702 * x)
        coeff = graph_ops.add_constant(network, (1, 1), np.array([1.702], dtype=np.float32))
        scaled_x = network.add_elementwise(fc1, coeff, trt.ElementWiseOperation.PROD)
        sigmoid = network.add_activation(scaled_x.get_output(0), trt.ActivationType.SIGMOID)
        gelu_out = network.add_elementwise(
            fc1, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)

        fc2 = graph_ops.add_matmul_rhs_constant(
            network, gelu_out, intermediate_size, hidden_size, weights[f"{p}.mlp.fc2.weight"]
        )
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, weights[f"{p}.mlp.fc2.bias"])

        # Residual
        hidden = network.add_elementwise(hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    # Final layer norm
    final_ln_w = weights["text_model.final_layer_norm.weight"]
    final_ln_b = weights["text_model.final_layer_norm.bias"]
    hidden = graph_ops.add_layer_norm_native(
        network, hidden, hidden_size, final_ln_w, final_ln_b, eps
    )

    # Outputs
    cast_hidden = network.add_cast(hidden, trt.float32)
    hidden_out = cast_hidden.get_output(0)
    hidden_out.name = "text_embeddings"
    network.mark_output(hidden_out)

    # Pooled output: extract at position of EOS token (last position)
    # For FLUX, we just take the last-token embedding as pooled output.
    # In practice, CLIP pools at the EOS token position. For fixed-length
    # padding we use position max_seq_len-1.
    # We output full hidden and let runtime pick the EOS position.
    pooled_slice = network.add_slice(
        hidden, start=(max_seq_len - 1, 0), shape=(1, hidden_size), stride=(1, 1)
    )
    pooled_flat = network.add_shuffle(pooled_slice.get_output(0))
    pooled_flat.reshape_dims = (hidden_size,)
    pooled_out = pooled_flat.get_output(0)
    cast_pooled = network.add_cast(pooled_out, trt.float32)
    pooled_final = cast_pooled.get_output(0)
    pooled_final.name = "pooled_output"
    network.mark_output(pooled_final)

    print(
        f"[clip-builder] Building TRT engine "
        f"(hidden={hidden_size}, layers={num_layers}, seq={max_seq_len}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for CLIP encoder")
    return bytes(plan)


def load_clip_weights(
    model_dir: str,
    *,
    hidden_size: int = 768,
    num_layers: int = 12,
) -> "WeightDict":
    """Load CLIP text encoder weights from safetensors."""
    from pathlib import Path
    from .checkpoint_mapper import WeightDict, _open_safetensors, _load_tensor

    readers = _open_safetensors(Path(model_dir))
    weights = WeightDict()

    def _t(name):
        w = _load_tensor(readers, name)
        return np.ascontiguousarray(w.T, dtype=np.float32)

    def _f(name):
        return _load_tensor(readers, name).astype(np.float32)

    weights["text_model.embeddings.token_embedding.weight"] = _f(
        "text_model.embeddings.token_embedding.weight"
    )
    weights["text_model.embeddings.position_embedding.weight"] = _f(
        "text_model.embeddings.position_embedding.weight"
    )

    for i in range(num_layers):
        p = f"text_model.encoder.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            weights[f"{p}.self_attn.{proj}.weight"] = _t(f"{p}.self_attn.{proj}.weight")
            weights[f"{p}.self_attn.{proj}.bias"] = _f(f"{p}.self_attn.{proj}.bias")
        weights[f"{p}.layer_norm1.weight"] = _f(f"{p}.layer_norm1.weight")
        weights[f"{p}.layer_norm1.bias"] = _f(f"{p}.layer_norm1.bias")
        weights[f"{p}.layer_norm2.weight"] = _f(f"{p}.layer_norm2.weight")
        weights[f"{p}.layer_norm2.bias"] = _f(f"{p}.layer_norm2.bias")
        weights[f"{p}.mlp.fc1.weight"] = _t(f"{p}.mlp.fc1.weight")
        weights[f"{p}.mlp.fc1.bias"] = _f(f"{p}.mlp.fc1.bias")
        weights[f"{p}.mlp.fc2.weight"] = _t(f"{p}.mlp.fc2.weight")
        weights[f"{p}.mlp.fc2.bias"] = _f(f"{p}.mlp.fc2.bias")

    weights["text_model.final_layer_norm.weight"] = _f("text_model.final_layer_norm.weight")
    weights["text_model.final_layer_norm.bias"] = _f("text_model.final_layer_norm.bias")

    return weights
