# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 CLIP text encoder TensorRT builder.

SAM3 image PCS uses text prompts.  This builder covers the model-card text
prompt branch by compiling the CLIP text tower and its SAM3 projection into a
TensorRT plan.  The vision, DETR, and mask-decoder engines are separate work.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def _trt():
    return trt_compat.get_trt()


def build_sam3_text_encoder_engine(
    weights: "WeightDict",
    *,
    hidden_size: int,
    projected_size: int,
    num_heads: int,
    intermediate_size: int,
    num_layers: int,
    vocab_size: int,
    max_seq_len: int,
    eps: float,
    precision: str = "fp32",
    hidden_act: str = "gelu",
    verbose: bool = False,
) -> bytes:
    """Build the SAM3 text-prompt encoder plan with TensorRT APIs."""
    del vocab_size
    trt = _trt()
    head_dim = hidden_size // num_heads
    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    input_ids = network.add_input("input_ids", trt.int32, (max_seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (max_seq_len,))

    tok_embed_w = weights["text_model.embeddings.token_embedding.weight"]
    pos_embed_w = weights["text_model.embeddings.position_embedding.weight"]

    tok_const = graph_ops.add_constant(
        network, tok_embed_w.shape, tok_embed_w, dtype=work_np_dtype)
    tok_gather = network.add_gather(tok_const, input_ids, axis=0)
    hidden = tok_gather.get_output(0)

    pos_ids_np = np.arange(max_seq_len, dtype=np.int32)
    pos_ids = network.add_constant(
        (max_seq_len,), trt.Weights(np.ascontiguousarray(pos_ids_np))).get_output(0)
    pos_const = graph_ops.add_constant(
        network, pos_embed_w.shape, pos_embed_w, dtype=work_np_dtype)
    pos_gather = network.add_gather(pos_const, pos_ids, axis=0)
    hidden = network.add_elementwise(
        hidden, pos_gather.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    mask_min = -1e4 if precision == "fp16" else -1e9
    causal_np = np.full(
        (max_seq_len, max_seq_len), mask_min, dtype=work_np_dtype)
    for i in range(max_seq_len):
        causal_np[i, : i + 1] = 0.0
    causal_const = graph_ops.add_constant(
        network, (1, 1, max_seq_len, max_seq_len), causal_np[None, None],
        dtype=work_np_dtype)
    mask_float = network.add_cast(attention_mask, work_trt_dtype).get_output(0)
    valid_ones = graph_ops.add_constant(
        network, (max_seq_len,), np.ones((max_seq_len,), dtype=work_np_dtype),
        dtype=work_np_dtype)
    invalid_mask = network.add_elementwise(
        valid_ones, mask_float, trt.ElementWiseOperation.SUB).get_output(0)
    invalid_mask_4d = network.add_shuffle(invalid_mask)
    invalid_mask_4d.reshape_dims = (1, 1, 1, max_seq_len)
    mask_penalty = graph_ops.add_constant(
        network, (1, 1, 1, 1), np.array([mask_min], dtype=work_np_dtype),
        dtype=work_np_dtype)
    padding_bias = network.add_elementwise(
        invalid_mask_4d.get_output(0), mask_penalty, trt.ElementWiseOperation.PROD)
    attention_bias = network.add_elementwise(
        causal_const, padding_bias.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(num_layers):
        prefix = f"text_model.encoder.layers.{layer_idx}"

        normed = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.layer_norm1.weight"],
            weights[f"{prefix}.layer_norm1.bias"],
            eps,
        )

        q = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size,
            weights[f"{prefix}.self_attn.q_proj.weight"])
        q = graph_ops.add_bias_sum(
            network, q, hidden_size, weights[f"{prefix}.self_attn.q_proj.bias"])
        k = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size,
            weights[f"{prefix}.self_attn.k_proj.weight"])
        k = graph_ops.add_bias_sum(
            network, k, hidden_size, weights[f"{prefix}.self_attn.k_proj.bias"])
        v = graph_ops.add_matmul_rhs_constant(
            network, normed, hidden_size, hidden_size,
            weights[f"{prefix}.self_attn.v_proj.weight"])
        v = graph_ops.add_bias_sum(
            network, v, hidden_size, weights[f"{prefix}.self_attn.v_proj.bias"])

        context = graph_ops.add_attention_from_rows(
            network,
            q,
            k,
            v,
            num_heads=num_heads,
            head_dim=head_dim,
            q_seq=max_seq_len,
            kv_seq=max_seq_len,
            mask=attention_bias,
        )

        attn_out = graph_ops.add_matmul_rhs_constant(
            network, context, hidden_size, hidden_size,
            weights[f"{prefix}.self_attn.out_proj.weight"])
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden_size,
            weights[f"{prefix}.self_attn.out_proj.bias"])
        hidden = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)

        normed2 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.layer_norm2.weight"],
            weights[f"{prefix}.layer_norm2.bias"],
            eps,
        )
        fc1 = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"{prefix}.mlp.fc1.weight"])
        fc1 = graph_ops.add_bias_sum(
            network, fc1, intermediate_size, weights[f"{prefix}.mlp.fc1.bias"])
        if hidden_act == "quick_gelu":
            coeff = graph_ops.add_constant(
                network, (1, 1), np.array([1.702], dtype=work_np_dtype),
                dtype=work_np_dtype)
            scaled = network.add_elementwise(fc1, coeff, trt.ElementWiseOperation.PROD)
            sigmoid = network.add_activation(scaled.get_output(0), trt.ActivationType.SIGMOID)
            activated = network.add_elementwise(
                fc1, sigmoid.get_output(0), trt.ElementWiseOperation.PROD).get_output(0)
        else:
            activated = graph_ops.add_gelu_new(network, fc1)

        fc2 = graph_ops.add_matmul_rhs_constant(
            network, activated, intermediate_size, hidden_size,
            weights[f"{prefix}.mlp.fc2.weight"])
        fc2 = graph_ops.add_bias_sum(
            network, fc2, hidden_size, weights[f"{prefix}.mlp.fc2.bias"])
        hidden = network.add_elementwise(
            hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    hidden = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights["text_model.final_layer_norm.weight"],
        weights["text_model.final_layer_norm.bias"],
        eps,
    )

    projection = graph_ops.add_matmul_rhs_constant(
        network,
        hidden,
        hidden_size,
        projected_size,
        weights["text_projection.weight"],
    )
    if "text_projection.bias" in weights:
        projection = graph_ops.add_bias_sum(
            network, projection, projected_size, weights["text_projection.bias"])

    projected = network.add_cast(projection, trt.float32).get_output(0)
    projected.name = "sam3_text_features"
    network.mark_output(projected)

    hidden_out = network.add_cast(hidden, trt.float32).get_output(0)
    hidden_out.name = "sam3_text_hidden_states"
    network.mark_output(hidden_out)

    if verbose:
        print(
            f"[sam3-text-builder] Building TRT engine "
            f"(hidden={hidden_size}, projected={projected_size}, "
            f"layers={num_layers}, seq={max_seq_len}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for SAM3 text encoder")
    return bytes(plan)
