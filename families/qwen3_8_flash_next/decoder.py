# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose prepared Flash-Next decoder stages without an Edge or ONNX dependency."""

import ml_dtypes
import numpy as np
import tensorrt as trt

from .delta_attention import delta_attention_sublayer
from .moe import route_experts
from .ple import dequantize_packed_ngram_rows, lexical_injection
from .residual import collapse_residual
from .sparse_attention import sparse_attention_sublayer


def embed_tokens(network, token_ids, media_mask, media_vectors, checkpoint, vocab):
    """Select actual patch vectors at expanded media positions; preserve IDs for PLE."""
    w = checkpoint.bf16("model.language_model.embed_tokens.weight", (vocab, 2560))
    table = network.add_constant(w.shape, w).get_output(0)
    embedded = network.add_gather(table, token_ids, 0).get_output(0)
    zero = network.add_constant((1, 1), np.zeros((1, 1), np.int32)).get_output(0)
    mask = network.add_elementwise(media_mask, zero, trt.ElementWiseOperation.GREATER).get_output(0)
    selected = network.add_select(mask, media_vectors, embedded).get_output(0)
    streams = network.add_concatenation([selected] * 4)
    streams.axis = 1
    return streams.get_output(0)


def token_logits(network, hyper, checkpoint, vocab, eps):
    """The model exits directly through the trained mixer, with no extra RMSNorm."""
    last = collapse_residual(network, hyper, checkpoint, eps)
    w = checkpoint.bf16("lm_head.weight", (vocab, 2560))
    table = network.add_constant(w.shape, w).get_output(0)
    x = network.add_cast(last, trt.float32).get_output(0)
    table = network.add_cast(table, trt.float32).get_output(0)
    y = network.add_matrix_multiply(
        x, trt.MatrixOperation.NONE, table, trt.MatrixOperation.TRANSPOSE
    )
    return network.add_cast(y.get_output(0), trt.float16).get_output(0)


def decoder_layer(network, hyper, state, checkpoint, config, layer, max_chunk):
    """Return the residual skip, sparse FFN inputs and the tentative attention state.

    The caller runs the selected experts, combines them, and commits cache
    outputs only after successfully processing the request. All exposed
    state is per-request; rotary positions are distinct from KV slots.
    """
    eps = config["rms_norm_eps"]
    updated = {}
    if layer + 1 in config["ple_layer_ids"]:
        prefix = f"model.language_model.layers.{layer}.ple."
        scale = checkpoint.tensor(
            prefix + "ple_embedding.ngram_embedding.weight_scale", ml_dtypes.bfloat16, (1,)
        ).item()
        rows = dequantize_packed_ngram_rows(network, state["lexical_words"], scale)

        class Weights:
            def __getitem__(self, name):
                key = prefix + name
                return checkpoint.bf16(key, checkpoint.array(key).shape, fp32=True)

        lexical, updated["new_lexical_history"] = lexical_injection(
            network, rows, hyper, state["lexical_history"], Weights(), eps
        )
        hyper = network.add_elementwise(hyper, lexical, trt.ElementWiseOperation.SUM).get_output(0)

    kind = config["layer_types"][layer]
    if kind == "linear_attention":
        hyper, (updated["new_history"], updated["new_state"]) = delta_attention_sublayer(
            network, hyper, state["history"], state["state"], checkpoint, layer, eps, max_chunk
        )
    elif kind == "full_attention":
        past = state["key"], state["value"], state["index_key"]
        hyper, (cache, _) = sparse_attention_sublayer(
            network, hyper, state["position"], state["rotary"], past, checkpoint, layer, eps
        )
        updated["new_key"], updated["new_value"], updated["new_index_key"] = cache
    else:
        raise ValueError(f"Unsupported Flash-Next attention type: {kind}")

    h, ids, weights, shared, inject = route_experts(network, hyper, checkpoint, layer, eps)
    return {
        "skip": hyper,
        "tokens": h,
        "selected": ids,
        "weights": weights,
        "shared": shared,
        "inject": inject,
        **updated,
    }
