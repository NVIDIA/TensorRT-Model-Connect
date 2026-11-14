# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP text encoder with projection (``CLIPTextModelWithProjection``) as one engine.

SD3 reads two things from each CLIP encoder: the hidden states of the
penultimate layer, taken before the final layer norm, and the projected pooled
vector, taken after the final layer norm at the end-of-text position. The
pipeline passes no attention mask, so attention is causal only and padding
positions are attended like any other token.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph_ops import Graph, build_engine
from .weights import take


def _activation(graph: Graph, tensor, name: str):
    if name == "quick_gelu":
        return graph.quick_gelu(tensor)
    if name == "gelu":
        return graph.gelu_erf(tensor)
    raise NotImplementedError(f"stable_diffusion_3 CLIP activation {name!r}")


def encode(graph: Graph, input_ids, eos_index, weights: dict, cfg: dict):
    """Return (penultimate hidden states [1, L, W], projected pooled vector [1, P])."""
    width = cfg["width"]
    heads = cfg["heads"]
    head_dim = width // heads
    length = cfg["length"]
    eps = cfg["eps"]
    prefix = "text_model."

    table = graph.constant(take(weights, prefix + "embeddings.token_embedding.weight"))
    hidden = graph.fp32(graph.network.add_gather(table, input_ids, 0).get_output(0))
    positions = np.asarray(take(weights, prefix + "embeddings.position_embedding.weight"))
    hidden = graph.add(hidden, graph.constant(positions[:length].reshape(1, length, width),
                                              dtype=trt.float32))

    penultimate = None
    for index in range(cfg["layers"]):
        layer = f"{prefix}encoder.layers.{index}."
        normed = graph.layer_norm(hidden, eps, take(weights, layer + "layer_norm1.weight"),
                                  take(weights, layer + "layer_norm1.bias"))
        projections = []
        for name in ("q_proj", "k_proj", "v_proj"):
            projected = graph.linear(normed, take(weights, f"{layer}self_attn.{name}.weight"),
                                     take(weights, f"{layer}self_attn.{name}.bias"))
            projections.append(graph.split_heads(projected, heads, head_dim))
        context = graph.attention(*projections, scale=head_dim ** -0.5, causal=True)
        context = graph.linear(graph.merge_heads(context, heads, head_dim),
                               take(weights, layer + "self_attn.out_proj.weight"),
                               take(weights, layer + "self_attn.out_proj.bias"))
        hidden = graph.add(hidden, context)

        normed = graph.layer_norm(hidden, eps, take(weights, layer + "layer_norm2.weight"),
                                  take(weights, layer + "layer_norm2.bias"))
        inner = graph.linear(normed, take(weights, layer + "mlp.fc1.weight"),
                             take(weights, layer + "mlp.fc1.bias"))
        inner = _activation(graph, inner, cfg["activation"])
        hidden = graph.add(hidden, graph.linear(inner, take(weights, layer + "mlp.fc2.weight"),
                                                take(weights, layer + "mlp.fc2.bias")))
        if index == cfg["layers"] - 2:
            # transformers' hidden_states[-2]: the output of the second-to-last layer.
            penultimate = hidden
    if penultimate is None:
        raise NotImplementedError("stable_diffusion_3 needs CLIP encoders with two or more layers")

    final = graph.layer_norm(hidden, eps, take(weights, prefix + "final_layer_norm.weight"),
                             take(weights, prefix + "final_layer_norm.bias"))
    pooled = graph.network.add_gather(final, eos_index, 1).get_output(0)
    pooled = graph.reshape(pooled, (1, width))
    projected = graph.linear(pooled, take(weights, "text_projection.weight"))
    return penultimate, projected


def build_clip_engine(weights: dict, cfg: dict, *, precision: str, verbose: bool):
    def populate(graph: Graph) -> None:
        input_ids = graph.network.add_input("input_ids", trt.int32, (1, cfg["length"]))
        eos_index = graph.network.add_input("eos_index", trt.int32, (1,))
        hidden, pooled = encode(graph, input_ids, eos_index, weights, cfg)
        graph.output(hidden, "hidden_states")
        graph.output(pooled, "text_embeds")

    return build_engine(populate, precision=precision, verbose=verbose)
