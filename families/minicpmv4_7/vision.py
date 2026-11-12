# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.7 vision engine: one image crop (thumbnail or slice) to LLM-width visual tokens.

Inputs, all prepared by the runtime from the crop's (h, w) patch grid:
  ``pixel_patches`` [L, 3 * p * p]  normalized pixels, one row per patch in (channel, y, x) order
  ``position_ids``  [L]             nearest-bucket indices into the learned position grid
  ``window_index``  [L]             2x2-window order of the full grid for the in-tower merger
  ``merge_index``   [L / 4]         2x2-block order of the merged grid for the projector
Output ``image_features`` [L / 16, text hidden] in the network work dtype.
"""

from __future__ import annotations

import tensorrt as trt

from .checkpoint import Checkpoint
from .config import ModelConfig, VisionConfig
from .graph import Graph


def load_vision(checkpoint: Checkpoint, config: ModelConfig) -> dict:
    vision = config.vision
    hidden, inner = vision.hidden_size, vision.intermediate_size
    window_hidden, window_inner = 4 * hidden, 4 * inner
    text_hidden = config.text.hidden_size
    get = checkpoint.get

    def attention(prefix):
        return {name: (get(f"{prefix}.{name}.weight", (hidden, hidden)), get(f"{prefix}.{name}.bias", (hidden,)))
                for name in ("q_proj", "k_proj", "v_proj", "out_proj")}

    def norm(prefix, width):
        return get(f"{prefix}.weight", (width,)), get(f"{prefix}.bias", (width,))

    def dense(prefix, out_width, in_width):
        return get(f"{prefix}.weight", (out_width, in_width)), get(f"{prefix}.bias", (out_width,))

    tower = "model.vision_tower"
    layers = []
    for i in range(vision.num_hidden_layers):
        prefix = f"{tower}.encoder.layers.{i}"
        layers.append({
            "norm1": norm(f"{prefix}.layer_norm1", hidden), "attention": attention(f"{prefix}.self_attn"),
            "norm2": norm(f"{prefix}.layer_norm2", hidden), "fc1": dense(f"{prefix}.mlp.fc1", inner, hidden),
            "fc2": dense(f"{prefix}.mlp.fc2", hidden, inner),
        })
    merger = f"{tower}.vit_merger"
    projector = "model.merger.mlp.0"
    patch = get(f"{tower}.embeddings.patch_embedding.weight",
                (hidden, vision.num_channels, vision.patch_size, vision.patch_size))
    return {
        "patch_weight": patch.reshape(hidden, vision.patch_width),
        "patch_bias": get(f"{tower}.embeddings.patch_embedding.bias", (hidden,)),
        "positions": get(f"{tower}.embeddings.position_embedding.weight", (vision.patches_per_side ** 2, hidden)),
        "layers": layers,
        "window": {
            "norm1": norm(f"{merger}.layer_norm1", hidden), "attention": attention(f"{merger}.self_attn"),
            "pre_norm": norm(f"{merger}.pre_norm", window_hidden),
            "linear_1": dense(f"{merger}.linear_1", window_inner, window_hidden),
            "linear_2": dense(f"{merger}.linear_2", hidden, window_inner),
        },
        "post_norm": norm(f"{tower}.post_layernorm", hidden),
        "projector": {
            "pre_norm": norm(f"{projector}.pre_norm", window_hidden),
            "linear_1": dense(f"{projector}.linear_1", window_hidden, window_hidden),
            "linear_2": dense(f"{projector}.linear_2", text_hidden, window_hidden),
        },
    }


def _attention(g: Graph, x: trt.ITensor, weights: dict, vision: VisionConfig, window: int | None) -> trt.ITensor:
    """Self-attention over ``x`` [N, C]: the whole crop, or independent consecutive windows of ``window``."""
    heads, head = vision.num_attention_heads, vision.head_dim
    groups = (1, -1) if window is None else (-1, window)

    def project(name):
        y = g.linear(x, *weights[name])  # [N, C]
        return g.reshape(y, (*groups, heads, head), transpose=(0, 2, 1, 3))  # [G, H, S, D]

    context = g.attention(project("q_proj"), project("k_proj"), project("v_proj"), head ** -0.5)
    context = g.reshape(g.transpose(context, (0, 2, 1, 3)), (-1, vision.hidden_size))
    return g.linear(g.cast(context, x.dtype), *weights["out_proj"])


def _encoder_layer(g: Graph, x: trt.ITensor, weights: dict, vision: VisionConfig) -> trt.ITensor:
    eps = vision.layer_norm_eps
    h = g.add(x, _attention(g, g.layer_norm(x, *weights["norm1"], eps), weights["attention"], vision, None))
    y = g.linear(g.layer_norm(h, *weights["norm2"], eps), *weights["fc1"])
    return g.add(h, g.linear(g.gelu_tanh(y), *weights["fc2"]))


def _window_merger(g: Graph, x: trt.ITensor, window_index: trt.ITensor, weights: dict,
                   vision: VisionConfig) -> trt.ITensor:
    """Attention inside each 2x2 window, then merge the window's four tokens into one."""
    eps, hidden = vision.layer_norm_eps, vision.hidden_size
    ordered_residual = g.gather(x, window_index, 0)
    ordered = g.gather(g.layer_norm(x, *weights["norm1"], eps), window_index, 0)
    attended = g.add(ordered_residual, _attention(g, ordered, weights["attention"], vision, 4))
    windows = g.reshape(attended, (-1, 4, hidden))
    patch_residual = g.cast(g.reduce(g.f32(windows), trt.ReduceOperation.AVG, 1, keep=False), x.dtype)
    flat = g.reshape(windows, (-1, 4 * hidden))
    y = g.linear(g.layer_norm(flat, *weights["pre_norm"], eps), *weights["linear_1"])
    y = g.linear(g.gelu_tanh(y), *weights["linear_2"])
    return g.add(y, patch_residual)


def build_vision_network(network: trt.INetworkDefinition, weights: dict, config: ModelConfig,
                         work: trt.DataType) -> Graph:
    vision = config.vision
    g = Graph(network, work)
    hidden = vision.hidden_size
    patches = network.add_input("pixel_patches", trt.float32, (-1, vision.patch_width))
    positions = network.add_input("position_ids", trt.int32, (-1,))
    window_index = network.add_input("window_index", trt.int32, (-1,))
    merge_index = network.add_input("merge_index", trt.int32, (-1,))

    x = g.linear(g.cast(patches, work), weights["patch_weight"], weights["patch_bias"])
    x = g.add(x, g.gather(g.const(weights["positions"]), positions, 0))
    for index, layer in enumerate(weights["layers"]):
        x = _encoder_layer(g, x, layer, vision)
        if index == vision.insert_layer_id:
            x = _window_merger(g, x, window_index, weights["window"], vision)
    x = g.layer_norm(x, *weights["post_norm"], vision.layer_norm_eps)

    projector = weights["projector"]
    blocks = g.reshape(g.gather(x, merge_index, 0), (-1, 4 * hidden))
    y = g.layer_norm(blocks, *projector["pre_norm"], 1e-6)
    y = g.linear(g.gelu_erf(g.linear(y, *projector["linear_1"])), *projector["linear_2"])
    y.name = "image_features"
    network.mark_output(y)
    return g


def vision_profile(builder: trt.Builder, vision: VisionConfig, max_patches: int) -> trt.IOptimizationProfile:
    """Crop sizes from the smallest 4x4 patch grid up to ``max_patches`` patches."""
    if max_patches < 16 or max_patches % 16:
        raise ValueError("MiniCPM-V 4.7 vision capacity must be a positive multiple of 16 patches")
    profile = builder.create_optimization_profile()
    opt = min(1024, max_patches)
    profile.set_shape("pixel_patches", (16, vision.patch_width), (opt, vision.patch_width),
                      (max_patches, vision.patch_width))
    for name, divisor in (("position_ids", 1), ("window_index", 1), ("merge_index", 4)):
        profile.set_shape(name, (16 // divisor,), (opt // divisor,), (max_patches // divisor,))
    return profile
