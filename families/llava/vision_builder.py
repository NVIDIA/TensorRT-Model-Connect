# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP penultimate patch states followed by LLaVA's two-layer projector."""

import numpy as np
import tensorrt as trt
from . import vision_graph as g


def build_vision(checkpoint, config, *, verbose=False):
    cfg = config.raw["vision_config"]
    width, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    size, patch = cfg["image_size"], cfg["patch_size"]
    count = (size // patch) ** 2
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 1
    settings.max_aux_streams = 0
    inp = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    inp = network.add_cast(inp, trt.float16).get_output(0)

    def weight(name):
        return checkpoint.tensor(f"vision_tower.vision_model.{name}")

    patches = g.add_conv2d(network, inp, weight("embeddings.patch_embedding.weight"),
                           stride=(patch, patch), dtype=np.float16)
    rows = network.add_shuffle(patches)
    rows.reshape_dims = (width, count)
    rows.second_transpose = (1, 0)
    cls = g.add_constant(network, (1, width), weight("embeddings.class_embedding"), dtype=np.float16)
    hidden = network.add_concatenation([cls, rows.get_output(0)])
    hidden.axis = 0
    hidden = g.add_sum(network, hidden.get_output(0), g.add_constant(
        network, (count + 1, width), weight("embeddings.position_embedding.weight"), dtype=np.float16))
    eps = cfg.get("layer_norm_eps", 1e-5)
    hidden = g.add_layer_norm(network, hidden, weight("pre_layrnorm.weight"), weight("pre_layrnorm.bias"), eps)
    # hidden_states[-2] is the output of layer N-2, before the final CLIP layer.
    for index in range(cfg["num_hidden_layers"] - 1):
        prefix = f"encoder.layers.{index}"

        def linear(name, tensor):
            return g.add_linear(network, tensor, weight(f"{prefix}.{name}.weight"),
                                weight(f"{prefix}.{name}.bias"), dtype=np.float16)

        norm = g.add_layer_norm(network, hidden, weight(f"{prefix}.layer_norm1.weight"),
                                weight(f"{prefix}.layer_norm1.bias"), eps)
        qkv = [linear(f"self_attn.{name}_proj", norm) for name in ("q", "k", "v")]
        attention = g.add_attention(network, *qkv, num_heads=heads, head_dim=width // heads)
        hidden = g.add_sum(network, hidden, linear("self_attn.out_proj", attention))
        norm = g.add_layer_norm(network, hidden, weight(f"{prefix}.layer_norm2.weight"),
                                weight(f"{prefix}.layer_norm2.bias"), eps)
        mlp = g.add_quick_gelu(network, linear("mlp.fc1", norm), dtype=np.float16)
        hidden = g.add_sum(network, hidden, linear("mlp.fc2", mlp))
    hidden = network.add_slice(hidden, (1, 0), (count, width), (1, 1)).get_output(0)
    for index in (1, 2):
        prefix = f"multi_modal_projector.linear_{index}"
        hidden = g.add_linear(network, hidden, checkpoint.tensor(f"{prefix}.weight"),
                              checkpoint.tensor(f"{prefix}.bias"), dtype=np.float16)
        if index == 1:
            hidden = network.add_activation(hidden, trt.ActivationType.GELU_ERF).get_output(0)
    hidden = network.add_cast(hidden, trt.float32).get_output(0)
    hidden.name = "image_features"
    network.mark_output(hidden)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("LLaVA vision engine build failed")
    return bytes(plan)
