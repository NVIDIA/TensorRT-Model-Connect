# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef's full Qwen vision encoder as a native TensorRT graph."""

import tensorrt as trt

from .graph import Graph


def vision_block(g, x, p, config, cos, sin, trace=None):
    width, heads = config["hidden_size"], config["num_heads"]
    dim = width // heads

    def record(name, value):
        if trace is not None:
            trace[name] = value
        return value

    def rope(x):
        f = g.cast(x, trt.float32)
        rotated = g.concat(
            [g.mul(g.slice(f, 2, dim // 2, dim // 2), -1), g.slice(f, 2, 0, dim // 2)]
        )
        return g.cast(g.add(g.mul(f, cos), g.mul(rotated, sin)), trt.bfloat16)

    n = record("norm1", g.norm(x, p + ".norm1", 1e-6))
    qkv = record("qkv", g.reshape(g.linear(n, p + ".attn.qkv"), (-1, 3, width)))
    q, k, v = [g.reshape(g.slice(qkv, 1, j, 1), (-1, heads, dim)) for j in range(3)]
    q, k = record("q_rotated", rope(q)), record("k_rotated", rope(k))
    # Vision head_dim=72 needs FP32 score scaling before softmax.
    attended = record(
        "attention",
        g.attention(
            g.reshape(q, (-1, width)),
            g.reshape(k, (-1, width)),
            g.reshape(v, (-1, width)),
            heads,
            fp32_accumulation=True,
        ),
    )
    projected = record("attn_proj", g.linear(attended, p + ".attn.proj"))
    x = record("post_attn", g.add(x, projected))
    n = record("norm2", g.norm(x, p + ".norm2", 1e-6))
    n = record("fc1", g.linear(n, p + ".mlp.linear_fc1"))
    # PyTorch evaluates GELU in FP32, then rounds once to the input dtype.
    # A native BF16 activation rounded intermediate terms and differed on
    # 813272 captured values; the explicit FP32 boundary matched all values.
    n = record(
        "activation",
        g.cast(g.activation(g.cast(n, trt.float32), trt.ActivationType.GELU_TANH), trt.bfloat16),
    )
    projected = record("fc2", g.linear(n, p + ".mlp.linear_fc2"))
    return record("output", g.add(x, projected))


def vision_graph(network, weights, config, debug=False):
    g = Graph(network, weights)
    # LayerNorm reduces and applies its affine parameters in FP32 in the
    # original vision encoder. A BF16 native normalization graph rounded the
    # affine boundary early; explicit FP32 reduced a captured mismatch from
    # 96233 values to 9 rounding-tie values out of 294912.
    g.precise_norm = True
    width = config["hidden_size"]
    heads = config["num_heads"]
    dim = width // heads
    patch_width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
    patches = network.add_input("patches", trt.float32, (-1, patch_width))
    positions = network.add_input("positions", trt.float32, (-1, width))
    cos = network.add_input("rope_cos", trt.float32, (-1, 1, dim))
    sin = network.add_input("rope_sin", trt.float32, (-1, 1, dim))
    g.attention_groups = network.add_input("frame_ids", trt.int32, (-1,))
    prefix = "model.visual"
    kernel = (config["temporal_patch_size"], config["patch_size"], config["patch_size"])
    patch_input = g.reshape(
        g.cast(g.cast(patches, trt.bfloat16), trt.float32),
        (-1, config["in_channels"], *kernel),
    )
    # Keep the original 3D convolution's accumulation order and its BF16
    # rounding before bias. A flattened Linear had sparse rounding differences.
    projection = network.add_convolution_nd(
        patch_input, width, kernel, trt.Weights(), trt.Weights()
    )
    projection.stride_nd = kernel
    projection.set_input(1, g.const(weights[prefix + ".patch_embed.proj.weight"]))
    projected = g.reshape(projection.get_output(0), (-1, width))
    rounded = g.cast(g.cast(projected, trt.bfloat16), trt.float32)
    bias = g.const(weights[prefix + ".patch_embed.proj.bias"].reshape(1, width))
    x = g.cast(g.add(rounded, bias), trt.bfloat16)
    trace = {"patch_embedding": x}
    x = g.add(x, g.cast(positions, trt.bfloat16))
    trace["positioned"] = x

    for i in range(config["depth"]):
        p = prefix + f".blocks.{i}"
        x = vision_block(g, x, p, config, cos, sin, trace if i == 0 and debug else None)
        trace[f"block_{i}"] = x
    p = prefix + ".merger"
    x = g.reshape(g.norm(x, p + ".norm", 1e-6), (-1, width * config["spatial_merge_size"] ** 2))
    x = g.linear(x, p + ".linear_fc1")
    x = g.cast(g.gelu(g.cast(x, trt.float32)), trt.bfloat16)
    x = g.linear(x, p + ".linear_fc2")
    x.name = "visual_embeddings"
    network.mark_output(x)
    if debug:
        for name, tensor in trace.items():
            tensor = g.cast(tensor, trt.float32)
            tensor.name = name
            network.mark_output(tensor)
    return g


def build_vision(weights, config, max_patches, debug=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    _graph = vision_graph(
        network, weights, config, debug
    )  # Owns constant storage until build ends.
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        tail = tuple(tensor.shape)[1:]
        profile.set_shape(
            tensor.name, (4, *tail), (min(256, max_patches), *tail), (max_patches, *tail)
        )
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.add_optimization_profile(profile)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Clef vision encoder build failed")
    return bytes(plan)
