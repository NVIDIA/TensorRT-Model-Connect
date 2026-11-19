# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build DINOv2 with registers as one family-owned TensorRT network."""

from pathlib import Path

import tensorrt as trt

from . import config, encoder_builder, graph as g, position


def build_engine(weights, cfg, *, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width, grid, size = cfg["hidden_size"], cfg["grid"], cfg["image_size"]
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    patches = g.add_conv2d(
        network,
        pixels,
        weights["embeddings.patch_embeddings.projection.weight"],
        weights["embeddings.patch_embeddings.projection.bias"],
        stride=(cfg["patch_size"], cfg["patch_size"]),
    )
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)
    cls = g.add_constant(network, (1, width), weights["embeddings.cls_token"])
    concat = network.add_concatenation([cls, flat.get_output(0)])
    concat.axis = 0
    table = position.interpolate_position_table(
        weights["embeddings.position_embeddings"], cfg["trained_grid"], grid
    )
    hidden = g.add_sum(network, concat.get_output(0), g.add_constant(network, table.shape, table))

    # Positions are added before registers, which have no positional embedding.
    cls = network.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    patches = network.add_slice(hidden, (1, 0), (grid * grid, width), (1, 1)).get_output(0)
    registers = g.add_constant(
        network, (cfg["num_register_tokens"], width), weights["embeddings.register_tokens"]
    )
    concat = network.add_concatenation([cls, registers, patches])
    concat.axis = 0
    hidden = encoder_builder.build_layers(network, concat.get_output(0), weights, cfg)
    hidden = g.add_layer_norm(
        network, hidden, weights["layernorm.weight"], weights["layernorm.bias"], cfg["eps"]
    )
    hidden.name = "last_hidden_state"
    network.mark_output(hidden)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("dinov2_with_registers engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "image_to_token_and_pooled_features":
        raise ValueError("dinov2_with_registers requires image_to_token_and_pooled_features")
    if request.backend != "trt" or request.precision != "fp32":
        raise NotImplementedError("dinov2_with_registers supports backend=trt, precision=fp32")
    if request.dynamic_kv_cache:
        raise NotImplementedError("dinov2_with_registers does not support dynamic_kv_cache")
    if (
        request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.dynamic_kv_cache
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.video_num_frames is not None
        or request.max_sequence_length is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("dinov2_with_registers requires a single-image unquantized build")
    from safetensors.numpy import load_file

    directory = Path(request.model_dir)
    weights = load_file(str(directory / "model.safetensors"))
    cfg = config.resolve(directory, weights)
    if any(
        value is not None and value != cfg["image_size"]
        for value in (request.image_height, request.image_width)
    ):
        raise NotImplementedError("image size must match the checkpoint preprocessor")
    writer.set_header(family="dinov2_with_registers", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, cfg, verbose=request.verbose))
    runtime = {
        key: cfg[key]
        for key in (
            "shortest_edge",
            "image_mean",
            "image_std",
            "resample",
            "patch_size",
            "grid",
            "hidden_size",
            "num_register_tokens",
        )
    }
    runtime.update(input_image_h=cfg["image_size"], input_image_w=cfg["image_size"])
    writer.add_json("runtime.json", runtime)
