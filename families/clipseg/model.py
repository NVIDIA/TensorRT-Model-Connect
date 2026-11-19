# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIPSeg refined decoder, with independent text conditioning and native masks."""

import json
from pathlib import Path
import numpy as np
import tensorrt as trt
from . import graph as g, encoder_builder, text_builder


def _tower(raw):
    return dict(
        hidden_size=raw["hidden_size"],
        num_heads=raw["num_attention_heads"],
        head_dim=raw["hidden_size"] // raw["num_attention_heads"],
        num_layers=raw["num_hidden_layers"],
        eps=raw["layer_norm_eps"],
        vocab_size=raw.get("vocab_size", 0),
    )


def _network(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    return builder, net, options, logger


def _plan(builder, net, options):
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("CLIPSeg engine build failed")
    return bytes(plan)


def build_text(weights, cfg, verbose=False):
    builder, net, options, _logger = _network(verbose)
    length = cfg["text_config"]["max_position_embeddings"]
    ids = net.add_input("input_ids", trt.int32, (length,))
    index = net.add_input("eos_index", trt.int32, (1,))
    features = text_builder.build_text_tower(
        net, ids, index, weights, _tower(cfg["text_config"]), weights["clip.text_projection.weight"]
    )
    features.name = "conditional_embeddings"
    net.mark_output(features)
    return _plan(builder, net, options)


def build_mask(weights, cfg, size, verbose=False):
    import torch

    builder, net, options, _logger = _network(verbose)
    vision = cfg["vision_config"]
    width = vision["hidden_size"]
    patch = vision["patch_size"]
    grid = size // patch
    raw = net.add_input("pixel_values", trt.float32, (1, 3, size, size))
    condition = net.add_input("conditional_embeddings", trt.float32, (1, cfg["projection_dim"]))
    prefix = "clip.vision_model"
    patches = g.add_conv2d(
        net, raw, weights[prefix + ".embeddings.patch_embedding.weight"], stride=(patch, patch)
    )
    flat = net.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)
    cls = g.add_constant(net, (1, width), weights[prefix + ".embeddings.class_embedding"])
    concat = net.add_concatenation([cls, flat.get_output(0)])
    concat.axis = 0
    table = weights[prefix + ".embeddings.position_embedding.weight"]
    trained = int(round((len(table) - 1) ** 0.5))
    positions = (
        torch.from_numpy(table[1:].copy()).reshape(1, trained, trained, width).permute(0, 3, 1, 2)
    )
    positions = torch.nn.functional.interpolate(
        positions, size=(grid, grid), mode="bicubic", align_corners=False
    )
    positions = positions.permute(0, 2, 3, 1).reshape(grid * grid, width).numpy()
    table = np.concatenate([table[:1], positions])
    hidden = g.add_sum(net, concat.get_output(0), g.add_constant(net, table.shape, table))
    hidden = g.add_layer_norm(
        net,
        hidden,
        weights[prefix + ".pre_layrnorm.weight"],
        weights[prefix + ".pre_layrnorm.bias"],
        vision["layer_norm_eps"],
    )
    captures = cfg["extract_layers"]
    tower = _tower(vision)
    tower["num_layers"] = max(captures) + 1
    features = encoder_builder.build_layers(
        net, hidden, weights, prefix, tower, capture=set(captures)
    )

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights[name + ".bias"])

    def norm(x, name):
        return g.add_layer_norm(
            net, x, weights[name + ".weight"], weights[name + ".bias"], vision["layer_norm_eps"]
        )

    output = None
    for index, layer_index in enumerate(reversed(captures)):
        reduced = linear(features[layer_index], f"decoder.reduces.{index}")
        output = reduced if output is None else g.add_sum(net, output, reduced)
        if index == cfg["conditional_layer"]:
            multiply = linear(condition, "decoder.film_mul")
            add = linear(condition, "decoder.film_add")
            output = g.add_sum(
                net,
                net.add_elementwise(output, multiply, trt.ElementWiseOperation.PROD).get_output(0),
                add,
            )
        layer = f"decoder.layers.{index}"
        q, k, v = [
            linear(output, layer + ".self_attn." + name) for name in ("q_proj", "k_proj", "v_proj")
        ]
        attention = g.add_attention(
            net,
            q,
            k,
            v,
            num_heads=cfg["decoder_num_attention_heads"],
            head_dim=cfg["reduce_dim"] // cfg["decoder_num_attention_heads"],
        )
        output = norm(
            g.add_sum(net, output, linear(attention, layer + ".self_attn.out_proj")),
            layer + ".layer_norm1",
        )
        inner = net.add_activation(
            linear(output, layer + ".mlp.fc1"), trt.ActivationType.RELU
        ).get_output(0)
        output = norm(
            g.add_sum(net, output, linear(inner, layer + ".mlp.fc2")), layer + ".layer_norm2"
        )
    patches = net.add_slice(output, (1, 0), (grid * grid, cfg["reduce_dim"]), (1, 1)).get_output(0)
    reshape = net.add_shuffle(patches)
    reshape.first_transpose = (1, 0)
    reshape.reshape_dims = (1, cfg["reduce_dim"], grid, grid)
    output = reshape.get_output(0)
    prefix = "decoder.transposed_convolution"
    conv = net.add_convolution_nd(
        output,
        cfg["reduce_dim"],
        (3, 3),
        trt.Weights(np.ascontiguousarray(weights[prefix + ".0.weight"])),
        trt.Weights(np.ascontiguousarray(weights[prefix + ".0.bias"])),
    )
    conv.padding_nd = (1, 1)
    output = net.add_activation(conv.get_output(0), trt.ActivationType.RELU).get_output(0)
    for index in (2, 4):
        weight = np.ascontiguousarray(weights[f"{prefix}.{index}.weight"])
        bias = np.ascontiguousarray(weights[f"{prefix}.{index}.bias"])
        kernel = tuple(weight.shape[-2:])
        deconv = net.add_deconvolution_nd(
            output, weight.shape[1], kernel, trt.Weights(weight), trt.Weights(bias)
        )
        deconv.stride_nd = kernel
        output = deconv.get_output(0)
        if index == 2:
            output = net.add_activation(output, trt.ActivationType.RELU).get_output(0)
    output = net.add_activation(output, trt.ActivationType.SIGMOID).get_output(0)
    output.name = "masks"
    net.mark_output(output)
    return _plan(builder, net, options)


def build(request, writer):
    if request.task != "text_prompted_segmentation" or request.backend != "trt":
        raise ValueError("CLIPSeg requires text_prompted_segmentation and backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("clipseg does not support dynamic_kv_cache")
    if (
        request.precision != "fp32"
        or request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.dynamic_kv_cache
        or request.fp32_layers
        or request.quantization not in (None, "none")
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
        or request.max_sequence_length is not None
    ):
        raise NotImplementedError("CLIPSeg supports single-example fp32 inference")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    prep = json.loads((directory / "preprocessor_config.json").read_text())
    size = prep["size"]["height"]
    if (
        cfg.get("model_type") != "clipseg"
        or not cfg.get("use_complex_transposed_convolution")
        or prep.get("resample") != 2
        or prep["size"]["width"] != size
    ):
        raise NotImplementedError(
            "CLIPSeg requires the refined decoder and square bilinear preprocessing"
        )
    if any(
        value is not None and value != size for value in (request.image_height, request.image_width)
    ):
        raise ValueError("CLIPSeg image size must match the checkpoint processor")
    from safetensors.numpy import load_file

    weights = load_file(str(directory / "model.safetensors"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.set_header(family="clipseg", task=request.task, backend=request.backend)
    writer.add_bytes("text.plan", build_text(weights, cfg, request.verbose))
    writer.add_bytes("mask.plan", build_mask(weights, cfg, size, request.verbose))
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        {
            "image_size": size,
            "max_text_len": cfg["text_config"]["max_position_embeddings"],
            "projection_dim": cfg["projection_dim"],
            "image_mean": prep["image_mean"],
            "image_std": prep["image_std"],
        },
    )
