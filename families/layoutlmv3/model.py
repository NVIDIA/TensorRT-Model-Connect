# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LayoutLMv3 text/layout embeddings and joint visual transformer features."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.task != "document_words_to_token_features" or request.backend != "trt":
        raise ValueError("LayoutLMv3 requires document_words_to_token_features with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.image_height not in (None, 224)
        or request.image_width not in (None, 224)
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "LayoutLMv3 supports one fp32 document at its 224-pixel visual profile"
        )
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        cfg["model_type"] != "layoutlmv3"
        or not cfg["visual_embed"]
        or not cfg.get("text_embed", True)
        or cfg.get("patch_size", 16) != 16
        or cfg["input_size"] != 224
        or cfg["hidden_act"] != "gelu"
        or not cfg["has_relative_attention_bias"]
        or not cfg["has_spatial_attention_bias"]
        or preprocessing["image_mean"] != [0.5, 0.5, 0.5]
        or preprocessing["image_std"] != [0.5, 0.5, 0.5]
        or not preprocessing["do_normalize"]
        or not preprocessing["do_resize"]
        or preprocessing["resample"] != 2
    ):
        raise ValueError("Unsupported LayoutLMv3 model or image preprocessing")
    length = request.max_sequence_length or 128
    if not 4 <= length <= cfg["max_position_embeddings"] - cfg["pad_token_id"] - 1:
        raise ValueError("LayoutLMv3 document word profile exceeds its position embeddings")
    from safetensors import safe_open
    from transformers import LayoutLMv3Tokenizer
    import torch

    tokenizer = LayoutLMv3Tokenizer.from_pretrained(directory)
    with safe_open(str(directory / "model.safetensors"), framework="np") as handle:
        weights = {
            name.removeprefix("layoutlmv3."): handle.get_tensor(name).astype(np.float32)
            for name in handle.keys()
            if name.startswith("layoutlmv3.")
        }
    if not weights:
        raise ValueError("LayoutLMv3 checkpoint lacks its document backbone")
    width, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    dim, visual = width // heads, 197
    total = length + visual

    def bucket(distances, bins, maximum):
        half = bins // 2
        exact = half // 2
        magnitude = distances.abs()
        large = (
            exact
            + (
                torch.log(magnitude.float() / exact) / np.log(maximum / exact) * (half - exact)
            ).long()
        )
        large = torch.minimum(large, torch.full_like(large, half - 1))
        return (
            ((distances > 0).long() * half + torch.where(magnitude < exact, magnitude, large))
            .numpy()
            .astype(np.int32)
        )

    positions = torch.cat([torch.arange(length), torch.arange(visual)])
    one_d = bucket(positions[None, :] - positions[:, None], cfg["rel_pos_bins"], cfg["max_rel_pos"])
    spatial_buckets = bucket(
        torch.arange(-1000, 1001), cfg["rel_2d_pos_bins"], cfg["max_rel_2d_pos"]
    )
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value, dtype=np.float32):
        value = np.ascontiguousarray(value, dtype=dtype)
        return net.add_constant(value.shape, value).get_output(0)

    def shuf(x, shape=None, perm=None):
        layer = net.add_shuffle(x)
        if perm is not None:
            layer.first_transpose = perm
        if shape is not None:
            layer.reshape_dims = shape
        return layer.get_output(0)

    def op(a, b, kind):
        return net.add_elementwise(a, b, kind).get_output(0)

    def add(a, b):
        return op(a, b, trt.ElementWiseOperation.SUM)

    def mul(a, b):
        return op(a, b, trt.ElementWiseOperation.PROD)

    def cat(values, axis):
        layer = net.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name, epsilon=None):
        return g.add_layer_norm(
            net,
            x,
            weights[name + ".weight"],
            weights[name + ".bias"],
            cfg["layer_norm_eps"] if epsilon is None else epsilon,
        )

    def gather(name, indices):
        return net.add_gather(const(weights[name + ".weight"]), indices, 0).get_output(0)

    def heads_view(x):
        return shuf(shuf(x, (1, total, heads, dim)), (1, heads, total, dim), (0, 2, 1, 3))

    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    positions = net.add_input("position_ids", trt.int32, (length,))
    boxes = net.add_input("layout_indices", trt.int32, (length, 6))
    pixels = net.add_input("pixel_values", trt.float32, (1, 3, 224, 224))
    rx = net.add_input("relative_x", trt.int32, (total, total))
    ry = net.add_input("relative_y", trt.int32, (total, total))
    prefix = "embeddings"
    text = add(
        gather(prefix + ".word_embeddings", ids),
        const(weights[prefix + ".token_type_embeddings.weight"][:1]),
    )
    text = add(text, gather(prefix + ".position_embeddings", positions))
    layout = []
    for index, name in enumerate(["x", "y", "x", "y", "h", "w"]):
        indices = shuf(
            net.add_slice(boxes, (0, index), (length, 1), (1, 1)).get_output(0), (length,)
        )
        layout.append(gather(prefix + "." + name + "_position_embeddings", indices))
    text = norm(add(text, cat(layout, 1)), "embeddings.LayerNorm")
    weight = weights["patch_embed.proj.weight"]
    convolution = net.add_convolution_nd(
        pixels, width, (16, 16), weight, weights["patch_embed.proj.bias"]
    )
    convolution.stride_nd = (16, 16)
    image = shuf(shuf(convolution.get_output(0), (width, 196)), (196, width), (1, 0))
    image = cat([const(weights["cls_token"].reshape(1, width)), image], 0)
    image = norm(add(image, const(weights["pos_embed"].reshape(visual, width))), "norm", 1e-6)
    hidden = norm(cat([text, image], 0), "LayerNorm")
    base_bias = weights["encoder.rel_pos_bias.weight"].T[one_d].transpose(2, 0, 1)[None]
    rel = []
    for name, indices in [("x", rx), ("y", ry)]:
        table = const(weights["encoder.rel_pos_" + name + "_bias.weight"].T)
        value = net.add_gather(table, indices, 0).get_output(0)
        rel.append(shuf(value, (1, heads, total, total), (2, 0, 1)))
    bias = mul(add(const(base_bias), add(*rel)), const(np.full((1, 1, 1, 1), dim**-0.5)))
    full_valid = cat([valid, const(np.ones(visual), np.int32)], 0)
    blocked = op(full_valid, const(np.zeros(1), np.int32), trt.ElementWiseOperation.EQUAL)
    mask = net.add_select(
        shuf(blocked, (1, 1, 1, total)),
        const(np.full((1, 1, 1, total), np.finfo(np.float32).min)),
        const(np.zeros((1, 1, 1, total))),
    ).get_output(0)
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"encoder.layer.{index}"
        query, key, value = [
            heads_view(linear(hidden, prefix + ".attention.self." + name))
            for name in ["query", "key", "value"]
        ]
        query = mul(query, const(np.full((1, 1, 1, 1), dim**-0.5)))
        scores = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(add(scores, bias), mask)
        scaled = op(scores, const(np.full((1, 1, 1, 1), 32)), trt.ElementWiseOperation.DIV)
        maximum = net.add_reduce(scaled, trt.ReduceOperation.MAX, 8, True).get_output(0)
        stabilized = mul(
            op(scaled, maximum, trt.ElementWiseOperation.SUB), const(np.full((1, 1, 1, 1), 32))
        )
        softmax = net.add_softmax(stabilized)
        softmax.axes = 8
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        context = shuf(context, (total, width), (0, 2, 1, 3))
        hidden = norm(
            add(hidden, linear(context, prefix + ".attention.output.dense")),
            prefix + ".attention.output.LayerNorm",
        )
        inner = linear(hidden, prefix + ".intermediate.dense")
        inner = net.add_activation(inner, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = norm(
            add(hidden, linear(inner, prefix + ".output.dense")), prefix + ".output.LayerNorm"
        )
    features = net.add_slice(hidden, (0, 0), (length, width), (1, 1)).get_output(0)
    features.name = "features"
    net.mark_output(features)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("LayoutLMv3 document encoder build failed")
    writer.set_header(family="layoutlmv3", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_bytes("tokenizer.json", tokenizer._tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=width,
            image_size=224,
            vocab_size=cfg["vocab_size"],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            cls_box=tokenizer.cls_token_box,
            sep_box=tokenizer.sep_token_box,
            pad_box=tokenizer.pad_token_box,
            spatial_buckets=spatial_buckets.tolist(),
        ),
    )
