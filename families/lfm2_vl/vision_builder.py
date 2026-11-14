# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned SigLIP2 NaFlex encoder and pixel-unshuffle projector."""

from __future__ import annotations

import numpy as np
import tensorrt as trt
from .checkpoint_mapper import _load_tensor, _open_safetensors


def build_vision_engine(model_dir, raw, precision, *, max_patches=4096):
    from .model import _cast, _constant, _precision_dtypes, _work_constant

    readers = _open_safetensors(model_dir)
    vision = raw["vision_config"]
    hidden = int(vision["hidden_size"])
    heads = int(vision["num_attention_heads"])
    head_dim = hidden // heads
    storage, dtype = _precision_dtypes(precision)
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    cfg = builder.create_builder_config()
    cfg.builder_optimization_level = 1
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    prefix = "model.vision_tower.vision_model."

    def weight(name):
        return _load_tensor(readers, name)

    def constant(value, work=dtype):
        value = np.asarray(value, dtype=np.float32)
        # FP32 arithmetic scalars (epsilon, attention scale, mask sentinel) must
        # not pass through FP16 storage before entering an FP32 operation.
        stored = np.dtype(np.float32) if work == trt.float32 else storage
        return _work_constant(net, value.shape, value, storage_dtype=stored, work_dtype=work)

    def norm_weight(name):
        return _cast(net, constant(weight(name)[None, :]), trt.float32)

    def linear(x, name):
        w = weight(name + ".weight").T.copy()
        out = net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, constant(w), trt.MatrixOperation.NONE
        ).get_output(0)
        b = weight(name + ".bias")[None, :]
        return net.add_elementwise(out, constant(b), trt.ElementWiseOperation.SUM).get_output(0)

    def norm(x, name):
        x = _cast(net, x, trt.float32)
        mean = net.add_reduce(x, trt.ReduceOperation.AVG, 2, True).get_output(0)
        centered = net.add_elementwise(x, mean, trt.ElementWiseOperation.SUB).get_output(0)
        square = net.add_elementwise(centered, centered, trt.ElementWiseOperation.PROD).get_output(
            0
        )
        variance = net.add_reduce(square, trt.ReduceOperation.AVG, 2, True).get_output(0)
        eps = constant(np.full((1, 1), vision["layer_norm_eps"]), trt.float32)
        variance = net.add_elementwise(variance, eps, trt.ElementWiseOperation.SUM).get_output(0)
        inv = net.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        out = net.add_elementwise(centered, inv, trt.ElementWiseOperation.DIV).get_output(0)
        out = net.add_elementwise(
            out,
            norm_weight(name + ".weight"),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        out = net.add_elementwise(
            out,
            norm_weight(name + ".bias"),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        return _cast(net, out, dtype)

    pixels = net.add_input("pixel_values", trt.float32, (-1, 768))
    positions = net.add_input("position_embeddings", trt.float32, (-1, hidden))
    pixel_mask = net.add_input("pixel_attention_mask", trt.int32, (-1,))
    pixel_mask.set_dimension_name(0, "patches")
    zero = _constant(net, (1,), np.zeros(1, dtype=np.int32), np_dtype=np.dtype(np.int32))
    valid = net.add_elementwise(pixel_mask, zero, trt.ElementWiseOperation.GREATER).get_output(0)
    valid_keys = net.add_shuffle(valid)
    valid_keys.reshape_dims = (1, 1, -1)
    unshuffle = net.add_input("unshuffle_indices", trt.int32, (-1, 4))
    pixels.set_dimension_name(0, "patches")
    positions.set_dimension_name(0, "patches")
    profile = builder.create_optimization_profile()
    profile.set_shape("pixel_values", (4, 768), (1024, 768), (max_patches, 768))
    profile.set_shape("position_embeddings", (4, hidden), (1024, hidden), (max_patches, hidden))
    profile.set_shape("pixel_attention_mask", (4,), (1024,), (max_patches,))
    profile.set_shape("unshuffle_indices", (1, 4), (256, 4), (max_patches // 4, 4))
    cfg.add_optimization_profile(profile)
    x = linear(_cast(net, pixels, dtype), prefix + "embeddings.patch_embedding")
    x = net.add_elementwise(
        x, _cast(net, positions, dtype), trt.ElementWiseOperation.SUM
    ).get_output(0)
    for index in range(int(vision["num_hidden_layers"])):
        p = prefix + f"encoder.layers.{index}."
        inp = norm(x, p + "layer_norm1")
        qkv = []
        for kind in ("q", "k", "v"):
            v = linear(inp, p + f"self_attn.{kind}_proj")
            shaped = net.add_shuffle(v)
            shaped.reshape_dims = (-1, heads, head_dim)
            shaped.second_transpose = (1, 0, 2)
            qkv.append(_cast(net, shaped.get_output(0), trt.float32))
        scores = net.add_matrix_multiply(
            qkv[0], trt.MatrixOperation.NONE, qkv[1], trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = net.add_elementwise(
            scores,
            constant(np.full((1, 1, 1), head_dim**-0.5), trt.float32),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        scores = net.add_select(
            valid_keys.get_output(0),
            scores,
            constant(np.full((1, 1, 1), np.finfo(np.float32).min), trt.float32),
        ).get_output(0)
        softmax = net.add_softmax(scores)
        softmax.axes = 4
        attention = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, qkv[2], trt.MatrixOperation.NONE
        ).get_output(0)
        shape = net.add_shuffle(_cast(net, attention, dtype))
        shape.first_transpose = (1, 0, 2)
        shape.reshape_dims = (-1, hidden)
        out = linear(shape.get_output(0), p + "self_attn.out_proj")
        x = net.add_elementwise(x, out, trt.ElementWiseOperation.SUM).get_output(0)
        out = norm(x, p + "layer_norm2")
        out = linear(out, p + "mlp.fc1")
        # Torch's GELU computes the nonlinearity in FP32 and rounds once.
        out = net.add_activation(
            _cast(net, out, trt.float32), trt.ActivationType.GELU_TANH
        ).get_output(0)
        out = linear(_cast(net, out, dtype), p + "mlp.fc2")
        x = net.add_elementwise(x, out, trt.ElementWiseOperation.SUM).get_output(0)
    x = norm(x, prefix + "post_layernorm")
    gathered = net.add_gather(x, unshuffle, 0).get_output(0)
    shaped = net.add_shuffle(gathered)
    shaped.reshape_dims = (-1, hidden * 4)
    p = "model.multi_modal_projector."
    x = linear(shaped.get_output(0), p + "linear_1")
    x = net.add_activation(_cast(net, x, trt.float32), trt.ActivationType.GELU_ERF).get_output(0)
    x = linear(_cast(net, x, dtype), p + "linear_2")
    x = _cast(net, x, trt.float32)
    x.name = "image_embeddings"
    net.mark_output(x)
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("TensorRT failed to build LFM2-VL vision encoder")
    positions = weight(prefix + "embeddings.position_embedding.weight")
    return bytes(plan), positions
