# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V-JEPA2 video patch embedding, axial rotary attention and encoder features."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("vjepa2 does not support dynamic_kv_cache")
    if request.task != "video_to_token_features" or request.backend != "trt":
        raise ValueError("V-JEPA2 requires video_to_token_features with backend=trt")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    preprocessing = json.loads((directory / "video_preprocessor_config.json").read_text())
    frames, size = cfg["frames_per_clip"], cfg["crop_size"]
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.max_sequence_length is not None
        or request.image_height not in (None, size)
        or request.image_width not in (None, size)
        or request.video_num_frames not in (None, frames)
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "V-JEPA2 supports one fp32 clip at its checkpoint's frame count and crop size"
        )
    if (
        cfg["model_type"] != "vjepa2"
        or cfg["hidden_act"] != "gelu"
        or cfg["use_SiLU"]
        or not preprocessing["do_resize"]
        or not preprocessing["do_center_crop"]
        or not preprocessing["do_rescale"]
        or not preprocessing["do_normalize"]
        or preprocessing["resample"] != 2
        or preprocessing["image_mean"] != [0.485, 0.456, 0.406]
        or preprocessing["image_std"] != [0.229, 0.224, 0.225]
        or preprocessing["rescale_factor"] != 1 / 255
        or preprocessing["crop_size"] != dict(height=size, width=size)
        or preprocessing["size"]["shortest_edge"] < size
    ):
        raise ValueError("Unsupported V-JEPA2 encoder or video preprocessing")
    patch, tubelet, width, heads = [
        cfg[key] for key in ("patch_size", "tubelet_size", "hidden_size", "num_attention_heads")
    ]
    if frames % tubelet or size % patch or width % heads:
        raise ValueError("V-JEPA2 profile must contain complete spatiotemporal patches")
    grid = size // patch
    temporal = frames // tubelet
    length = temporal * grid * grid
    dim = width // heads
    rotary_dim = 2 * ((dim // 3) // 2)
    if not rotary_dim:
        raise ValueError("V-JEPA2 requires nonempty rotary dimensions for all three axes")
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {
            key: handle.get_tensor(key).float().numpy()
            for key in handle.keys()
            if key.startswith("encoder.")
        }
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def add(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(x, name):
        return g.add_linear(net, x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name):
        return g.add_layer_norm(
            net, x, weights[name + ".weight"], weights[name + ".bias"], cfg["layer_norm_eps"]
        )

    def split_heads(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, length, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    from .native_kernels import add_fp32_kernel

    def rope(x):
        return add_fp32_kernel(
            net,
            "rotary",
            [x],
            x.shape,
            [{"type": "int", "value": grid}],
        )

    pixels = net.add_input("pixel_values", trt.float32, (1, 3, frames, size, size))
    prefix = "encoder.embeddings.patch_embeddings.proj"
    convolution = net.add_convolution_nd(
        pixels,
        width,
        (tubelet, patch, patch),
        weights[prefix + ".weight"],
        weights[prefix + ".bias"],
    )
    convolution.stride_nd = (tubelet, patch, patch)
    flat = net.add_shuffle(convolution.get_output(0))
    flat.reshape_dims = (width, length)
    flat.second_transpose = (1, 0)
    hidden = flat.get_output(0)
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"encoder.layer.{index}"
        normalized = norm(hidden, prefix + ".norm1")
        query, key, value = [
            split_heads(linear(normalized, prefix + ".attention." + name))
            for name in ("query", "key", "value")
        ]
        scores = net.add_matrix_multiply(
            rope(query), trt.MatrixOperation.NONE, rope(key), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = mul(scores, const(np.full((1, 1, 1, 1), dim**-0.5)))
        probabilities = add_fp32_kernel(net, "softmax", [scores], scores.shape)
        context = net.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        merged = net.add_shuffle(context)
        merged.first_transpose = (0, 2, 1, 3)
        merged.reshape_dims = (length, width)
        hidden = add(hidden, linear(merged.get_output(0), prefix + ".attention.proj"))
        inner = linear(norm(hidden, prefix + ".norm2"), prefix + ".mlp.fc1")
        inner = add_fp32_kernel(net, "gelu", [inner], inner.shape)
        hidden = add(hidden, linear(inner, prefix + ".mlp.fc2"))
    hidden = norm(hidden, "encoder.layernorm")
    hidden.name = "features"
    net.mark_output(hidden)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("V-JEPA2 encoder build failed")
    writer.set_header(family="vjepa2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json(
        "runtime.json",
        dict(
            frames=frames,
            size=size,
            shortest_edge=preprocessing["size"]["shortest_edge"],
            patch=patch,
            tubelet=tubelet,
            grid=grid,
            grid_frames=temporal,
            hidden_size=width,
        ),
    )
