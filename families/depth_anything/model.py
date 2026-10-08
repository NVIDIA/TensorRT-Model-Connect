# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Depth Anything monocular-depth engine.

A DINOv2 ViT backbone feeds a DPT neck and a small convolution head. Four
details are taken from the reference rather than inferred, because each one is
silently wrong-but-plausible if guessed:

* DINOv2 applies **layer scale** (``layer_scale1/2.lambda1``) to the attention and
  MLP branch outputs before each residual add. A ViT without it still builds and
  still produces a smooth depth map.
* The reassemble stage drops the class token, then resizes per tap by
  ``reassemble_factors``: >1 is a transposed convolution, 1 is identity, and <1 is
  a stride-``1/factor`` convolution.
* The fusion stage walks the taps **deepest first**, and each layer upsamples to
  the *next* tap's spatial size, falling back to a factor of two on the last one.
* The fusion and head upsamples use ``align_corners=True`` while the residual
  interpolation inside a fusion layer uses ``align_corners=False``. These map to
  different TensorRT coordinate transformations.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module
from . import graph as graph_ops

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_tensors(model_dir: Path) -> dict:
    paths = sorted(Path(model_dir).glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    tensors: dict = {}
    for path in paths:
        with safe_open(str(path), framework="numpy") as reader:
            for key in reader.keys():
                tensors[key] = reader.get_tensor(key)
    return tensors


def _count_encoder_layers(tensors: dict) -> int:
    """How many encoder layers the checkpoint actually carries."""
    indices = set()
    for key in tensors:
        if key.startswith("backbone.encoder.layer."):
            part = key.split(".")[3]
            if part.isdigit():
                indices.add(int(part))
    if not indices:
        raise ValueError("Depth Anything checkpoint has no backbone encoder layers")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("Depth Anything encoder layer indices are not contiguous")
    return len(indices)


def _require(tensors: dict, name: str) -> np.ndarray:
    if name not in tensors:
        raise KeyError(f"Depth Anything checkpoint is missing {name}")
    return np.asarray(tensors[name], dtype=np.float32)


def _backbone(network, image, tensors, cfg, work_np):
    """DINOv2 ViT. Returns the hidden states at the configured tap layers."""
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    grid = cfg["patch_grid"]
    patch = cfg["patch_size"]
    eps = cfg["layer_norm_eps"]
    tokens = 1 + grid * grid

    conv = network.add_convolution_nd(
        image, hidden, (patch, patch),
        trt.Weights(np.ascontiguousarray(
            _require(tensors, "backbone.embeddings.patch_embeddings.projection.weight"),
            dtype=work_np)),
        trt.Weights(np.ascontiguousarray(
            _require(tensors, "backbone.embeddings.patch_embeddings.projection.bias"),
            dtype=work_np)),
    )
    conv.stride_nd = (patch, patch)
    flat = network.add_shuffle(conv.get_output(0))
    flat.reshape_dims = (1, hidden, grid * grid)
    flat.second_transpose = (0, 2, 1)

    cls_token = graph_ops.add_constant(
        network, (1, 1, hidden),
        _require(tensors, "backbone.embeddings.cls_token").reshape(1, 1, hidden), dtype=work_np)
    sequence = network.add_concatenation([cls_token, flat.get_output(0)])
    sequence.axis = 1
    states = sequence.get_output(0)

    position = _require(tensors, "backbone.embeddings.position_embeddings").reshape(1, -1, hidden)
    if position.shape[1] != tokens:
        raise ValueError(
            f"Depth Anything position embeddings cover {position.shape[1]} tokens but the "
            f"configured image size needs {tokens}; this family builds the checkpoint's "
            "native resolution only")
    states = graph_ops.add_sum(
        network, states,
        graph_ops.add_constant(network, (1, tokens, hidden), position, dtype=work_np))

    scale = 1.0 / math.sqrt(head_dim)
    taps: dict[int, object] = {}
    wanted = set(cfg["out_layer_indices"])
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"backbone.encoder.layer.{index}"
        residual = states
        normed = graph_ops.add_layer_norm(
            network, states, _require(tensors, f"{prefix}.norm1.weight"),
            _require(tensors, f"{prefix}.norm1.bias"), eps, dtype=work_np)
        attention = f"{prefix}.attention.attention"
        query = graph_ops.split_heads(
            network, graph_ops.add_linear(
                network, normed, _require(tensors, f"{attention}.query.weight"),
                _require(tensors, f"{attention}.query.bias"), dtype=work_np),
            tokens, heads, head_dim)
        key = graph_ops.split_heads(
            network, graph_ops.add_linear(
                network, normed, _require(tensors, f"{attention}.key.weight"),
                _require(tensors, f"{attention}.key.bias"), dtype=work_np),
            tokens, heads, head_dim)
        value = graph_ops.split_heads(
            network, graph_ops.add_linear(
                network, normed, _require(tensors, f"{attention}.value.weight"),
                _require(tensors, f"{attention}.value.bias"), dtype=work_np),
            tokens, heads, head_dim)
        context = graph_ops.merge_heads(
            network, graph_ops.add_attention(network, query, key, value, scale), tokens, hidden)
        projected = graph_ops.add_linear(
            network, context, _require(tensors, f"{prefix}.attention.output.dense.weight"),
            _require(tensors, f"{prefix}.attention.output.dense.bias"), dtype=work_np)
        projected = _layer_scale(network, projected, tensors, f"{prefix}.layer_scale1", work_np)
        states = graph_ops.add_sum(network, projected, residual)

        residual = states
        normed = graph_ops.add_layer_norm(
            network, states, _require(tensors, f"{prefix}.norm2.weight"),
            _require(tensors, f"{prefix}.norm2.bias"), eps, dtype=work_np)
        inner = graph_ops.add_gelu(
            network, graph_ops.add_linear(
                network, normed, _require(tensors, f"{prefix}.mlp.fc1.weight"),
                _require(tensors, f"{prefix}.mlp.fc1.bias"), dtype=work_np))
        outer = graph_ops.add_linear(
            network, inner, _require(tensors, f"{prefix}.mlp.fc2.weight"),
            _require(tensors, f"{prefix}.mlp.fc2.bias"), dtype=work_np)
        outer = _layer_scale(network, outer, tensors, f"{prefix}.layer_scale2", work_np)
        states = graph_ops.add_sum(network, outer, residual)

        if index in wanted:
            taps[index] = states

    missing = wanted - set(taps)
    if missing:
        raise ValueError(f"Depth Anything tap layers {sorted(missing)} are past the encoder")

    # Dinov2Backbone applies its final layernorm to every tapped state
    # (apply_layernorm defaults to True and the checkpoint does not override it).
    # Without this the neck still builds and still yields a plausible depth map,
    # just a wrong one.
    gamma = _require(tensors, "backbone.layernorm.weight")
    beta = _require(tensors, "backbone.layernorm.bias")
    return [
        graph_ops.add_layer_norm(network, taps[index], gamma, beta, eps, dtype=work_np)
        for index in cfg["out_layer_indices"]
    ]


def _layer_scale(network, branch, tensors: dict, prefix: str, work_np):
    """DINOv2 scales each residual branch per channel before adding it back."""
    lambda1 = _require(tensors, f"{prefix}.lambda1")
    width = int(lambda1.size)
    weights = graph_ops.add_constant(
        network, (1, 1, width), lambda1.reshape(1, 1, width), dtype=work_np)
    return network.add_elementwise(
        branch, weights, trt.ElementWiseOperation.PROD).get_output(0)


def _reassemble(network, taps, tensors, cfg, work_np):
    """Drop the class token, lay the patches back out as a map, project and resize."""
    grid = cfg["patch_grid"]
    hidden = cfg["hidden_size"]
    outputs = []
    for index, (state, factor) in enumerate(zip(taps, cfg["reassemble_factors"])):
        prefix = f"neck.reassemble_stage.layers.{index}"
        trimmed = network.add_slice(state, (0, 1, 0), (1, grid * grid, hidden), (1, 1, 1))
        feature = graph_ops.tokens_to_feature_map(
            network, trimmed.get_output(0), grid, grid, hidden)
        feature = graph_ops.add_conv2d(
            network, feature, _require(tensors, f"{prefix}.projection.weight"),
            _require(tensors, f"{prefix}.projection.bias"), dtype=work_np)
        if factor > 1.0:
            step = int(round(factor))
            feature = graph_ops.add_deconv2d(
                network, feature, _require(tensors, f"{prefix}.resize.weight"),
                _require(tensors, f"{prefix}.resize.bias"), (step, step), dtype=work_np)
        elif factor < 1.0:
            step = int(round(1.0 / factor))
            feature = graph_ops.add_conv2d(
                network, feature, _require(tensors, f"{prefix}.resize.weight"),
                _require(tensors, f"{prefix}.resize.bias"),
                stride=(step, step), padding=(1, 1), dtype=work_np)
        outputs.append(feature)
    return outputs


def _residual_unit(network, x, tensors, prefix, work_np):
    """Pre-activation residual unit: conv2(relu(conv1(relu(x)))) + x."""
    hidden = graph_ops.add_relu(network, x)
    hidden = graph_ops.add_conv2d(
        network, hidden, _require(tensors, f"{prefix}.convolution1.weight"),
        _require(tensors, f"{prefix}.convolution1.bias"), padding=(1, 1), dtype=work_np)
    hidden = graph_ops.add_relu(network, hidden)
    hidden = graph_ops.add_conv2d(
        network, hidden, _require(tensors, f"{prefix}.convolution2.weight"),
        _require(tensors, f"{prefix}.convolution2.bias"), padding=(1, 1), dtype=work_np)
    return graph_ops.add_sum(network, hidden, x)


def _neck_convs(network, features, tensors, work_np):
    """Each reassembled tap is mapped to the fusion width by a bias-free 3x3."""
    return [
        graph_ops.add_conv2d(
            network, feature, _require(tensors, f"neck.convs.{index}.weight"), None,
            padding=(1, 1), dtype=work_np)
        for index, feature in enumerate(features)
    ]


def _fusion(network, features, tensors, cfg, work_np, sizes):
    """Walk the taps deepest first, upsampling each fused map onto the next one."""
    ordered = list(reversed(features))
    ordered_sizes = list(reversed(sizes))
    fused = None
    for index, feature in enumerate(ordered):
        prefix = f"neck.fusion_stage.layers.{index}"
        if fused is None:
            hidden = feature
        else:
            # The residual is brought onto the running map's own grid, with
            # align_corners=False, before the unit is applied. Compare the actual
            # tensors: the running map has already been upsampled by the previous
            # layer, so deriving the target from the tap sizes gets it backwards.
            residual = feature
            fused_shape = tuple(int(v) for v in fused.shape)[-2:]
            residual_shape = tuple(int(v) for v in residual.shape)[-2:]
            if residual_shape != fused_shape:
                residual = graph_ops.add_resize_to(
                    network, residual, fused_shape, align_corners=False)
            residual = _residual_unit(
                network, residual, tensors, f"{prefix}.residual_layer1", work_np)
            hidden = graph_ops.add_sum(network, fused, residual)
        hidden = _residual_unit(
            network, hidden, tensors, f"{prefix}.residual_layer2", work_np)
        # Upsample onto the next shallower tap, or by two on the last layer.
        target = ordered_sizes[index + 1] if index + 1 < len(ordered_sizes) else (
            ordered_sizes[index][0] * 2, ordered_sizes[index][1] * 2)
        hidden = graph_ops.add_resize_to(network, hidden, target, align_corners=True)
        fused = graph_ops.add_conv2d(
            network, hidden, _require(tensors, f"{prefix}.projection.weight"),
            _require(tensors, f"{prefix}.projection.bias"), dtype=work_np)
    return fused


def _head(network, fused, tensors, cfg, work_np):
    hidden = graph_ops.add_conv2d(
        network, fused, _require(tensors, "head.conv1.weight"),
        _require(tensors, "head.conv1.bias"), padding=(1, 1), dtype=work_np)
    target = (cfg["patch_grid"] * cfg["patch_size"], cfg["patch_grid"] * cfg["patch_size"])
    hidden = graph_ops.add_resize_to(network, hidden, target, align_corners=True)
    hidden = graph_ops.add_conv2d(
        network, hidden, _require(tensors, "head.conv2.weight"),
        _require(tensors, "head.conv2.bias"), padding=(1, 1), dtype=work_np)
    hidden = graph_ops.add_relu(network, hidden)
    hidden = graph_ops.add_conv2d(
        network, hidden, _require(tensors, "head.conv3.weight"),
        _require(tensors, "head.conv3.bias"), dtype=work_np)
    # Relative depth uses a ReLU tail; max_depth is 1.0 unless the config says otherwise.
    hidden = graph_ops.add_relu(network, hidden)
    if cfg["max_depth"] != 1.0:
        scale = graph_ops.add_constant(
            network, (1, 1, 1, 1), np.array([cfg["max_depth"]], dtype=np.float32), dtype=work_np)
        hidden = network.add_elementwise(
            hidden, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return hidden


def _build_network(network, tensors: dict, cfg: dict, work_np, work_trt):
    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, cfg["image_size"], cfg["image_size"]))
    image = pixel_values
    if image.dtype != work_trt:
        image = network.add_cast(image, work_trt).get_output(0)

    taps = _backbone(network, image, tensors, cfg, work_np)
    features = _reassemble(network, taps, tensors, cfg, work_np)

    # Spatial size of each tap after its reassemble factor, needed because the
    # fusion stage upsamples onto the next tap rather than by a fixed ratio.
    grid = cfg["patch_grid"]
    sizes = []
    for factor in cfg["reassemble_factors"]:
        if factor >= 1.0:
            side = int(round(grid * factor))
        else:
            side = int(math.ceil(grid / round(1.0 / factor)))
        sizes.append((side, side))

    features = _neck_convs(network, features, tensors, work_np)
    fused = _fusion(network, features, tensors, cfg, work_np, sizes)
    depth = _head(network, fused, tensors, cfg, work_np)

    if depth.dtype != trt.float32:
        depth = network.add_cast(depth, trt.float32).get_output(0)
    depth.name = "predicted_depth"
    network.mark_output(depth)


def build_engine(tensors: dict, cfg: dict, *, precision: str, verbose: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if precision == "fp16":
        work_np, work_trt = np.float16, trt.float16
    else:
        work_np, work_trt = np.float32, trt.float32
    _build_network(network, tensors, cfg, work_np, work_trt)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("Depth Anything TensorRT engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one Depth Anything monocular-depth bundle."""
    if request.task != "monocular_depth":
        raise ValueError("depth_anything supports only task=monocular_depth")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("depth_anything supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("depth_anything does not support dynamic_kv_cache")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("depth_anything supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("depth_anything does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("depth_anything does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("depth_anything does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("depth_anything does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("depth_anything does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("depth_anything does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("depth_anything does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    model_config = config_module.ModelConfig.from_dir(model_dir)
    if model_config.model_type.lower() != "depth_anything":
        raise ValueError(
            f"depth_anything does not support model_type={model_config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported depth_anything precision: {precision}")

    cfg = config_module.resolve(model_config.raw)
    for label, requested in (("image_height", request.image_height),
                             ("image_width", request.image_width)):
        if requested is not None and int(requested) != cfg["image_size"]:
            raise NotImplementedError(
                f"depth_anything builds the checkpoint's native {label} only; its "
                "position embeddings are not interpolated")

    tensors = _load_tensors(model_dir)
    # The checkpoint states the layer count outright; the config may not.
    counted = _count_encoder_layers(tensors)
    if counted != cfg["num_hidden_layers"]:
        cfg["num_hidden_layers"] = counted
    if max(cfg["out_layer_indices"]) >= counted:
        raise ValueError(
            f"Depth Anything out_indices name a layer past the checkpoint's {counted}")
    plan = build_engine(tensors, cfg, precision=precision, verbose=bool(request.verbose))

    writer.set_header(family="depth_anything", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": cfg["image_size"],
            "input_image_w": cfg["image_size"],
            "image_mean": list(_IMAGENET_MEAN),
            "image_std": list(_IMAGENET_STD),
        },
    )
