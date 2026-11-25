# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one DPT-Hybrid monocular-depth engine.

A BiT ResNet stem and three bottleneck stages feed a ViT-B/16 encoder through a
1x1 projection; the DPT neck reassembles BiT stages 1-2 and two ViT taps into
four maps and fuses them; a small convolution head predicts relative inverse
depth at the input resolution. Details that are silently wrong-but-plausible
when guessed are taken from the reference:

* Every BiT convolution standardizes its weight per output channel (mean 0,
  biased variance 1, epsilon 1e-8) before convolving. The weights are
  constants, so the standardization is applied here, in float64, once.
* BiT pads the 7x7 stride-2 stem convolution, the stem max pool and every
  stride-2 3x3 convolution TF-style ("SAME"): asymmetric, extra row and column
  at the bottom and right. The max pool pads with zeros, not -inf.
* The neck takes the ViT layers named by ``backbone_out_indices[2:]`` before the
  encoder's final LayerNorm, concatenates each token with the class token, and
  projects the pair back to the hidden width ("project" readout).
* Fusion upsamples with ``align_corners=True``; a residual that has to be
  brought onto the running map uses ``align_corners=False``.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module
from . import graph as graph_ops
from .checkpoint import load_tensors

_BIT = config_module.BIT_PREFIX


class _Weights:
    """Checkpoint tensors by name; every lookup is recorded so leftovers can be refused."""

    def __init__(self, tensors: dict):
        self._tensors = tensors
        self.used: set[str] = set()

    def get(self, name: str, shape=None) -> np.ndarray:
        if name not in self._tensors:
            raise KeyError(f"DPT checkpoint is missing {name}")
        value = np.asarray(self._tensors[name], dtype=np.float32)
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"DPT tensor {name} has shape {value.shape}, expected {tuple(shape)}")
        self.used.add(name)
        return value

    def has(self, name: str) -> bool:
        return name in self._tensors

    def unused(self) -> list[str]:
        return sorted(set(self._tensors) - self.used)


def standardize(weight: np.ndarray, eps: float) -> np.ndarray:
    """BiT weight standardization: per output channel, zero mean and unit biased variance."""
    flat = np.asarray(weight, dtype=np.float64).reshape(weight.shape[0], -1)
    mean = flat.mean(axis=1, keepdims=True)
    variance = np.square(flat - mean).mean(axis=1, keepdims=True)
    return ((flat - mean) / np.sqrt(variance + eps)).reshape(weight.shape).astype(np.float32)


def _ws_conv(network, x, weights: _Weights, name, shape, *, stride=1, padding=(0, 0), work_np,
             eps):
    weight = standardize(weights.get(f"{name}.weight", shape), eps)
    if weights.has(f"{name}.bias"):
        raise NotImplementedError(f"dpt expects the bias-free BiT convolution {name}")
    return graph_ops.add_conv2d(
        network, x, weight, None, stride=(stride, stride),
        pre_padding=(padding[0], padding[0]), post_padding=(padding[1], padding[1]),
        dtype=work_np)


def _group_norm(network, x, weights: _Weights, name, groups, work_np, *, relu: bool):
    channels = int(tuple(x.shape)[1])
    y = graph_ops.add_group_norm(
        network, x, weights.get(f"{name}.weight", (channels,)),
        weights.get(f"{name}.bias", (channels,)), groups, config_module.GROUP_NORM_EPS,
        dtype=work_np)
    return graph_ops.add_relu(network, y) if relu else y


def _bit(network, image, weights: _Weights, cfg, work_np):
    """BiT stem and bottleneck stages. Returns the output map of every stage."""
    backbone = cfg["backbone"]
    groups = backbone["num_groups"]
    eps = config_module.LAYER_WEIGHT_EPS

    x = _ws_conv(network, image, weights, f"{_BIT}.embedder.convolution",
                 (backbone["embedding_size"], 3, 7, 7), stride=2,
                 padding=backbone["stem_padding"], work_np=work_np,
                 eps=config_module.STEM_WEIGHT_EPS)
    x = _group_norm(network, x, weights, f"{_BIT}.embedder.norm", groups, work_np, relu=True)
    pool_before, pool_after = backbone["pool_padding"]
    x = graph_ops.add_zero_pad(network, x, (pool_before, pool_before), (pool_after, pool_after))
    x = graph_ops.add_max_pool(network, x, (3, 3), (2, 2))

    outputs = []
    for stage_index, stage in enumerate(backbone["stages"]):
        for layer_index in range(stage["depth"]):
            prefix = f"{_BIT}.encoder.stages.{stage_index}.layers.{layer_index}"
            shapes = config_module.bit_layer_shapes(stage, layer_index)
            stride = stage["stride"] if layer_index == 0 else 1
            shortcut = x
            if layer_index == 0:
                shortcut = _ws_conv(network, x, weights, f"{prefix}.downsample.conv",
                                    shapes["downsample.conv"],
                                    stride=stride, padding=stage["downsample_padding"],
                                    work_np=work_np, eps=eps)
                shortcut = _group_norm(network, shortcut, weights, f"{prefix}.downsample.norm",
                                       groups, work_np, relu=False)
            y = _ws_conv(network, x, weights, f"{prefix}.conv1", shapes["conv1"],
                         work_np=work_np, eps=eps)
            y = _group_norm(network, y, weights, f"{prefix}.norm1", groups, work_np, relu=True)
            y = _ws_conv(network, y, weights, f"{prefix}.conv2", shapes["conv2"], stride=stride,
                         padding=stage["conv2_padding"] if layer_index == 0 else (1, 1),
                         work_np=work_np, eps=eps)
            y = _group_norm(network, y, weights, f"{prefix}.norm2", groups, work_np, relu=True)
            y = _ws_conv(network, y, weights, f"{prefix}.conv3", shapes["conv3"],
                         work_np=work_np, eps=eps)
            y = _group_norm(network, y, weights, f"{prefix}.norm3", groups, work_np, relu=False)
            x = graph_ops.add_relu(network, graph_ops.add_sum(network, y, shortcut))
        expected = (1, stage["out_channels"], stage["out_side"], stage["out_side"])
        if tuple(int(v) for v in x.shape) != expected:
            raise RuntimeError(f"BiT stage {stage_index} produced {tuple(x.shape)}, not {expected}")
        outputs.append(x)
    return outputs


def _vit(network, feature_map, weights: _Weights, cfg, work_np):
    """Hybrid patch embedding and the ViT encoder; returns the tapped hidden states."""
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    grid = cfg["patch_grid"]
    tokens = 1 + grid * grid
    eps = cfg["layer_norm_eps"]
    channels = int(tuple(feature_map.shape)[1])

    projected = graph_ops.add_conv2d(
        network, feature_map, weights.get("dpt.embeddings.projection.weight",
                                          (hidden, channels, 1, 1)),
        weights.get("dpt.embeddings.projection.bias", (hidden,)), dtype=work_np)
    patches = graph_ops.map_to_tokens(network, projected)
    cls_token = graph_ops.add_constant(
        network, (1, 1, hidden), weights.get("dpt.embeddings.cls_token", (1, 1, hidden)),
        dtype=work_np)
    sequence = network.add_concatenation([cls_token, patches])
    sequence.axis = 1
    position = weights.get("dpt.embeddings.position_embeddings", (1, tokens, hidden))
    states = graph_ops.add_sum(
        network, sequence.get_output(0),
        graph_ops.add_constant(network, (1, tokens, hidden), position, dtype=work_np))

    linear_bias = cfg["qkv_bias"]
    scale = 1.0 / math.sqrt(head_dim)
    taps = {}
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"dpt.encoder.layer.{index}"
        residual = states
        normed = graph_ops.add_layer_norm(
            network, states, weights.get(f"{prefix}.layernorm_before.weight", (hidden,)),
            weights.get(f"{prefix}.layernorm_before.bias", (hidden,)), eps, dtype=work_np)
        projections = []
        for part in ("query", "key", "value"):
            name = f"{prefix}.attention.attention.{part}"
            bias = weights.get(f"{name}.bias", (hidden,)) if linear_bias else None
            projections.append(graph_ops.split_heads(
                network,
                graph_ops.add_linear(network, normed, weights.get(f"{name}.weight",
                                                                   (hidden, hidden)),
                                     bias, dtype=work_np),
                tokens, heads, head_dim))
        context = graph_ops.merge_heads(
            network, graph_ops.add_attention(network, *projections, scale), tokens, hidden)
        attended = graph_ops.add_linear(
            network, context, weights.get(f"{prefix}.attention.output.dense.weight",
                                          (hidden, hidden)),
            weights.get(f"{prefix}.attention.output.dense.bias", (hidden,)), dtype=work_np)
        states = graph_ops.add_sum(network, attended, residual)

        residual = states
        normed = graph_ops.add_layer_norm(
            network, states, weights.get(f"{prefix}.layernorm_after.weight", (hidden,)),
            weights.get(f"{prefix}.layernorm_after.bias", (hidden,)), eps, dtype=work_np)
        inner = graph_ops.add_gelu(network, graph_ops.add_linear(
            network, normed, weights.get(f"{prefix}.intermediate.dense.weight",
                                         (cfg["intermediate_size"], hidden)),
            weights.get(f"{prefix}.intermediate.dense.bias", (cfg["intermediate_size"],)),
            dtype=work_np))
        outer = graph_ops.add_linear(
            network, inner, weights.get(f"{prefix}.output.dense.weight",
                                        (hidden, cfg["intermediate_size"])),
            weights.get(f"{prefix}.output.dense.bias", (hidden,)), dtype=work_np)
        states = graph_ops.add_sum(network, outer, residual)
        if index in cfg["tap_layers"]:
            taps[index] = states
        if index >= max(cfg["tap_layers"]):
            # Layers past the last tap (and the encoder's final LayerNorm) feed only
            # the pooled output, which depth estimation does not use.
            break
    return [taps[index] for index in cfg["tap_layers"]]


def _reassemble(network, state, zeros, weights: _Weights, cfg, stage: int, work_np):
    """Drop the class token, append it to every token, project, lay out and resize.

    ``zeros`` is a [1, patches, hidden] zero tensor; adding the class token to it
    broadcasts the token over every patch exactly.
    """
    hidden = cfg["hidden_size"]
    grid = cfg["patch_grid"]
    prefix = "neck.reassemble_stage"
    patches = network.add_slice(state, (0, 1, 0), (1, grid * grid, hidden), (1, 1, 1))
    cls_token = network.add_slice(state, (0, 0, 0), (1, 1, hidden), (1, 1, 1))
    readout = graph_ops.add_sum(network, zeros, cls_token.get_output(0))
    pair = network.add_concatenation([patches.get_output(0), readout])
    pair.axis = 2
    projected = graph_ops.add_gelu(network, graph_ops.add_linear(
        network, pair.get_output(0),
        weights.get(f"{prefix}.readout_projects.{stage}.0.weight", (hidden, 2 * hidden)),
        weights.get(f"{prefix}.readout_projects.{stage}.0.bias", (hidden,)), dtype=work_np))
    feature = graph_ops.tokens_to_map(network, projected, grid, grid)

    channels = cfg["neck_hidden_sizes"][stage]
    feature = graph_ops.add_conv2d(
        network, feature, weights.get(f"{prefix}.layers.{stage}.projection.weight",
                                      (channels, hidden, 1, 1)),
        weights.get(f"{prefix}.layers.{stage}.projection.bias", (channels,)), dtype=work_np)
    factor = cfg["reassemble_factors"][stage]
    if factor < 1.0:
        step = int(round(1.0 / factor))
        feature = graph_ops.add_conv2d(
            network, feature, weights.get(f"{prefix}.layers.{stage}.resize.weight",
                                          (channels, channels, 3, 3)),
            weights.get(f"{prefix}.layers.{stage}.resize.bias", (channels,)),
            stride=(step, step), pre_padding=(1, 1), post_padding=(1, 1), dtype=work_np)
    return feature


def _residual_unit(network, x, weights: _Weights, prefix, cfg, work_np):
    """Pre-activation residual unit: conv2(relu(conv1(relu(x)))) + x."""
    channels = cfg["fusion_hidden_size"]
    hidden = graph_ops.add_relu(network, x)
    for index, conv in enumerate(("convolution1", "convolution2")):
        bias = (weights.get(f"{prefix}.{conv}.bias", (channels,)) if cfg["fusion_bias"]
                else None)
        hidden = graph_ops.add_conv2d(
            network, hidden, weights.get(f"{prefix}.{conv}.weight", (channels, channels, 3, 3)),
            bias, pre_padding=(1, 1), post_padding=(1, 1), dtype=work_np)
        if index == 0:
            hidden = graph_ops.add_relu(network, hidden)
    return graph_ops.add_sum(network, hidden, x)


def _neck(network, features, weights: _Weights, cfg, work_np):
    fusion = cfg["fusion_hidden_size"]
    maps = [
        graph_ops.add_conv2d(
            network, feature, weights.get(f"neck.convs.{index}.weight",
                                          (fusion, cfg["neck_hidden_sizes"][index], 3, 3)),
            None, pre_padding=(1, 1), post_padding=(1, 1), dtype=work_np)
        for index, feature in enumerate(features)
    ]
    fused = None
    for index, feature in enumerate(reversed(maps)):
        prefix = f"neck.fusion_stage.layers.{index}"
        if fused is None:
            hidden = feature
        else:
            residual = feature
            target = tuple(int(v) for v in fused.shape)[-2:]
            if tuple(int(v) for v in residual.shape)[-2:] != target:
                residual = graph_ops.add_resize_to(network, residual, target, align_corners=False)
            residual = _residual_unit(network, residual, weights, f"{prefix}.residual_layer1",
                                      cfg, work_np)
            hidden = graph_ops.add_sum(network, fused, residual)
        hidden = _residual_unit(network, hidden, weights, f"{prefix}.residual_layer2", cfg,
                                work_np)
        side = tuple(int(v) for v in hidden.shape)[-2:]
        hidden = graph_ops.add_resize_to(network, hidden, (side[0] * 2, side[1] * 2),
                                         align_corners=True)
        fused = graph_ops.add_conv2d(
            network, hidden, weights.get(f"{prefix}.projection.weight", (fusion, fusion, 1, 1)),
            weights.get(f"{prefix}.projection.bias", (fusion,)), dtype=work_np)
    return fused


def _head(network, fused, weights: _Weights, cfg, work_np):
    fusion = cfg["fusion_hidden_size"]
    half = fusion // 2
    hidden = graph_ops.add_conv2d(
        network, fused, weights.get("head.head.0.weight", (half, fusion, 3, 3)),
        weights.get("head.head.0.bias", (half,)), pre_padding=(1, 1), post_padding=(1, 1),
        dtype=work_np)
    side = tuple(int(v) for v in hidden.shape)[-2:]
    hidden = graph_ops.add_resize_to(network, hidden, (side[0] * 2, side[1] * 2),
                                     align_corners=True)
    hidden = graph_ops.add_conv2d(
        network, hidden, weights.get("head.head.2.weight", (32, half, 3, 3)),
        weights.get("head.head.2.bias", (32,)), pre_padding=(1, 1), post_padding=(1, 1),
        dtype=work_np)
    hidden = graph_ops.add_relu(network, hidden)
    hidden = graph_ops.add_conv2d(
        network, hidden, weights.get("head.head.4.weight", (1, 32, 1, 1)),
        weights.get("head.head.4.bias", (1,)), dtype=work_np)
    return graph_ops.add_relu(network, hidden)


def _build_network(network, weights: _Weights, cfg: dict, work_np, work_trt) -> None:
    size = cfg["image_size"]
    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    image = graph_ops.retype(network, pixel_values, work_trt)

    stages = _bit(network, image, weights, cfg, work_np)
    taps = _vit(network, stages[-1], weights, cfg, work_np)
    patches, hidden = cfg["patch_grid"] ** 2, cfg["hidden_size"]
    zeros = graph_ops.add_constant(network, (1, patches, hidden), np.zeros((1, patches, hidden)),
                                   dtype=work_np)
    features = stages[:2] + [
        _reassemble(network, state, zeros, weights, cfg, stage, work_np)
        for stage, state in zip((2, 3), taps)
    ]
    depth = _head(network, _neck(network, features, weights, cfg, work_np), weights, cfg,
                  work_np)
    if tuple(int(v) for v in depth.shape) != (1, 1, size, size):
        raise RuntimeError(f"DPT head produced {tuple(depth.shape)}, not (1, 1, {size}, {size})")
    depth = graph_ops.retype(network, depth, trt.float32)
    depth.name = "predicted_depth"
    network.mark_output(depth)

    # Weights the depth path never reads: the encoder's final LayerNorm and the
    # layers after the last tap feed only the pooled output, the first fusion
    # layer has no residual, and the identity reassemble stages carry nothing.
    unused = [name for name in weights.unused() if not _ignorable(name, cfg)]
    if unused:
        raise ValueError("DPT checkpoint has tensors this family does not map: "
                         + ", ".join(unused[:8]))


def _ignorable(name: str, cfg: dict) -> bool:
    if name.startswith("dpt.layernorm.") or name.startswith("neck.fusion_stage.layers.0."
                                                            "residual_layer1."):
        return True
    if name.startswith("dpt.encoder.layer."):
        return int(name.split(".")[3]) > max(cfg["tap_layers"])
    return False


def build_engine(weights: _Weights, cfg: dict, *, precision: str, verbose: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if precision == "fp16":
        work_np, work_trt = np.float16, trt.float16
    else:
        work_np, work_trt = np.float32, trt.float32
        # An FP32 engine computes in FP32: no TF32 GEMMs or convolutions.
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    _build_network(network, weights, cfg, work_np, work_trt)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("DPT TensorRT engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one DPT-Hybrid monocular-depth bundle."""
    if request.task != "monocular_depth":
        raise ValueError("dpt supports only task=monocular_depth")
    if request.backend != "trt":
        raise ValueError("dpt supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("dpt does not support dynamic_kv_cache")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("dpt supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("dpt does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("dpt does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("dpt does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("dpt does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(f"dpt does not support quantization={request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("dpt does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("dpt does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Unsupported dpt precision: {precision}")

    model_dir = Path(request.model_dir)
    cfg = config_module.load(model_dir)
    config_module.check_requested_image_size("image_height", request.image_height,
                                             cfg["image_size"])
    config_module.check_requested_image_size("image_width", request.image_width,
                                             cfg["image_size"])
    config_module.check_engine_input(cfg["processor"], precision)

    tensors = load_tensors(model_dir)
    # Before any graph is built: a backbone configured wider or narrower than the
    # checkpoint's weights is refused, not built from the weights alone.
    config_module.check_weight_shapes(tensors, itertools.chain(
        config_module.bit_weight_shapes(cfg), config_module.vit_embedding_shapes(cfg)))
    weights = _Weights(tensors)
    plan = build_engine(weights, cfg, precision=precision, verbose=bool(request.verbose))

    processor = cfg["processor"]
    writer.set_header(family="dpt", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": cfg["image_size"],
            "input_image_w": cfg["image_size"],
            "rescale_factor": processor["rescale_factor"],
            "image_mean": processor["image_mean"],
            "image_std": processor["image_std"],
        },
    )
