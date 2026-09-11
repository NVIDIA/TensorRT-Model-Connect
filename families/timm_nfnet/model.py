# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm NFNet classifiers.

NFNet is normalizer-free: it has no batch norm anywhere. Two things stand in
for it. Every convolution standardises its own weights, which is a host-side
transform and is done once at build time. Every activation is scaled by a
constant that keeps its output variance near one, and each residual branch is
scaled by a per-block pair of scalars so the signal does not grow with depth.

The layout comes from the checkpoint: stage depths from the
`stages.<stage>.<block>` keys, the group count of the 3x3 pair from its own
weight shape, and the presence of the second 3x3 and of the squeeze gate from
whether those leaves exist. The residual scalars are not stored anywhere, so
they are recomputed from the schedule that produced them.

Scope is the seven `dm_nfnet_f*` checkpoints. The `nfnet_l*` and `eca_nfnet_l*`
variants use a different activation and a different attention block.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import Checkpoint


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


# timm builds every NFNet convolution with this epsilon, not the 1e-6 default
# of the standardised convolution itself.
_WEIGHT_STANDARDISATION_EPSILON = 1e-5
# The gain that keeps GELU's output variance at one. Every dm_nfnet uses GELU.
_GELU_GAMMA = 1.7015043497085571
# How much of the residual branch is added back, and the gate's own gain.
_RESIDUAL_ALPHA = 0.2
_ATTENTION_GAIN = 2.0
_BLOCK = re.compile(r"^stages\.(\d+)\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"NFNet model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("NFNet config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("dm_nfnet"):
        raise ValueError(
            f"unsupported timm NFNet model identity: {identity!r}; only the dm_nfnet "
            "checkpoints use GELU with a squeeze gate"
        )
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("pretrained_cfg")
    source = nested if isinstance(nested, dict) else raw
    input_size = source.get("input_size", [3, 192, 192])
    if isinstance(input_size, int):
        height = width = input_size
    elif (
        isinstance(input_size, list)
        and len(input_size) == 3
        and all(isinstance(value, int) and not isinstance(value, bool) for value in input_size)
        and input_size[0] == 3
    ):
        height, width = input_size[-2:]
    else:
        raise ValueError("NFNet pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("NFNet image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("NFNet image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.9)),
        "interpolation": str(source.get("interpolation", "bicubic")),
        # Every dm_nfnet asks for "squash": both axes are resized to the same
        # length before the crop, so the aspect ratio is deliberately not kept.
        # Reading this wrong changes the pixels the engine sees, not the engine.
        "crop_mode": str(source.get("crop_mode", "center")),
    }
    if (
        result["image_height"] <= 0
        or result["image_width"] <= 0
        or result["num_classes"] <= 0
        or not 0.0 < result["crop_pct"] <= 1.0
        or any(value == 0.0 for value in result["std"])
        or result["interpolation"] not in {"bilinear", "bicubic"}
        or result["crop_mode"] not in {"center", "squash"}
    ):
        raise ValueError("NFNet preprocessing or classifier config is invalid")
    return result


def same_padding(size: int, kernel: int, stride: int) -> tuple[int, int, int, int]:
    """TensorFlow's SAME padding as (top, left, bottom, right).

    When the stride does not divide the input evenly the extra pixel goes on
    the bottom and the right, so the padding is not symmetric and cannot be
    given to TensorRT as a single pair.
    """
    needed = max((-(-size // stride) - 1) * stride + kernel - size, 0)
    start = needed // 2
    return (start, start, needed - start, needed - start)


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(name[match.end() :].split(".", 1)[0])
    if not leaves:
        raise ValueError("NFNet checkpoint has no stages.<stage>.<block> tensors")
    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("NFNet stage indices are not contiguous from 0")

    blocks: list[dict[str, object]] = []
    # The residual scalars are a schedule, not stored weights: the branch gain
    # grows with depth and is reset at the head of every stage.
    expected_variance = 1.0
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"NFNet stage {stage} block indices are not contiguous from 0")
        for index in indices:
            present = leaves[(stage, index)]
            if not {"conv1", "conv2", "conv3"}.issubset(present):
                raise ValueError(f"NFNet stages.{stage}.{index} is missing a convolution")
            blocks.append(
                {
                    "prefix": f"stages.{stage}.{index}",
                    "stage": stage,
                    # Every stage after the first halves at its head.
                    "stride": 2 if (stage > 0 and index == 0) else 1,
                    "beta": float(expected_variance**-0.5),
                    "has_downsample": "downsample" in present,
                    "has_second_conv": "conv2b" in present,
                    "has_gate": "attn_last" in present,
                }
            )
            if index == 0:
                expected_variance = 1.0
            expected_variance += _RESIDUAL_ALPHA**2
    return blocks


def standardise(weight: np.ndarray, gain: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Standardise one convolution's weights, as the checkpoint expects.

    Each output filter is centred and scaled to unit variance, then multiplied
    by its learned gain and by one over the square root of its fan-in. The
    variance is the biased one, matching the batch norm call timm standardises
    with. The statistics stay float32 through the division so a narrow filter
    does not lose precision in fp16.
    """
    flat = weight.reshape(int(weight.shape[0]), -1).astype(np.float32)
    fan_in = flat.shape[1]
    mean = flat.mean(axis=1, keepdims=True)
    variance = flat.var(axis=1, keepdims=True)
    scale = gain.reshape(-1, 1).astype(np.float32) * float(fan_in) ** -0.5
    standardised = (flat - mean) / np.sqrt(variance + _WEIGHT_STANDARDISATION_EPSILON) * scale
    return standardised.reshape(weight.shape).astype(dtype)


def _standardised(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.weight": standardise(
            checkpoint.tensor(f"{prefix}.weight"), checkpoint.tensor(f"{prefix}.gain"), dtype
        ),
        f"{prefix}.bias": checkpoint.tensor(f"{prefix}.bias").astype(dtype),
    }


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    stem = sorted(name for name in checkpoint.names if re.fullmatch(r"stem\.conv\d+\.weight", name))
    if not stem:
        raise ValueError("NFNet checkpoint has no stem convolutions")
    result["stem.count"] = np.array([len(stem)], dtype=np.int32)
    for position in range(len(stem)):
        result.update(_standardised(checkpoint, f"stem.conv{position + 1}", dtype))

    for block in blocks:
        prefix = str(block["prefix"])
        leaves = ["conv1", "conv2", "conv3"]
        if bool(block["has_second_conv"]):
            leaves.append("conv2b")
        if bool(block["has_downsample"]):
            leaves.append("downsample.conv")
        for leaf in leaves:
            result.update(_standardised(checkpoint, f"{prefix}.{leaf}", dtype))
        if bool(block["has_gate"]):
            for leaf in ("fc1", "fc2"):
                result[f"{prefix}.attn_last.{leaf}.weight"] = checkpoint.tensor(
                    f"{prefix}.attn_last.{leaf}.weight"
                ).astype(dtype)
                result[f"{prefix}.attn_last.{leaf}.bias"] = checkpoint.tensor(
                    f"{prefix}.attn_last.{leaf}.bias"
                ).astype(dtype)
        result[f"{prefix}.skipinit_gain"] = checkpoint.tensor(f"{prefix}.skipinit_gain").astype(
            np.float32
        )

    result.update(_standardised(checkpoint, "final_conv", dtype))
    result["head.fc.weight"] = checkpoint.tensor("head.fc.weight").astype(dtype)
    result["head.fc.bias"] = checkpoint.tensor("head.fc.bias").astype(dtype)
    if result["head.fc.weight"].ndim != 2 or result["head.fc.bias"].shape != (
        result["head.fc.weight"].shape[0],
    ):
        raise ValueError("NFNet classifier weights have incompatible shapes")
    return result


def _activation(network, tensor, dtype: np.dtype):
    return graph.gamma_activation(network, tensor, gamma=_GELU_GAMMA, dtype=dtype)


def _convolution(network, tensor, weights, prefix, dtype, *, stride=1, groups=1):
    weight = weights[f"{prefix}.weight"]
    kernel = int(weight.shape[2])
    size = int(tensor.shape[2])
    return graph.convolution(
        network,
        tensor,
        weight,
        weights[f"{prefix}.bias"],
        stride=stride,
        padding=same_padding(size, kernel, stride),
        groups=groups,
        dtype=dtype,
    )


def _block(network, tensor, weights, block: dict[str, object], dtype: np.dtype):
    """One normalizer-free block.

    The activation comes first and is scaled by beta, and both the shortcut and
    the residual branch read that same scaled tensor. The branch is then scaled
    by its own learned gain and by alpha before being added back.
    """
    prefix = str(block["prefix"])
    stride = int(block["stride"])
    entry = graph.scale_by(
        network, _activation(network, tensor, dtype), float(block["beta"]), dtype=dtype
    )

    shortcut = tensor
    if bool(block["has_downsample"]):
        # The shortcut is taken from the activated tensor, not the block input.
        pooled = entry
        if stride > 1:
            pooled = graph.average_pool(network, entry, kernel=2, stride=stride)
        shortcut = _convolution(network, pooled, weights, f"{prefix}.downsample.conv", dtype)

    out = _convolution(network, entry, weights, f"{prefix}.conv1", dtype)
    grouped = weights[f"{prefix}.conv2.weight"]
    groups = max(1, int(grouped.shape[0]) // int(grouped.shape[1]))
    out = _convolution(
        network,
        _activation(network, out, dtype),
        weights,
        f"{prefix}.conv2",
        dtype,
        stride=stride,
        groups=groups,
    )
    if bool(block["has_second_conv"]):
        second = weights[f"{prefix}.conv2b.weight"]
        out = _convolution(
            network,
            _activation(network, out, dtype),
            weights,
            f"{prefix}.conv2b",
            dtype,
            groups=max(1, int(second.shape[0]) // int(second.shape[1])),
        )
    out = _convolution(network, _activation(network, out, dtype), weights, f"{prefix}.conv3", dtype)
    if bool(block["has_gate"]):
        gated = graph.squeeze_excite(
            network,
            out,
            weights[f"{prefix}.attn_last.fc1.weight"],
            weights[f"{prefix}.attn_last.fc1.bias"],
            weights[f"{prefix}.attn_last.fc2.weight"],
            weights[f"{prefix}.attn_last.fc2.bias"],
            dtype=dtype,
        )
        out = graph.scale_by(network, gated, _ATTENTION_GAIN, dtype=dtype)
    out = graph.scale_by(network, out, float(weights[f"{prefix}.skipinit_gain"]), dtype=dtype)
    out = graph.scale_by(network, out, _RESIDUAL_ALPHA, dtype=dtype)
    return graph.add(network, out, shortcut)


def _build_engine(
    raw: dict[str, Any],
    checkpoint: Checkpoint,
    precision: str,
    verbose: bool,
) -> tuple[bytes, dict[str, Any]]:
    if precision == "fp16":
        numpy_dtype, tensor_dtype = np.float16, trt.float16
    elif precision == "fp32":
        numpy_dtype, tensor_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"unsupported timm NFNet precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["head.fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("NFNet classifier dimensions do not match config.json")

    # The stem strides twice, then every stage after the first halves again.
    total_stride = 4
    for block in blocks:
        total_stride *= int(block["stride"])
    height, width = config["image_height"], config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"NFNet input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_nfnet: "
            f"image={height}x{width}, blocks={len(blocks)}, "
            f"classes={config['num_classes']}, precision={precision}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 1
    builder_config.avg_timing_iterations = 8
    builder_config.max_aux_streams = 0
    builder_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    _configure_precision(builder_config, precision)

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the NFNet input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the NFNet input cast")
        hidden = cast.get_output(0)

    # The stem alternates convolution and activation, and opens straight on a
    # convolution: the first block's own activation follows the last one.
    count = int(weights["stem.count"][0])
    for position in range(count):
        prefix = f"stem.conv{position + 1}"
        if position:
            hidden = _activation(network, hidden, numpy_dtype)
        stride = 2 if position in (0, count - 1) else 1
        hidden = _convolution(network, hidden, weights, prefix, numpy_dtype, stride=stride)

    for block in blocks:
        hidden = _block(network, hidden, weights, block, numpy_dtype)

    hidden = _convolution(network, hidden, weights, "final_conv", numpy_dtype)
    hidden = _activation(network, hidden, numpy_dtype)
    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network, hidden, weights["head.fc.weight"], weights["head.fc.bias"], dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the NFNet output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm NFNet engine build failed")
    return bytes(plan), config


def _configure_precision(builder_config, precision: str) -> None:
    """Switch off TensorRT's reduced-precision fp32 path for fp32 builds.

    TensorRT runs fp32 convolutions in TF32 by default, keeping ten mantissa
    bits rather than twenty-four. An fp32 build here means fp32.
    """
    if precision == "fp32":
        builder_config.clear_flag(trt.BuilderFlag.TF32)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm NFNet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_nfnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_nfnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_nfnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_nfnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_nfnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_nfnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_nfnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_nfnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_nfnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_nfnet does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_nfnet supports only max_sequence_length=1")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw, Checkpoint.open(model_dir), str(request.precision).lower(), bool(request.verbose)
    )
    writer.set_header(family="timm_nfnet", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": runtime["image_height"],
            "input_image_w": runtime["image_width"],
            "crop_pct": runtime["crop_pct"],
            "interpolation": runtime["interpolation"],
            "crop_mode": runtime["crop_mode"],
            "image_mean": runtime["mean"],
            "image_std": runtime["std"],
        },
    )
