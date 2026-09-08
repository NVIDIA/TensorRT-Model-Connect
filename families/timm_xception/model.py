# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm Xception classifiers.

Most of the layout is recovered from the checkpoint: the block count from the
`blocks.<n>` keys, and the stride from whether a block carries a projection
shortcut, because Xception downsamples exactly in the blocks that project.

One thing is not recoverable, because activations carry no weights: the final
block is built differently from the rest. Earlier blocks apply a ReLU *before*
each separable convolution, none inside, and add a residual. The last block
inverts both - it takes no residual and applies its activations inside the
separable convolutions instead. That is keyed on the block being last.
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


# Xception is a TensorFlow port and uses the TensorFlow batch-norm epsilon,
# not the PyTorch default. The two differ enough to change the argmax.
_BATCH_NORM_EPSILON = 1e-3

_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")
_STACK_CONVS = ("conv1", "conv2", "conv3")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Xception model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Xception config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("xception"):
        raise ValueError(f"unsupported timm Xception model identity: {identity!r}")
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("pretrained_cfg")
    source = nested if isinstance(nested, dict) else raw
    input_size = source.get("input_size", [3, 299, 299])
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
        raise ValueError("Xception pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.5, 0.5, 0.5])
    std = source.get("std", [0.5, 0.5, 0.5])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("Xception image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("Xception image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.903)),
        "interpolation": str(source.get("interpolation", "bicubic")),
    }
    if (
        result["image_height"] <= 0
        or result["image_width"] <= 0
        or result["num_classes"] <= 0
        or not 0.0 < result["crop_pct"] <= 1.0
        or any(value == 0.0 for value in result["std"])
        or result["interpolation"] not in {"bilinear", "bicubic"}
    ):
        raise ValueError("Xception preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[int, set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            leaves.setdefault(int(match.group(1)), set()).add(match.group(2).split(".", 1)[0])
    if not leaves:
        raise ValueError("Xception checkpoint has no blocks.<index> tensors")
    indices = sorted(leaves)
    if indices != list(range(len(indices))):
        raise ValueError("Xception block indices are not contiguous")
    blocks: list[dict[str, object]] = []
    for index in indices:
        present = leaves[index]
        if "stack" not in present:
            raise ValueError(f"Xception blocks.{index} has no separable convolution stack")
        has_shortcut = "shortcut" in present
        blocks.append(
            {
                "prefix": f"blocks.{index}",
                "has_shortcut": has_shortcut,
                "stride": 2 if has_shortcut else 1,
                "is_exit": index == indices[-1],
            }
        )
    return blocks


def _fold(checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    """Fold a batch norm into the convolution ahead of it.

    The statistics stay float32 through the division; only the result is cast,
    so a small running variance does not lose precision in fp16.
    """
    weight = checkpoint.tensor(conv)
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"Xception norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"Xception norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"Xception norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    folded = weight * scale.reshape(-1, 1, 1, 1)
    shift = beta - mean * scale
    return folded.astype(dtype), shift.astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for stem in ("stem.0", "stem.1"):
        weight, bias = _fold(checkpoint, f"{stem}.conv.weight", f"{stem}.bn", dtype)
        result[f"{stem}.weight"], result[f"{stem}.bias"] = weight, bias
    for block in blocks:
        prefix = str(block["prefix"])
        for leaf in _STACK_CONVS:
            stack = f"{prefix}.stack.{leaf}"
            for part, norm in (("dw", "bn_dw"), ("pw", "bn_pw")):
                weight, bias = _fold(
                    checkpoint, f"{stack}.conv_{part}.weight", f"{stack}.{norm}", dtype
                )
                result[f"{stack}.{part}.weight"] = weight
                result[f"{stack}.{part}.bias"] = bias
        if bool(block["has_shortcut"]):
            weight, bias = _fold(
                checkpoint, f"{prefix}.shortcut.conv.weight", f"{prefix}.shortcut.bn", dtype
            )
            result[f"{prefix}.shortcut.weight"] = weight
            result[f"{prefix}.shortcut.bias"] = bias
    result["head.fc.weight"] = checkpoint.tensor("head.fc.weight").astype(dtype)
    result["head.fc.bias"] = checkpoint.tensor("head.fc.bias").astype(dtype)
    if result["head.fc.weight"].ndim != 2 or result["head.fc.bias"].shape != (
        result["head.fc.weight"].shape[0],
    ):
        raise ValueError("Xception classifier weights have incompatible shapes")
    return result


def _separable(network, tensor, weights, prefix: str, dtype, *, stride: int, inner_act: bool):
    """A depthwise convolution then a pointwise one, each with a folded norm."""
    depthwise = weights[f"{prefix}.dw.weight"]
    tensor = graph.convolution(
        network,
        tensor,
        depthwise,
        weights[f"{prefix}.dw.bias"],
        stride=stride,
        padding=1,
        groups=int(depthwise.shape[0]),
        dtype=dtype,
    )
    if inner_act:
        tensor = graph.relu(network, tensor)
    tensor = graph.convolution(
        network,
        tensor,
        weights[f"{prefix}.pw.weight"],
        weights[f"{prefix}.pw.bias"],
        dtype=dtype,
    )
    if inner_act:
        tensor = graph.relu(network, tensor)
    return tensor


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
        raise ValueError(f"unsupported timm Xception precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["head.fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("Xception classifier dimensions do not match config.json")

    # The two stem convolutions stride once, then every projecting block halves.
    total_stride = 2
    for block in blocks:
        total_stride *= int(block["stride"])
    height = config["image_height"]
    width = config["image_width"]
    if verbose:
        print(
            "[trtmc build] timm_xception: "
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

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the Xception input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Xception input cast")
        hidden = cast.get_output(0)

    for stem, stride in (("stem.0", 2), ("stem.1", 1)):
        hidden = graph.convolution(
            network,
            hidden,
            weights[f"{stem}.weight"],
            weights[f"{stem}.bias"],
            stride=stride,
            padding=1,
            dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)

    for block in blocks:
        prefix = str(block["prefix"])
        stride = int(block["stride"])
        is_exit = bool(block["is_exit"])
        skip = hidden
        tensor = hidden
        for position, leaf in enumerate(_STACK_CONVS):
            if not is_exit:
                tensor = graph.relu(network, tensor)
            tensor = _separable(
                network,
                tensor,
                weights,
                f"{prefix}.stack.{leaf}",
                numpy_dtype,
                # Only the third separable convolution carries the stride.
                stride=stride if position == len(_STACK_CONVS) - 1 else 1,
                inner_act=is_exit,
            )
        if bool(block["has_shortcut"]):
            skip = graph.convolution(
                network,
                skip,
                weights[f"{prefix}.shortcut.weight"],
                weights[f"{prefix}.shortcut.bias"],
                stride=stride,
                dtype=numpy_dtype,
            )
        hidden = tensor if is_exit else graph.add(network, tensor, skip)

    shape = hidden.shape
    hidden = graph.global_average_pool(network, hidden, int(shape[2]), int(shape[3]))
    logits = graph.classifier(
        network,
        hidden,
        weights["head.fc.weight"],
        weights["head.fc.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Xception output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm Xception engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm Xception image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_xception does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_xception does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_xception does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_xception does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_xception does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_xception does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_xception does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_xception supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_xception does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_xception does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_xception", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": runtime["image_height"],
            "input_image_w": runtime["image_width"],
            "crop_pct": runtime["crop_pct"],
            "interpolation": runtime["interpolation"],
            "image_mean": runtime["mean"],
            "image_std": runtime["std"],
        },
    )
