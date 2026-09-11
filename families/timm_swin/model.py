# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm Swin Transformer classifiers.

Swin is a convolutional network shaped like a transformer: a large-kernel
depthwise convolution stands in for attention, followed by a LayerNorm and a
two-layer MLP with one GELU.

timm applies that MLP by transposing to channels-last, running two Linear
layers, and transposing back. A Linear over the channel axis is exactly a 1x1
convolution on NCHW, so it is emitted that way here and the network never
leaves NCHW.
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


_LAYER_NORM_EPSILON = 1e-5

# Positions that the cyclic roll placed in the same window but that are not
# actually neighbours are pushed this far below the real scores.
_MASK_FILL = -100.0
_BLOCK = re.compile(r"^layers\.(\d+)\.blocks\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Swin model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Swin config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("swin"):
        raise ValueError(f"unsupported timm Swin Transformer model identity: {identity!r}")
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("pretrained_cfg")
    source = nested if isinstance(nested, dict) else raw
    input_size = source.get("input_size", [3, 224, 224])
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
        raise ValueError("Swin pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("Swin image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("Swin image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.95)),
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
        raise ValueError("Swin preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[int]:
    depths: dict[int, set[int]] = {}
    for name in checkpoint.names:
        match = _BLOCK.match(name)
        if match:
            depths.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    if not depths:
        raise ValueError("Swin checkpoint has no layers.<layer>.blocks.<index> tensors")
    layers = sorted(depths)
    if layers != list(range(len(layers))):
        raise ValueError("Swin layer indices are not contiguous")
    result: list[int] = []
    for layer in layers:
        indices = sorted(depths[layer])
        if indices != list(range(len(indices))):
            raise ValueError(f"Swin layer {layer} block indices are not contiguous")
        result.append(len(indices))
    return result


def _attention_bias(
    index: np.ndarray, table: np.ndarray, mask: np.ndarray | None, area: int, heads: int
) -> np.ndarray:
    """Fold the relative-position bias and the window mask into one tensor.

    Returns the additive bias TensorRT's attention layer takes. Doing this on the
    host keeps the gather and the broadcast out of the engine, and the result is
    constant for a fixed input size.
    """
    bias = table[index.reshape(-1).astype(np.int64)].reshape(area, area, heads)
    bias = np.transpose(bias, (2, 0, 1))[None]
    if mask is None:
        return bias
    filled = np.where(mask != 0, _MASK_FILL, 0.0).astype(np.float32)
    return bias + filled[:, None]


def _weights(checkpoint: Checkpoint, depths: list[int], dtype: np.dtype):
    def take(name: str) -> np.ndarray:
        return checkpoint.tensor(name).astype(dtype)

    result: dict[str, np.ndarray] = {
        "patch.weight": take("patch_embed.proj.weight"),
        "patch.bias": take("patch_embed.proj.bias"),
        "patch.norm.weight": take("patch_embed.norm.weight"),
        "patch.norm.bias": take("patch_embed.norm.bias"),
        "norm.weight": take("norm.weight"),
        "norm.bias": take("norm.bias"),
        "head.weight": take("head.fc.weight"),
        "head.bias": take("head.fc.bias"),
    }
    for layer, depth in enumerate(depths):
        downsample = f"layers.{layer}.downsample"
        if f"{downsample}.reduction.weight" in checkpoint.names:
            result[f"{downsample}.norm.weight"] = take(f"{downsample}.norm.weight")
            result[f"{downsample}.norm.bias"] = take(f"{downsample}.norm.bias")
            result[f"{downsample}.weight"] = take(f"{downsample}.reduction.weight")
        for index in range(depth):
            prefix = f"layers.{layer}.blocks.{index}"
            for leaf in ("norm1", "norm2"):
                result[f"{prefix}.{leaf}.weight"] = take(f"{prefix}.{leaf}.weight")
                result[f"{prefix}.{leaf}.bias"] = take(f"{prefix}.{leaf}.bias")
            for leaf in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                result[f"{prefix}.{leaf}.weight"] = take(f"{prefix}.{leaf}.weight")
                result[f"{prefix}.{leaf}.bias"] = take(f"{prefix}.{leaf}.bias")
            # Geometry, not learned weights: read exactly and only on the host.
            result[f"{prefix}.rel_index"] = checkpoint.tensor(
                f"{prefix}.attn.relative_position_index"
            )
            result[f"{prefix}.rel_table"] = checkpoint.tensor(
                f"{prefix}.attn.relative_position_bias_table"
            )
            if f"{prefix}.attn_mask" in checkpoint.names:
                result[f"{prefix}.attn_mask"] = checkpoint.tensor(f"{prefix}.attn_mask")
    if result["head.weight"].ndim != 2 or result["head.bias"].shape != (
        result["head.weight"].shape[0],
    ):
        raise ValueError("Swin classifier weights have incompatible shapes")
    return result


def _linear(network, tensor, weights, prefix: str, dtype):
    return graph.matmul_constant(
        network, tensor, weights[f"{prefix}.weight"], weights.get(f"{prefix}.bias"), dtype=dtype
    )


def _window_attention(
    network, tensor, weights, prefix: str, dtype, *, height, width, channels, window
):
    """Window attention over an NHWC map whose sides divide the window."""
    windows_h, windows_w = height // window, width // window
    num_windows = windows_h * windows_w
    area = window * window
    table = np.asarray(weights[f"{prefix}.rel_table"], dtype=np.float32)
    heads = int(table.shape[1])
    head_dim = channels // heads
    mask = weights.get(f"{prefix}.attn_mask")
    bias = _attention_bias(
        np.asarray(weights[f"{prefix}.rel_index"]),
        table,
        None if mask is None else np.asarray(mask, dtype=np.float32),
        area,
        heads,
    )
    if bias.shape[0] == 1 and num_windows != 1:
        bias = np.broadcast_to(bias, (num_windows, heads, area, area))

    tokens = graph.permute(
        network, tensor,
        (windows_h, window, windows_w, window, channels),
        (0, 2, 1, 3, 4),
        (num_windows, area, channels),
    )
    qkv = _linear(network, tokens, weights, f"{prefix}.attn.qkv", dtype)
    qkv = graph.permute(
        network, qkv, (num_windows, area, 3, heads, head_dim), (2, 0, 3, 1, 4), None
    )
    parts = []
    for which in range(3):
        piece = graph.slice_tokens(network, qkv, which, 1, axis=0)
        parts.append(graph.reshape(network, piece, (num_windows, heads, area, head_dim)))
    bias_tensor = graph.constant(
        network, np.ascontiguousarray(bias), dtype=dtype, like=parts[0]
    )
    context = graph.attention(network, parts[0], parts[1], parts[2], bias_tensor, dtype=dtype)
    merged = graph.permute(network, context, None, (0, 2, 1, 3), (num_windows, area, channels))
    merged = _linear(network, merged, weights, f"{prefix}.attn.proj", dtype)
    return graph.permute(
        network, merged,
        (windows_h, windows_w, window, window, channels),
        (0, 2, 1, 3, 4),
        (1, height, width, channels),
    )


def _block(network, tensor, weights, prefix: str, dtype, *, height, width, channels, window):
    shift = window // 2 if f"{prefix}.attn_mask" in weights else 0
    residual = tensor
    hidden = graph.layer_norm(
        network, tensor, weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=dtype,
    )
    if shift:
        hidden = graph.roll(network, hidden, (-shift, -shift), (1, 2))
    hidden = _window_attention(
        network, hidden, weights, prefix, dtype,
        height=height, width=width, channels=channels, window=window,
    )
    if shift:
        hidden = graph.roll(network, hidden, (shift, shift), (1, 2))
    tensor = graph.add(network, residual, hidden)

    residual = tensor
    hidden = graph.layer_norm(
        network, tensor, weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=dtype,
    )
    hidden = _linear(network, hidden, weights, f"{prefix}.mlp.fc1", dtype)
    hidden = graph.gelu(network, hidden, dtype=dtype)
    hidden = _linear(network, hidden, weights, f"{prefix}.mlp.fc2", dtype)
    return graph.add(network, residual, hidden)


def _patch_merging(network, tensor, weights, prefix: str, dtype, *, height, width, channels):
    """Fold each 2x2 patch into the channel axis, then project it down.

    The four positions are interleaved column-inner-then-row-inner, which is the
    order the reduction weights were trained against; swapping them still builds.
    """
    folded = graph.permute(
        network, tensor,
        (height // 2, 2, width // 2, 2, channels),
        (0, 2, 3, 1, 4),
        (1, height // 2, width // 2, 4 * channels),
    )
    folded = graph.layer_norm(
        network, folded, weights[f"{prefix}.norm.weight"], weights[f"{prefix}.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=dtype,
    )
    return graph.matmul_constant(
        network, folded, weights[f"{prefix}.weight"], None, dtype=dtype
    )


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
        raise ValueError(f"unsupported timm Swin precision: {precision}")
    config = _preprocess_config(raw)
    depths = _layout(checkpoint)
    weights = _weights(checkpoint, depths, numpy_dtype)
    if weights["head.weight"].shape[0] != config["num_classes"]:
        raise ValueError("Swin classifier dimensions do not match config.json")

    patch_weight = weights["patch.weight"]
    patch = int(patch_weight.shape[2])
    channels = int(patch_weight.shape[0])
    height = config["image_height"] // patch
    width = config["image_width"] // patch
    # relative_position_index is [area, area] and a window is square, so the
    # side length is the fourth root of its element count.
    area = int(round(np.sqrt(np.asarray(weights["layers.0.blocks.0.rel_index"]).size)))
    window = int(round(np.sqrt(area)))
    if window * window != area:
        raise ValueError(f"Swin relative_position_index implies a non-square window: {area}")
    if verbose:
        print(
            "[trtmc build] timm_swin: "
            f"image={config['image_height']}x{config['image_width']}, patch={patch}, "
            f"depths={depths}, window={window}, classes={config['num_classes']}, "
            f"precision={precision}",
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

    pixels = network.add_input(
        "pixel_values", trt.float32, (1, 3, config["image_height"], config["image_width"])
    )
    if pixels is None:
        raise RuntimeError("TensorRT rejected the Swin input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Swin input cast")
        hidden = cast.get_output(0)

    hidden = graph.patch_convolution(
        network, hidden, patch_weight, weights["patch.bias"], patch=patch, dtype=numpy_dtype
    )
    # NCHW -> NHWC: the rest of the model indexes tokens spatially.
    hidden = graph.permute(network, hidden, None, (0, 2, 3, 1), None)
    hidden = graph.layer_norm(
        network, hidden, weights["patch.norm.weight"], weights["patch.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=numpy_dtype,
    )

    for layer, depth in enumerate(depths):
        downsample = f"layers.{layer}.downsample"
        if f"{downsample}.weight" in weights:
            hidden = _patch_merging(
                network, hidden, weights, downsample, numpy_dtype,
                height=height, width=width, channels=channels,
            )
            height, width = height // 2, width // 2
            channels = int(weights[f"{downsample}.weight"].shape[0])
        for index in range(depth):
            hidden = _block(
                network, hidden, weights, f"layers.{layer}.blocks.{index}", numpy_dtype,
                height=height, width=width, channels=channels,
                # The last stage is one window wide, so it cannot shift.
                window=min(window, height, width),
            )

    hidden = graph.layer_norm(
        network, hidden, weights["norm.weight"], weights["norm.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=numpy_dtype,
    )
    hidden = graph.mean_spatial_nhwc(network, hidden)
    logits = _linear(network, hidden, weights, "head", numpy_dtype)
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Swin output cast")
        logits = cast.get_output(0)
    logits = graph.reshape(network, logits, (1, config["num_classes"]))
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm Swin engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm Swin Transformer image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_swin does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_swin does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_swin does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_swin does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_swin does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_swin does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_swin does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_swin supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_swin does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_swin does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_swin", task=request.task, backend=request.backend)
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
