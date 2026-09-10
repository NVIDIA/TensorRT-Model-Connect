# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm Res2Net classifiers.

Res2Net replaces the single 3x3 of a ResNet bottleneck with a chain of
narrower 3x3 convolutions. The 1x1 output is split into `scale` equal
chunks; each chunk after the first is added to the running result before
its own convolution, and the last chunk bypasses the chain entirely. The
chunks are concatenated back before the final 1x1, so one block sees several
receptive field sizes at once.

The layout comes from the checkpoint: stage depths from the
`layer<stage>.<block>` keys, the scale from how many `convs` entries a block
has, the chunk width and group count from those convolutions own shapes, the
stem shape from whether `conv1` is a single tensor or a sequence, and the
shortcut form from whether `downsample` starts at index 0 or 1.
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


_BATCH_NORM_EPSILON = 1e-5
_STAGES = ("layer1", "layer2", "layer3", "layer4")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Res2Net model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Res2Net config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith(("res2net", "res2next")):
        raise ValueError(f"unsupported timm Res2Net model identity: {identity!r}")
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
        raise ValueError("Res2Net pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("Res2Net image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("Res2Net image std must contain three values")
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
        raise ValueError("Res2Net preprocessing or classifier config is invalid")
    return result


def _pooled_shortcut(names: frozenset[str]) -> bool:
    """Whether the projection shortcut pools before its convolution."""
    plain = "layer1.0.downsample.0.weight" in names
    pooled = "layer1.0.downsample.1.weight" in names
    if plain:
        return False
    if pooled:
        return True
    raise ValueError("Res2Net checkpoint has no projection shortcut in layer1.0")


def _layout(checkpoint: Checkpoint) -> dict[str, Any]:
    names = checkpoint.names
    depths: list[int] = []
    for stage in _STAGES:
        pattern = re.compile(rf"^{stage}\.(\d+)\.")
        indices = {int(match.group(1)) for match in map(pattern.match, names) if match}
        if not indices:
            raise ValueError(f"Res2Net checkpoint has no blocks for stage {stage}")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Res2Net stage {stage} block indices are not contiguous")
        depths.append(len(indices))

    chain = re.compile(r"^layer1\.0\.convs\.(\d+)\.weight$")
    rungs = {int(match.group(1)) for match in map(chain.match, names) if match}
    if not rungs:
        raise ValueError(
            "Res2Net checkpoint has no convs chain; a plain ResNet bottleneck "
            "belongs to a different family"
        )
    if sorted(rungs) != list(range(len(rungs))):
        raise ValueError("Res2Net convs indices are not contiguous from 0")
    # The last chunk skips the chain, so there is one convolution fewer than
    # there are chunks.
    scale = len(rungs) + 1

    return {
        "depths": depths,
        "scale": scale,
        # The "d" variants open with three 3x3 convolutions instead of one 7x7.
        "deep_stem": "conv1.0.weight" in names,
        # ResNet-D shortcuts pool before the 1x1, which shifts the indices by
        # one. Index 1 is a norm in the plain layout and the convolution in the
        # pooled one, so index 0 is what tells the two apart.
        "pooled_shortcut": _pooled_shortcut(names),
    }


def _fold_norm(
    checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a batch norm into the convolution ahead of it."""
    weight = checkpoint.tensor(conv)
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"Res2Net norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"Res2Net norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"Res2Net norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    layout: dict[str, Any],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if layout["deep_stem"]:
        # conv1.0 -> conv1.1, conv1.3 -> conv1.4, conv1.6 -> bn1
        for position, (conv, norm) in enumerate(
            (("conv1.0", "conv1.1"), ("conv1.3", "conv1.4"), ("conv1.6", "bn1"))
        ):
            weight, bias = _fold_norm(checkpoint, f"{conv}.weight", norm, dtype)
            result[f"stem.{position}.weight"] = weight
            result[f"stem.{position}.bias"] = bias
    else:
        result["stem.0.weight"], result["stem.0.bias"] = _fold_norm(
            checkpoint, "conv1.weight", "bn1", dtype
        )

    shortcut_conv = "downsample.1" if layout["pooled_shortcut"] else "downsample.0"
    shortcut_norm = "downsample.2" if layout["pooled_shortcut"] else "downsample.1"
    for stage, depth in zip(_STAGES, layout["depths"]):
        for index in range(depth):
            prefix = f"{stage}.{index}"
            for position, leaf in ((1, "conv1"), (3, "conv3")):
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.{leaf}.weight", f"{prefix}.bn{position}", dtype
                )
                result[f"{prefix}.{leaf}.weight"] = weight
                result[f"{prefix}.{leaf}.bias"] = bias
            for rung in range(layout["scale"] - 1):
                weight, bias = _fold_norm(
                    checkpoint,
                    f"{prefix}.convs.{rung}.weight",
                    f"{prefix}.bns.{rung}",
                    dtype,
                )
                result[f"{prefix}.convs.{rung}.weight"] = weight
                result[f"{prefix}.convs.{rung}.bias"] = bias
            if f"{prefix}.{shortcut_conv}.weight" in checkpoint.names:
                weight, bias = _fold_norm(
                    checkpoint,
                    f"{prefix}.{shortcut_conv}.weight",
                    f"{prefix}.{shortcut_norm}",
                    dtype,
                )
                result[f"{prefix}.downsample.weight"] = weight
                result[f"{prefix}.downsample.bias"] = bias

    result["fc.weight"] = checkpoint.tensor("fc.weight").astype(dtype)
    result["fc.bias"] = checkpoint.tensor("fc.bias").astype(dtype)
    if result["fc.weight"].ndim != 2 or result["fc.bias"].shape != (result["fc.weight"].shape[0],):
        raise ValueError("Res2Net classifier weights have incompatible shapes")
    return result


def _block(
    network,
    tensor,
    weights,
    prefix: str,
    dtype,
    *,
    stride: int,
    scale: int,
    heads_stage: bool,
    pooled_shortcut: bool,
):
    """One Res2Net bottleneck.

    The 1x1 output is split into `scale` chunks. Chunk 0 starts the chain; each
    later chunk is added to the previous rung's result before its own 3x3,
    except at the head of a stage where every rung starts from its own chunk
    because the spatial size has just changed. The final chunk skips the chain.
    """
    shortcut = tensor
    out = graph.convolution(
        network,
        tensor,
        weights[f"{prefix}.conv1.weight"],
        weights[f"{prefix}.conv1.bias"],
        dtype=dtype,
    )
    out = graph.relu(network, out)

    width = int(weights[f"{prefix}.convs.0.weight"].shape[0])
    chunks = [graph.channel_slice(network, out, rung * width, width) for rung in range(scale)]

    pieces = []
    carried = None
    for rung in range(scale - 1):
        source = (
            chunks[rung]
            if (rung == 0 or heads_stage)
            else graph.add(network, carried, chunks[rung])
        )
        grouped = weights[f"{prefix}.convs.{rung}.weight"]
        groups = max(1, int(grouped.shape[0]) // int(grouped.shape[1]))
        carried = graph.relu(
            network,
            graph.convolution(
                network,
                source,
                grouped,
                weights[f"{prefix}.convs.{rung}.bias"],
                stride=stride,
                padding=1,
                groups=groups,
                dtype=dtype,
            ),
        )
        pieces.append(carried)

    # The last chunk carries no convolution. At the head of a stage it still
    # has to lose the same resolution as the rungs did, which timm does with an
    # average pool that counts its padding.
    last = chunks[-1]
    if heads_stage:
        last = graph.average_pool(
            network, last, kernel=3, stride=stride, padding=1, count_include_pad=True
        )
    pieces.append(last)

    out = graph.concatenate(network, pieces)
    out = graph.convolution(
        network,
        out,
        weights[f"{prefix}.conv3.weight"],
        weights[f"{prefix}.conv3.bias"],
        dtype=dtype,
    )
    if f"{prefix}.downsample.weight" in weights:
        if pooled_shortcut:
            # ResNet-D pools first, then projects with a stride-1 convolution.
            if stride > 1:
                shortcut = graph.average_pool(
                    network,
                    shortcut,
                    kernel=2,
                    stride=stride,
                    padding=0,
                    count_include_pad=False,
                )
            shortcut = graph.convolution(
                network,
                shortcut,
                weights[f"{prefix}.downsample.weight"],
                weights[f"{prefix}.downsample.bias"],
                dtype=dtype,
            )
        else:
            shortcut = graph.convolution(
                network,
                shortcut,
                weights[f"{prefix}.downsample.weight"],
                weights[f"{prefix}.downsample.bias"],
                stride=stride,
                dtype=dtype,
            )
    return graph.relu(network, graph.add(network, out, shortcut))


def _configure_precision(builder_config, precision: str) -> None:
    """Switch off TensorRT's reduced-precision fp32 path for fp32 builds.

    TensorRT runs fp32 convolutions in TF32 by default, which keeps ten
    mantissa bits rather than twenty-four. Measured against timm on the nine
    published Res2Net checkpoints, that costs about three orders of magnitude
    of accuracy, and on res2net50_26w_8s it moves the predicted class. An fp32
    build here means fp32; fp16 builds are left alone.
    """
    if precision == "fp32":
        builder_config.clear_flag(trt.BuilderFlag.TF32)


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
        raise ValueError(f"unsupported timm Res2Net precision: {precision}")
    config = _preprocess_config(raw)
    layout = _layout(checkpoint)
    weights = _weights(checkpoint, layout, numpy_dtype)
    if weights["fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("Res2Net classifier dimensions do not match config.json")

    # The stem strides twice, then every stage after the first halves again.
    total_stride = 4 * 2 ** (len(_STAGES) - 1)
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"Res2Net input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_res2net: "
            f"image={height}x{width}, depths={layout['depths']}, "
            f"bottleneck={layout['bottleneck']}, classes={config['num_classes']}, "
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
    _configure_precision(builder_config, precision)

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the Res2Net input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Res2Net input cast")
        hidden = cast.get_output(0)

    # A plain stem is one 7x7 at stride 2; the deep stem is three 3x3s where
    # only the first strides.
    for position in range(3 if layout["deep_stem"] else 1):
        stem = weights[f"stem.{position}.weight"]
        hidden = graph.convolution(
            network,
            hidden,
            stem,
            weights[f"stem.{position}.bias"],
            stride=2 if position == 0 else 1,
            padding=int(stem.shape[2]) // 2,
            dtype=numpy_dtype,
        )
        hidden = graph.relu(network, hidden)
    hidden = graph.max_pool(network, hidden, kernel=3, stride=2, padding=1)

    for position, (stage, depth) in enumerate(zip(_STAGES, layout["depths"])):
        for index in range(depth):
            hidden = _block(
                network,
                hidden,
                weights,
                f"{stage}.{index}",
                numpy_dtype,
                stride=2 if (position > 0 and index == 0) else 1,
                scale=layout["scale"],
                # Every stage opens with the block that changes the shape.
                heads_stage=index == 0,
                pooled_shortcut=bool(layout["pooled_shortcut"]),
            )

    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network, hidden, weights["fc.weight"], weights["fc.bias"], dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the Res2Net output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm Res2Net engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm Res2Net image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_res2net does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_res2net does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_res2net does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_res2net does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_res2net does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_res2net does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_res2net does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_res2net supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_res2net does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_res2net does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_res2net", task=request.task, backend=request.backend)
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
