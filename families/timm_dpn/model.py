# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm DPN classifiers.

A Dual Path Network carries two tensors between blocks: a residual path that is
summed and a densely connected path that is concatenated. Every block reads the
concatenation of the two and splits its own output back into the same pair.

The whole layout comes from the checkpoint. Stages and blocks are read from the
`features.conv<stage>_<block>` keys, the stride of a stage from whether its
first block projects with `c1x1_w_s1` or `c1x1_w_s2`, the group count of the
3x3 from its own weight shape, and the residual width from the shapes of the
projection and the block output. Nothing about the width schedule is tabulated
here, so dpn68/68b/92/98/107/131 all build from one code path.
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


# timm builds every DPN norm with eps=0.001, not the PyTorch default of 1e-5.
_BATCH_NORM_EPSILON = 1e-3
_BLOCK = re.compile(r"^features\.conv(\d+)_(\d+)\.(.+)$")
_STEM = "features.conv1_1"


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"DPN model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DPN config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("dpn"):
        raise ValueError(f"unsupported timm DPN model identity: {identity!r}")
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
        raise ValueError("DPN pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("DPN image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("DPN image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.875)),
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
        raise ValueError("DPN preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    """Recover the stage and block structure from the checkpoint keys."""
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.fullmatch(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".", 1)[0])
    # conv1_1 is the stem, not a dual-path block.
    stages = sorted({stage for stage, _ in leaves if stage > 1})
    if not stages:
        raise ValueError("DPN checkpoint has no features.conv<stage>_<block> tensors")
    if stages != list(range(2, len(stages) + 2)):
        raise ValueError("DPN stage indices are not contiguous from 2")

    blocks: list[dict[str, object]] = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"DPN stage {stage} block indices are not contiguous from 1")
        for index in indices:
            present = leaves[(stage, index)]
            if not {"c1x1_a", "c3x3_b", "c1x1_c"}.issubset(present):
                raise ValueError(
                    f"DPN features.conv{stage}_{index} is missing a dual-path convolution"
                )
            # The projection names its own stride: s1 keeps the resolution,
            # s2 halves it. Only the first block of a stage projects.
            projection = None
            stride = 1
            if "c1x1_w_s1" in present:
                projection, stride = "c1x1_w_s1", 1
            elif "c1x1_w_s2" in present:
                projection, stride = "c1x1_w_s2", 2
            if (index == 1) != (projection is not None):
                raise ValueError(
                    f"DPN features.conv{stage}_{index} projects only if it heads its stage"
                )
            blocks.append(
                {
                    "prefix": f"features.conv{stage}_{index}",
                    "stage": stage,
                    "stride": stride,
                    "projection": projection,
                    # The "b" widths end in two separate 1x1 convolutions
                    # instead of one convolution that is split afterwards.
                    "split_convs": "c1x1_c1" in present,
                }
            )
    return blocks


def _norm(checkpoint: Checkpoint, prefix: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one batch norm to a per-channel scale and shift.

    The statistics stay float32 through the division; only the result is cast,
    so a small running variance does not lose precision in fp16.
    """
    gamma = checkpoint.tensor(f"{prefix}.weight")
    beta = checkpoint.tensor(f"{prefix}.bias")
    mean = checkpoint.tensor(f"{prefix}.running_mean")
    variance = checkpoint.tensor(f"{prefix}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"DPN norm {prefix} has mismatched parameter shapes")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"DPN norm {prefix} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return scale.astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    result["stem.weight"] = checkpoint.tensor(f"{_STEM}.conv.weight").astype(dtype)
    scale, shift = _norm(checkpoint, f"{_STEM}.bn", dtype)
    result["stem.bn.scale"], result["stem.bn.shift"] = scale, shift

    for block in blocks:
        prefix = str(block["prefix"])
        leaves = ["c1x1_a", "c3x3_b"]
        projection = block["projection"]
        if isinstance(projection, str):
            leaves.append(projection)
        for leaf in leaves:
            scale, shift = _norm(checkpoint, f"{prefix}.{leaf}.bn", dtype)
            result[f"{prefix}.{leaf}.bn.scale"] = scale
            result[f"{prefix}.{leaf}.bn.shift"] = shift
            result[f"{prefix}.{leaf}.weight"] = checkpoint.tensor(
                f"{prefix}.{leaf}.conv.weight"
            ).astype(dtype)
        # c1x1_c always carries a norm; it carries a convolution only when the
        # block does not end in the two separate ones.
        scale, shift = _norm(checkpoint, f"{prefix}.c1x1_c.bn", dtype)
        result[f"{prefix}.c1x1_c.bn.scale"] = scale
        result[f"{prefix}.c1x1_c.bn.shift"] = shift
        if bool(block["split_convs"]):
            for leaf in ("c1x1_c1", "c1x1_c2"):
                result[f"{prefix}.{leaf}.weight"] = checkpoint.tensor(
                    f"{prefix}.{leaf}.weight"
                ).astype(dtype)
        else:
            result[f"{prefix}.c1x1_c.weight"] = checkpoint.tensor(
                f"{prefix}.c1x1_c.conv.weight"
            ).astype(dtype)

    last_stage = int(blocks[-1]["stage"])
    scale, shift = _norm(checkpoint, f"features.conv{last_stage}_bn_ac.bn", dtype)
    result["final.bn.scale"], result["final.bn.shift"] = scale, shift

    result["classifier.weight"] = checkpoint.tensor("classifier.weight").astype(dtype)
    result["classifier.bias"] = checkpoint.tensor("classifier.bias").astype(dtype)
    if result["classifier.weight"].ndim != 4 or result["classifier.bias"].shape != (
        result["classifier.weight"].shape[0],
    ):
        raise ValueError("DPN classifier weights have incompatible shapes")
    return result


def _stage_residual_widths(
    blocks: list[dict[str, object]], weights: dict[str, np.ndarray]
) -> dict[int, int]:
    """Channel count of the residual half of a block output, per stage.

    The width is a property of the stage, not of the block, so it is read once
    from the block that heads the stage and reused by the rest.

    With two output convolutions the first one is the residual half outright.
    With one convolution the split is implicit, so it comes from the widths:
    the projection emits the residual half plus two growth increments, and the
    output convolution emits the residual half plus one.
    """
    widths: dict[int, int] = {}
    for block in blocks:
        prefix = str(block["prefix"])
        stage = int(block["stage"])
        projection = block["projection"]
        if isinstance(projection, str):
            projected = int(weights[f"{prefix}.{projection}.weight"].shape[0])
            if bool(block["split_convs"]):
                produced = int(weights[f"{prefix}.c1x1_c1.weight"].shape[0])
                increment = int(weights[f"{prefix}.c1x1_c2.weight"].shape[0])
                residual = produced
            else:
                produced = int(weights[f"{prefix}.c1x1_c.weight"].shape[0])
                increment = projected - produced
                residual = produced - increment
            if increment <= 0 or residual <= 0 or projected != residual + 2 * increment:
                raise ValueError(f"DPN {prefix} has an inconsistent dual-path width")
            widths[stage] = residual
        elif stage not in widths:
            raise ValueError(f"DPN {prefix} appears before the block that heads its stage")
        elif bool(block["split_convs"]):
            # Every block of a stage must agree on where the split falls.
            produced = int(weights[f"{prefix}.c1x1_c1.weight"].shape[0])
            if produced != widths[stage]:
                raise ValueError(f"DPN {prefix} disagrees with its stage residual width")
    return widths


def _norm_relu(network, tensor, weights, prefix: str, dtype: np.dtype):
    return graph.relu(
        network,
        graph.batch_norm(
            network,
            tensor,
            weights[f"{prefix}.bn.scale"],
            weights[f"{prefix}.bn.shift"],
            dtype=dtype,
        ),
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
        raise ValueError(f"unsupported timm DPN precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["classifier.weight"].shape[0] != config["num_classes"]:
        raise ValueError("DPN classifier dimensions do not match config.json")

    # The stem convolution halves, then its max pool halves again.
    total_stride = 4
    for block in blocks:
        total_stride *= int(block["stride"])
    height = config["image_height"]
    width = config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"DPN input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_dpn: "
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
        raise RuntimeError("TensorRT rejected the DPN input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the DPN input cast")
        hidden = cast.get_output(0)

    stem = weights["stem.weight"]
    hidden = graph.convolution(
        network,
        hidden,
        stem,
        None,
        stride=2,
        padding=int(stem.shape[2]) // 2,
        dtype=numpy_dtype,
    )
    hidden = graph.batch_norm(
        network, hidden, weights["stem.bn.scale"], weights["stem.bn.shift"], dtype=numpy_dtype
    )
    hidden = graph.relu(network, hidden)
    hidden = graph.max_pool(network, hidden, 3, 2, 1)

    # Between blocks the network carries the pair (residual, dense).
    residual_widths = _stage_residual_widths(blocks, weights)
    residual = None
    dense = None
    for block in blocks:
        prefix = str(block["prefix"])
        stride = int(block["stride"])
        merged = hidden if residual is None else graph.concatenate(network, [residual, dense])
        width_residual = residual_widths[int(block["stage"])]

        projection = block["projection"]
        if isinstance(projection, str):
            shortcut = _norm_relu(network, merged, weights, f"{prefix}.{projection}", numpy_dtype)
            shortcut = graph.convolution(
                network,
                shortcut,
                weights[f"{prefix}.{projection}.weight"],
                None,
                stride=stride,
                dtype=numpy_dtype,
            )
            shape = tuple(int(value) for value in shortcut.shape)
            carried = graph.channel_slice(network, shortcut, 0, width_residual, shape)
            grown = graph.channel_slice(
                network, shortcut, width_residual, shape[1] - width_residual, shape
            )
        else:
            carried, grown = residual, dense

        tensor = _norm_relu(network, merged, weights, f"{prefix}.c1x1_a", numpy_dtype)
        tensor = graph.convolution(
            network, tensor, weights[f"{prefix}.c1x1_a.weight"], None, dtype=numpy_dtype
        )
        tensor = _norm_relu(network, tensor, weights, f"{prefix}.c3x3_b", numpy_dtype)
        # The 3x3 is grouped; the group count follows from its own weight shape.
        grouped = weights[f"{prefix}.c3x3_b.weight"]
        groups = max(1, int(grouped.shape[0]) // int(grouped.shape[1]))
        tensor = graph.convolution(
            network,
            tensor,
            grouped,
            None,
            stride=stride,
            padding=1,
            groups=groups,
            dtype=numpy_dtype,
        )
        tensor = _norm_relu(network, tensor, weights, f"{prefix}.c1x1_c", numpy_dtype)

        if bool(block["split_convs"]):
            produced_residual = graph.convolution(
                network, tensor, weights[f"{prefix}.c1x1_c1.weight"], None, dtype=numpy_dtype
            )
            produced_dense = graph.convolution(
                network, tensor, weights[f"{prefix}.c1x1_c2.weight"], None, dtype=numpy_dtype
            )
        else:
            tensor = graph.convolution(
                network, tensor, weights[f"{prefix}.c1x1_c.weight"], None, dtype=numpy_dtype
            )
            shape = tuple(int(value) for value in tensor.shape)
            produced_residual = graph.channel_slice(network, tensor, 0, width_residual, shape)
            produced_dense = graph.channel_slice(
                network, tensor, width_residual, shape[1] - width_residual, shape
            )

        residual = graph.add(network, carried, produced_residual)
        dense = graph.concatenate(network, [grown, produced_dense])

    hidden = graph.concatenate(network, [residual, dense])
    hidden = graph.batch_norm(
        network, hidden, weights["final.bn.scale"], weights["final.bn.shift"], dtype=numpy_dtype
    )
    hidden = graph.relu(network, hidden)
    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network,
        hidden,
        weights["classifier.weight"],
        weights["classifier.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the DPN output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm DPN engine build failed")
    return bytes(plan), config


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm DPN image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_dpn does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_dpn does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_dpn does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_dpn does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_dpn does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_dpn does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_dpn does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_dpn supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_dpn does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_dpn does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_dpn supports only max_sequence_length=1")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_dpn", task=request.task, backend=request.backend)
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
