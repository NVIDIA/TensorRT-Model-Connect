# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm MobileViT classifiers.

MobileViT alternates two kinds of block. Most of the network is MobileNetV2's
inverted residual: expand with a 1x1, filter with a depthwise kxk, project back
with a 1x1. Three stages end in a MobileViT block instead, which keeps the
convolution for local detail and adds self-attention for everything else.

The MobileViT block is the interesting part. It cuts the feature map into 2x2
patches and builds one token sequence per position inside a patch, so the four
sequences run independently and each token sees the same position in every
other patch. That is what makes the attention global while staying cheap. The
block then folds the tokens back into a feature map, projects, and fuses with
its own input.

The layout comes from the checkpoint: stage depths from the
`stages.<stage>.<block>` keys, which blocks are MobileViT ones from whether
they carry a `transformer`, the depthwise group count from its own weight
shape, and the token width from the attention weights.
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
_LAYER_NORM_EPSILON = 1e-5
# Every MobileViT block cuts the map into patches of this size.
_PATCH = 2
# Every published MobileViT width uses four attention heads; the checkpoint
# cannot say so, because a fused qkv has the same shape either way.
_ATTENTION_HEADS = 4
_BLOCK = re.compile(r"^stages\.(\d+)\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"MobileViT model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("MobileViT config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("mobilevit"):
        raise ValueError(f"unsupported timm MobileViT model identity: {identity!r}")
    if identity.lower().startswith("mobilevitv2"):
        raise NotImplementedError(
            "MobileViTv2 uses a separable linear attention and a different block; "
            "it is not covered by this family"
        )
    return value


def _preprocess_config(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("pretrained_cfg")
    source = nested if isinstance(nested, dict) else raw
    input_size = source.get("input_size", [3, 256, 256])
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
        raise ValueError("MobileViT pretrained input_size must be [3, height, width]")
    # MobileViT trains on raw [0, 1] pixels, so its mean is zero and its std one.
    mean = source.get("mean", [0.0, 0.0, 0.0])
    std = source.get("std", [1.0, 1.0, 1.0])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("MobileViT image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("MobileViT image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 0.9)),
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
        raise ValueError("MobileViT preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> list[dict[str, object]]:
    leaves: dict[tuple[int, int], set[str]] = {}
    for name in checkpoint.names:
        match = _BLOCK.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(name[match.end() :].split(".", 1)[0])
    if not leaves:
        raise ValueError("MobileViT checkpoint has no stages.<stage>.<block> tensors")
    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("MobileViT stage indices are not contiguous from 0")

    blocks: list[dict[str, object]] = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"MobileViT stage {stage} block indices are not contiguous from 0")
        for index in indices:
            present = leaves[(stage, index)]
            prefix = f"stages.{stage}.{index}"
            if "transformer" in present:
                depth = _transformer_depth(checkpoint, prefix)
                blocks.append(
                    {"prefix": prefix, "stage": stage, "kind": "mobilevit", "depth": depth}
                )
            elif {"conv1_1x1", "conv2_kxk", "conv3_1x1"}.issubset(present):
                blocks.append({"prefix": prefix, "stage": stage, "kind": "inverted"})
            else:
                raise ValueError(f"MobileViT {prefix} is neither an inverted residual nor a block")
    return blocks


def _transformer_depth(checkpoint: Checkpoint, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.transformer\.(\d+)\.")
    indices = {int(match.group(1)) for match in map(pattern.match, checkpoint.names) if match}
    if not indices:
        raise ValueError(f"MobileViT {prefix} has no transformer layers")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"MobileViT {prefix} transformer indices are not contiguous from 0")
    return len(indices)


def _fold_norm(
    checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"MobileViT norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"MobileViT norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"MobileViT norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _weights(
    checkpoint: Checkpoint,
    blocks: list[dict[str, object]],
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    result["stem.weight"], result["stem.bias"] = _fold_norm(
        checkpoint, "stem.conv.weight", "stem.bn", dtype
    )

    for block in blocks:
        prefix = str(block["prefix"])
        if block["kind"] == "inverted":
            for leaf in ("conv1_1x1", "conv2_kxk", "conv3_1x1"):
                weight, bias = _fold_norm(
                    checkpoint, f"{prefix}.{leaf}.conv.weight", f"{prefix}.{leaf}.bn", dtype
                )
                result[f"{prefix}.{leaf}.weight"] = weight
                result[f"{prefix}.{leaf}.bias"] = bias
            continue

        for leaf in ("conv_kxk", "conv_proj", "conv_fusion"):
            weight, bias = _fold_norm(
                checkpoint, f"{prefix}.{leaf}.conv.weight", f"{prefix}.{leaf}.bn", dtype
            )
            result[f"{prefix}.{leaf}.weight"] = weight
            result[f"{prefix}.{leaf}.bias"] = bias
        # conv_1x1 carries neither a norm nor a bias.
        result[f"{prefix}.conv_1x1.weight"] = checkpoint.tensor(f"{prefix}.conv_1x1.weight").astype(
            dtype
        )
        for leaf in ("weight", "bias"):
            result[f"{prefix}.norm.{leaf}"] = checkpoint.tensor(f"{prefix}.norm.{leaf}").astype(
                dtype
            )

        for layer in range(int(block["depth"])):
            source = f"{prefix}.transformer.{layer}"
            # The fused qkv is split on the host, so the graph needs no slice.
            qkv_weight = checkpoint.tensor(f"{source}.attn.qkv.weight")
            qkv_bias = checkpoint.tensor(f"{source}.attn.qkv.bias")
            width = qkv_weight.shape[0] // 3
            if qkv_weight.shape[0] != 3 * width or qkv_weight.shape[1] != width:
                raise ValueError(f"MobileViT {source} qkv is not three square projections")
            for position, name in enumerate(("query", "key", "value")):
                lo, hi = position * width, (position + 1) * width
                result[f"{source}.{name}.weight"] = qkv_weight[lo:hi].astype(dtype)
                result[f"{source}.{name}.bias"] = qkv_bias[lo:hi].astype(dtype)
            for leaf in ("attn.proj", "mlp.fc1", "mlp.fc2", "norm1", "norm2"):
                for part in ("weight", "bias"):
                    result[f"{source}.{leaf}.{part}"] = checkpoint.tensor(
                        f"{source}.{leaf}.{part}"
                    ).astype(dtype)

    result["final_conv.weight"], result["final_conv.bias"] = _fold_norm(
        checkpoint, "final_conv.conv.weight", "final_conv.bn", dtype
    )
    result["head.fc.weight"] = checkpoint.tensor("head.fc.weight").astype(dtype)
    result["head.fc.bias"] = checkpoint.tensor("head.fc.bias").astype(dtype)
    if result["head.fc.weight"].ndim != 2 or result["head.fc.bias"].shape != (
        result["head.fc.weight"].shape[0],
    ):
        raise ValueError("MobileViT classifier weights have incompatible shapes")
    return result


def _convolution(network, tensor, weights, prefix, dtype, *, stride=1, padding=0, groups=1):
    return graph.convolution(
        network,
        tensor,
        weights[f"{prefix}.weight"],
        weights.get(f"{prefix}.bias"),
        stride=stride,
        padding=padding,
        groups=groups,
        dtype=dtype,
    )


def _inverted_residual(network, tensor, weights, prefix: str, dtype: np.dtype, *, stride: int):
    """MobileNetV2's block: expand, filter depthwise, project, maybe add back.

    The shortcut exists only when the block changes neither the resolution nor
    the channel count, which is how timm decides it.
    """
    shortcut = tensor
    out = graph.silu(network, _convolution(network, tensor, weights, f"{prefix}.conv1_1x1", dtype))
    depthwise = weights[f"{prefix}.conv2_kxk.weight"]
    kernel = int(depthwise.shape[2])
    out = graph.silu(
        network,
        _convolution(
            network,
            out,
            weights,
            f"{prefix}.conv2_kxk",
            dtype,
            stride=stride,
            padding=kernel // 2,
            groups=int(depthwise.shape[0]),
        ),
    )
    out = _convolution(network, out, weights, f"{prefix}.conv3_1x1", dtype)
    if stride == 1 and int(shortcut.shape[1]) == int(out.shape[1]):
        return graph.add(network, out, shortcut)
    return out


def _mobilevit_block(network, tensor, weights, block: dict[str, object], dtype: np.dtype):
    """Local convolution, global attention over patch positions, then fusion."""
    prefix = str(block["prefix"])
    shortcut = tensor
    height, width = int(tensor.shape[2]), int(tensor.shape[3])
    if height % _PATCH or width % _PATCH:
        raise ValueError(f"MobileViT {prefix} needs a {height}x{width} map divisible by {_PATCH}")

    local = graph.silu(
        network, _convolution(network, tensor, weights, f"{prefix}.conv_kxk", dtype, padding=1)
    )
    local = _convolution(network, local, weights, f"{prefix}.conv_1x1", dtype)

    channels = int(local.shape[1])
    rows, columns = height // _PATCH, width // _PATCH
    patches = rows * columns
    area = _PATCH * _PATCH

    # Unfold. Every token sequence holds one position inside each patch, so the
    # four sequences run independently and a token attends across the whole map.
    tokens = graph.permute(
        network,
        local,
        (channels * rows, _PATCH, columns, _PATCH),
        (0, 2, 1, 3),
        (1, channels, patches, area),
    )
    tokens = graph.permute(network, tokens, None, (0, 3, 2, 1), (area, patches, channels))

    for layer in range(int(block["depth"])):
        tokens = _transformer_layer(
            network, tokens, weights, f"{prefix}.transformer.{layer}", dtype
        )
    tokens = graph.layer_norm(
        network,
        tokens,
        weights[f"{prefix}.norm.weight"],
        weights[f"{prefix}.norm.bias"],
        epsilon=_LAYER_NORM_EPSILON,
        dtype=dtype,
    )

    # Fold, exactly reversing the unfold.
    folded = graph.permute(
        network,
        tokens,
        (1, area, patches, channels),
        (0, 3, 2, 1),
        (channels * rows, columns, _PATCH, _PATCH),
    )
    folded = graph.permute(network, folded, None, (0, 2, 1, 3), (1, channels, height, width))

    projected = graph.silu(
        network, _convolution(network, folded, weights, f"{prefix}.conv_proj", dtype)
    )
    fused = graph.concatenate_channels(network, [shortcut, projected])
    return graph.silu(
        network, _convolution(network, fused, weights, f"{prefix}.conv_fusion", dtype, padding=1)
    )


def _transformer_layer(network, tokens, weights, prefix: str, dtype: np.dtype):
    """One pre-norm attention and feed-forward pair."""
    sequences, count, width = (int(value) for value in tokens.shape)
    if width % _ATTENTION_HEADS:
        raise ValueError(f"MobileViT {prefix} width {width} is not divisible by the head count")
    head_dim = width // _ATTENTION_HEADS

    normed = graph.layer_norm(
        network,
        tokens,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
        epsilon=_LAYER_NORM_EPSILON,
        dtype=dtype,
    )

    def project(name: str):
        projected = graph.matmul_constant(
            network,
            normed,
            weights[f"{prefix}.{name}.weight"],
            weights[f"{prefix}.{name}.bias"],
            dtype=dtype,
        )
        return graph.permute(
            network,
            projected,
            (sequences, count, _ATTENTION_HEADS, head_dim),
            (0, 2, 1, 3),
            None,
        )

    context = graph.attention(
        network, project("query"), project("key"), project("value"), None, dtype=dtype
    )
    context = graph.permute(network, context, None, (0, 2, 1, 3), (sequences, count, width))
    context = graph.matmul_constant(
        network,
        context,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
        dtype=dtype,
    )
    tokens = graph.add(network, tokens, context)

    normed = graph.layer_norm(
        network,
        tokens,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        epsilon=_LAYER_NORM_EPSILON,
        dtype=dtype,
    )
    hidden = graph.matmul_constant(
        network,
        normed,
        weights[f"{prefix}.mlp.fc1.weight"],
        weights[f"{prefix}.mlp.fc1.bias"],
        dtype=dtype,
    )
    hidden = graph.silu(network, hidden)
    hidden = graph.matmul_constant(
        network,
        hidden,
        weights[f"{prefix}.mlp.fc2.weight"],
        weights[f"{prefix}.mlp.fc2.bias"],
        dtype=dtype,
    )
    return graph.add(network, tokens, hidden)


def _configure_precision(builder_config, precision: str) -> None:
    """Switch off TensorRT's reduced-precision fp32 path for fp32 builds.

    TensorRT runs fp32 convolutions in TF32 by default, keeping ten mantissa
    bits rather than twenty-four. An fp32 build here means fp32.
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
        raise ValueError(f"unsupported timm MobileViT precision: {precision}")
    config = _preprocess_config(raw)
    blocks = _layout(checkpoint)
    weights = _weights(checkpoint, blocks, numpy_dtype)
    if weights["head.fc.weight"].shape[0] != config["num_classes"]:
        raise ValueError("MobileViT classifier dimensions do not match config.json")

    # The stem halves once, then every stage after the first halves again.
    stages = sorted({int(block["stage"]) for block in blocks})
    total_stride = 2 * (2 ** (len(stages) - 1))
    height, width = config["image_height"], config["image_width"]
    if height % total_stride or width % total_stride:
        raise ValueError(f"MobileViT input {height}x{width} must be divisible by {total_stride}")
    if verbose:
        print(
            "[trtmc build] timm_mobilevit: "
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
        raise RuntimeError("TensorRT rejected the MobileViT input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the MobileViT input cast")
        hidden = cast.get_output(0)

    stem = weights["stem.weight"]
    hidden = graph.silu(
        network,
        graph.convolution(
            network,
            hidden,
            stem,
            weights["stem.bias"],
            stride=2,
            padding=int(stem.shape[2]) // 2,
            dtype=numpy_dtype,
        ),
    )

    for block in blocks:
        prefix = str(block["prefix"])
        if block["kind"] == "inverted":
            # Only the block that opens a stage after the first changes scale.
            stride = 2 if (int(block["stage"]) > 0 and prefix.endswith(".0")) else 1
            hidden = _inverted_residual(
                network, hidden, weights, prefix, numpy_dtype, stride=stride
            )
        else:
            hidden = _mobilevit_block(network, hidden, weights, block, numpy_dtype)

    hidden = graph.silu(network, _convolution(network, hidden, weights, "final_conv", numpy_dtype))
    hidden = graph.global_average_pool(
        network, hidden, height // total_stride, width // total_stride
    )
    logits = graph.classifier(
        network, hidden, weights["head.fc.weight"], weights["head.fc.bias"], dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the MobileViT output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm MobileViT engine build failed")
    return bytes(plan), config


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm MobileViT image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_mobilevit does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_mobilevit does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_mobilevit does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_mobilevit does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_mobilevit does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_mobilevit does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_mobilevit does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_mobilevit supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_mobilevit does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_mobilevit does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_mobilevit supports only max_sequence_length=1")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw, Checkpoint.open(model_dir), str(request.precision).lower(), bool(request.verbose)
    )
    writer.set_header(family="timm_mobilevit", task=request.task, backend=request.backend)
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
