# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm XCiT classifiers.

XCiT transposes self-attention. Instead of every token attending to every other
token, it normalises the query and key along the *token* axis and multiplies
them the other way round, producing a channel-by-channel attention whose cost
grows linearly in the number of tokens rather than quadratically. A learned
temperature scales that map before the softmax.

Because channel attention mixes nothing spatially, each block also carries a
local patch interaction: the tokens are folded back into a feature map, run
through two depthwise convolutions, and unfolded again. Two class-attention
blocks then read the whole sequence into a single class token, which is what
the classifier sees.

The layout comes from the checkpoint: the transformer depth from the `blocks`
keys, the class-attention depth from `cls_attn_blocks`, the head count from the
shape of each block's `temperature`, and the patch stem depth from how many
convolutions it holds.
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
# XCiT builds its layer norms with 1e-6, not the 1e-5 the other families use.
_LAYER_NORM_EPSILON = 1e-6
# The Fourier position encoding's own constants.
_POSITION_TEMPERATURE = 10000.0
_POSITION_EPSILON = 1e-6
_BLOCK = re.compile(r"^blocks\.(\d+)\.")
_CLASS_BLOCK = re.compile(r"^cls_attn_blocks\.(\d+)\.")
_STEM = re.compile(r"^patch_embed\.proj\.(\d+)\.0\.weight$")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"XCiT model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("XCiT config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("xcit"):
        raise ValueError(f"unsupported timm XCiT model identity: {identity!r}")
    return value


def tokens_norm_of(architecture: str) -> bool:
    """Whether the second class-attention norm covers every token.

    The published configs set this False only for the nano widths and True
    everywhere else, and it is not recorded in a checkpoint: the norm has the
    same weights either way, only its input differs.
    """
    return not architecture.lower().startswith("xcit_nano")


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
        raise ValueError("XCiT pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("XCiT image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("XCiT image std must contain three values")
    result = {
        "image_height": int(height),
        "image_width": int(width),
        "num_classes": int(raw.get("num_classes", source.get("num_classes", 1000))),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "crop_pct": float(source.get("crop_pct", 1.0)),
        "interpolation": str(source.get("interpolation", "bicubic")),
        "crop_mode": str(source.get("crop_mode", "center")),
    }
    if (
        result["image_height"] <= 0
        or result["image_width"] <= 0
        or result["num_classes"] <= 0
        or not 0.0 < result["crop_pct"] <= 1.0
        or any(value == 0.0 for value in result["std"])
        or result["interpolation"] not in {"bilinear", "bicubic"}
    ):
        raise ValueError("XCiT preprocessing or classifier config is invalid")
    if result["crop_mode"] != "center":
        raise NotImplementedError(
            f"XCiT crop_mode {result['crop_mode']!r} is not implemented; every published "
            "checkpoint asks for center"
        )
    return result


def _depth(names: frozenset[str], pattern: re.Pattern[str], label: str) -> int:
    indices = {int(match.group(1)) for match in map(pattern.match, names) if match}
    if not indices:
        raise ValueError(f"XCiT checkpoint has no {label}")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"XCiT {label} indices are not contiguous from 0")
    return len(indices)


def _layout(checkpoint: Checkpoint) -> dict[str, int]:
    names = checkpoint.names
    stem = sorted(int(match.group(1)) for match in map(_STEM.match, names) if match)
    if not stem:
        raise ValueError("XCiT checkpoint has no patch embedding convolutions")
    # The stem alternates convolution and activation, so its convolutions sit
    # at the even indices. Four of them reach patch 16, three reach patch 8.
    if stem != list(range(0, 2 * len(stem), 2)):
        raise ValueError("XCiT patch embedding convolutions are not at the even indices")
    # Validate the block numbering before reading anything out of block zero,
    # so a gap reports the gap rather than a missing tensor.
    depth = _depth(names, _BLOCK, "blocks")
    class_depth = _depth(names, _CLASS_BLOCK, "cls_attn_blocks")
    heads = int(checkpoint.tensor("blocks.0.attn.temperature").shape[0])
    if heads <= 0:
        raise ValueError("XCiT temperature does not name a head count")
    return {
        "depth": depth,
        "class_depth": class_depth,
        "stem_convolutions": len(stem),
        "heads": heads,
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
        raise ValueError(f"XCiT norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"XCiT norm {norm} does not match convolution {conv}")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"XCiT norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def _norm_scale(
    checkpoint: Checkpoint, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a batch norm to a per-channel scale and shift."""
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if np.any(variance + _BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"XCiT norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _BATCH_NORM_EPSILON)
    return scale.astype(dtype), (beta - mean * scale).astype(dtype)


def fourier_position_encoding(
    height: int, width: int, hidden_dim: int, projection: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """The Fourier position encoding, already projected to the token width.

    The engine has one input size, so this whole thing is a constant. It
    mirrors timm's PositionalEncodingFourier for a fully valid map, then
    applies the 1x1 projection on the host rather than in the graph.
    """
    rows = np.repeat(np.arange(1, height + 1, dtype=np.float64)[:, None], width, axis=1)
    columns = np.repeat(np.arange(1, width + 1, dtype=np.float64)[None, :], height, axis=0)
    scale = 2.0 * np.pi
    rows = rows / (rows[-1:, :] + _POSITION_EPSILON) * scale
    columns = columns / (columns[:, -1:] + _POSITION_EPSILON) * scale

    index = np.arange(hidden_dim, dtype=np.float64)
    divisor = _POSITION_TEMPERATURE ** (2.0 * np.floor(index / 2.0) / hidden_dim)
    across = columns[:, :, None] / divisor
    down = rows[:, :, None] / divisor
    across = np.stack((np.sin(across[:, :, 0::2]), np.cos(across[:, :, 1::2])), axis=3).reshape(
        height, width, hidden_dim
    )
    down = np.stack((np.sin(down[:, :, 0::2]), np.cos(down[:, :, 1::2])), axis=3).reshape(
        height, width, hidden_dim
    )
    # The row lanes come first, matching the reference.
    stacked = np.concatenate((down, across), axis=2).transpose(2, 0, 1)[None]

    flat = stacked.reshape(1, stacked.shape[1], height * width)
    matrix = projection.reshape(projection.shape[0], projection.shape[1])
    projected = np.einsum("oc,bcn->bon", matrix, flat) + bias.reshape(1, -1, 1)
    return projected.reshape(1, matrix.shape[0], height, width)


def _weights(
    checkpoint: Checkpoint, layout: dict[str, int], dtype: np.dtype
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for position in range(layout["stem_convolutions"]):
        index = position * 2
        weight, bias = _fold_norm(
            checkpoint,
            f"patch_embed.proj.{index}.0.weight",
            f"patch_embed.proj.{index}.1",
            dtype,
        )
        result[f"stem.{position}.weight"] = weight
        result[f"stem.{position}.bias"] = bias

    result["pos_embed.weight"] = checkpoint.tensor("pos_embed.token_projection.weight")
    result["pos_embed.bias"] = checkpoint.tensor("pos_embed.token_projection.bias")

    for index in range(layout["depth"]):
        source = f"blocks.{index}"
        qkv_weight = checkpoint.tensor(f"{source}.attn.qkv.weight")
        qkv_bias = checkpoint.tensor(f"{source}.attn.qkv.bias")
        width = qkv_weight.shape[0] // 3
        for position, name in enumerate(("query", "key", "value")):
            lo, hi = position * width, (position + 1) * width
            result[f"{source}.{name}.weight"] = qkv_weight[lo:hi].astype(dtype)
            result[f"{source}.{name}.bias"] = qkv_bias[lo:hi].astype(dtype)
        result[f"{source}.temperature"] = checkpoint.tensor(f"{source}.attn.temperature").astype(
            dtype
        )
        for leaf in ("attn.proj", "mlp.fc1", "mlp.fc2"):
            for part in ("weight", "bias"):
                result[f"{source}.{leaf}.{part}"] = checkpoint.tensor(
                    f"{source}.{leaf}.{part}"
                ).astype(dtype)
        for which in ("norm1", "norm2", "norm3"):
            for part in ("weight", "bias"):
                result[f"{source}.{which}.{part}"] = checkpoint.tensor(
                    f"{source}.{which}.{part}"
                ).astype(dtype)
        for which in ("gamma1", "gamma2", "gamma3"):
            result[f"{source}.{which}"] = checkpoint.tensor(f"{source}.{which}").astype(dtype)
        # The local interaction keeps its norm between two convolutions, so it
        # cannot fold into either and stays a channel scale.
        for leaf in ("conv1", "conv2"):
            result[f"{source}.local.{leaf}.weight"] = checkpoint.tensor(
                f"{source}.local_mp.{leaf}.weight"
            ).astype(dtype)
            result[f"{source}.local.{leaf}.bias"] = checkpoint.tensor(
                f"{source}.local_mp.{leaf}.bias"
            ).astype(dtype)
        scale, shift = _norm_scale(checkpoint, f"{source}.local_mp.bn", dtype)
        result[f"{source}.local.bn.scale"] = scale
        result[f"{source}.local.bn.shift"] = shift

    for index in range(layout["class_depth"]):
        source = f"cls_attn_blocks.{index}"
        for leaf in (
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
            "norm1",
            "norm2",
        ):
            for part in ("weight", "bias"):
                result[f"{source}.{leaf}.{part}"] = checkpoint.tensor(
                    f"{source}.{leaf}.{part}"
                ).astype(dtype)
        for which in ("gamma1", "gamma2"):
            result[f"{source}.{which}"] = checkpoint.tensor(f"{source}.{which}").astype(dtype)

    result["cls_token"] = checkpoint.tensor("cls_token").astype(dtype)
    for part in ("weight", "bias"):
        result[f"norm.{part}"] = checkpoint.tensor(f"norm.{part}").astype(dtype)
        result[f"head.{part}"] = checkpoint.tensor(f"head.{part}").astype(dtype)
    return result


def _norm(network, tokens, weights, prefix: str, dtype: np.dtype):
    return graph.layer_norm(
        network,
        tokens,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        epsilon=_LAYER_NORM_EPSILON,
        dtype=dtype,
    )


def _scaled(network, tokens, weights, name: str, dtype: np.dtype):
    """LayerScale: one learned weight per channel."""
    gamma = weights[name]
    values = graph.constant(network, gamma.reshape(1, 1, -1), dtype=dtype, like=tokens)
    return graph.multiply(network, tokens, values)


def _cross_covariance_attention(network, tokens, weights, prefix, heads, dtype):
    """Attention over channels rather than tokens.

    The query and key are normalised along the token axis and multiplied the
    other way round, so the map is channel by channel and a learned temperature
    scales it before the softmax.
    """
    count, width = int(tokens.shape[1]), int(tokens.shape[2])
    head_dim = width // heads

    def project(name: str):
        projected = graph.matmul_constant(
            network,
            tokens,
            weights[f"{prefix}.{name}.weight"],
            weights[f"{prefix}.{name}.bias"],
            dtype=dtype,
        )
        # [1, N, C] -> [1, heads, head_dim, N]: the token axis goes last.
        return graph.permute(network, projected, (1, count, heads, head_dim), (0, 2, 3, 1), None)

    query = graph.l2_normalize(network, project("query"), axis=3, dtype=dtype)
    key = graph.l2_normalize(network, project("key"), axis=3, dtype=dtype)
    value = project("value")

    temperature = graph.constant(
        network, weights[f"{prefix}.temperature"].reshape(1, heads, 1, 1), dtype=dtype, like=query
    )
    scores = graph.multiply(
        network, graph.matmul(network, query, key, transpose_right=True), temperature
    )
    attended = graph.matmul(network, graph.softmax(network, scores, axis=3), value)
    merged = graph.permute(network, attended, None, (0, 3, 1, 2), (1, count, width))
    return graph.matmul_constant(
        network,
        merged,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
        dtype=dtype,
    )


def _local_interaction(network, tokens, weights, prefix, rows, columns, dtype):
    """Two depthwise convolutions over the tokens folded back into a map."""
    count, width = int(tokens.shape[1]), int(tokens.shape[2])
    spatial = graph.permute(network, tokens, None, (0, 2, 1), (1, width, rows, columns))
    first = weights[f"{prefix}.local.conv1.weight"]
    spatial = graph.convolution(
        network,
        spatial,
        first,
        weights[f"{prefix}.local.conv1.bias"],
        padding=int(first.shape[2]) // 2,
        groups=width,
        dtype=dtype,
    )
    spatial = graph.gelu(network, spatial, dtype=dtype)
    spatial = graph.channel_scale(
        network,
        spatial,
        weights[f"{prefix}.local.bn.scale"],
        weights[f"{prefix}.local.bn.shift"],
        dtype=dtype,
    )
    second = weights[f"{prefix}.local.conv2.weight"]
    spatial = graph.convolution(
        network,
        spatial,
        second,
        weights[f"{prefix}.local.conv2.bias"],
        padding=int(second.shape[2]) // 2,
        groups=width,
        dtype=dtype,
    )
    return graph.permute(network, spatial, (1, width, count), (0, 2, 1), (1, count, width))


def _feed_forward(network, tokens, weights, prefix, dtype):
    hidden = graph.matmul_constant(
        network,
        tokens,
        weights[f"{prefix}.mlp.fc1.weight"],
        weights[f"{prefix}.mlp.fc1.bias"],
        dtype=dtype,
    )
    hidden = graph.gelu(network, hidden, dtype=dtype)
    return graph.matmul_constant(
        network,
        hidden,
        weights[f"{prefix}.mlp.fc2.weight"],
        weights[f"{prefix}.mlp.fc2.bias"],
        dtype=dtype,
    )


def _block(network, tokens, weights, prefix, heads, rows, columns, dtype):
    """One XCiT block.

    The residual order is attention, then the local interaction, then the feed
    forward. timm keeps gamma3 on the local branch and gamma2 on the feed
    forward, which reads out of order but matches the published weights.
    """
    attended = _cross_covariance_attention(
        network,
        _norm(network, tokens, weights, f"{prefix}.norm1", dtype),
        weights,
        prefix,
        heads,
        dtype,
    )
    tokens = graph.add(
        network, tokens, _scaled(network, attended, weights, f"{prefix}.gamma1", dtype)
    )

    local = _local_interaction(
        network,
        _norm(network, tokens, weights, f"{prefix}.norm3", dtype),
        weights,
        prefix,
        rows,
        columns,
        dtype,
    )
    tokens = graph.add(network, tokens, _scaled(network, local, weights, f"{prefix}.gamma3", dtype))

    forwarded = _feed_forward(
        network, _norm(network, tokens, weights, f"{prefix}.norm2", dtype), weights, prefix, dtype
    )
    return graph.add(
        network, tokens, _scaled(network, forwarded, weights, f"{prefix}.gamma2", dtype)
    )


def _class_attention(network, tokens, weights, prefix, heads, dtype):
    """Only the class token asks a question; every token answers."""
    count, width = int(tokens.shape[1]), int(tokens.shape[2])
    head_dim = width // heads

    query_source = graph.slice_tokens(network, tokens, 0, 1)
    query = graph.matmul_constant(
        network,
        query_source,
        weights[f"{prefix}.attn.q.weight"],
        weights[f"{prefix}.attn.q.bias"],
        dtype=dtype,
    )
    query = graph.permute(network, query, (1, 1, heads, head_dim), (0, 2, 1, 3), None)

    def project(name: str):
        projected = graph.matmul_constant(
            network,
            tokens,
            weights[f"{prefix}.attn.{name}.weight"],
            weights[f"{prefix}.attn.{name}.bias"],
            dtype=dtype,
        )
        return graph.permute(network, projected, (1, count, heads, head_dim), (0, 2, 1, 3), None)

    attended = graph.attention(network, query, project("k"), project("v"), None, dtype=dtype)
    merged = graph.permute(network, attended, None, (0, 2, 1, 3), (1, 1, width))
    return graph.matmul_constant(
        network,
        merged,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
        dtype=dtype,
    )


def _class_block(network, tokens, weights, prefix, heads, dtype, *, tokens_norm: bool):
    """One class-attention block.

    Only the class token is rewritten. The rest of the sequence is carried
    through from the normalised tensor, not the original one, which is easy to
    get wrong because the two are interchangeable everywhere else.
    """
    count = int(tokens.shape[1])
    normed = _norm(network, tokens, weights, f"{prefix}.norm1", dtype)
    attended = _class_attention(network, normed, weights, prefix, heads, dtype)
    carried = graph.slice_tokens(network, normed, 1, count - 1)
    joined = graph.concatenate_tokens(network, [attended, carried])
    tokens = graph.add(
        network, tokens, _scaled(network, joined, weights, f"{prefix}.gamma1", dtype)
    )

    if tokens_norm:
        tokens = _norm(network, tokens, weights, f"{prefix}.norm2", dtype)
    else:
        head = _norm(
            network, graph.slice_tokens(network, tokens, 0, 1), weights, f"{prefix}.norm2", dtype
        )
        tokens = graph.concatenate_tokens(
            network, [head, graph.slice_tokens(network, tokens, 1, count - 1)]
        )

    cls_token = graph.slice_tokens(network, tokens, 0, 1)
    forwarded = _scaled(
        network,
        _feed_forward(network, cls_token, weights, prefix, dtype),
        weights,
        f"{prefix}.gamma2",
        dtype,
    )
    # The reference adds the whole sequence back, so every token except the
    # class one ends up doubled. A later layer norm makes that invisible in the
    # logits, but the tensor is what the reference produces and is kept.
    rest = graph.slice_tokens(network, tokens, 1, count - 1)
    return graph.add(network, tokens, graph.concatenate_tokens(network, [forwarded, rest]))


def _configure_precision(builder_config, precision: str) -> None:
    """Switch off TensorRT's reduced-precision fp32 path for fp32 builds."""
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
        raise ValueError(f"unsupported timm XCiT precision: {precision}")
    config = _preprocess_config(raw)
    layout = _layout(checkpoint)
    weights = _weights(checkpoint, layout, numpy_dtype)
    architecture = str(raw.get("architecture") or raw.get("model_type") or "")
    tokens_norm = tokens_norm_of(architecture)

    patch = 1 << layout["stem_convolutions"]
    height, width = config["image_height"], config["image_width"]
    if height % patch or width % patch:
        raise ValueError(f"XCiT input {height}x{width} must be divisible by {patch}")
    rows, columns = height // patch, width // patch
    count = rows * columns
    embed = int(weights["head.weight"].shape[1])
    if weights["head.weight"].shape[0] != config["num_classes"]:
        raise ValueError("XCiT classifier dimensions do not match config.json")
    if verbose:
        print(
            "[trtmc build] timm_xcit: "
            f"image={height}x{width}, patch={patch}, tokens={count}, depth={layout['depth']}, "
            f"class_depth={layout['class_depth']}, heads={layout['heads']}, "
            f"tokens_norm={tokens_norm}, precision={precision}",
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
        raise RuntimeError("TensorRT rejected the XCiT input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the XCiT input cast")
        hidden = cast.get_output(0)

    for position in range(layout["stem_convolutions"]):
        if position:
            hidden = graph.gelu(network, hidden, dtype=numpy_dtype)
        hidden = graph.convolution(
            network,
            hidden,
            weights[f"stem.{position}.weight"],
            weights[f"stem.{position}.bias"],
            stride=2,
            padding=1,
            dtype=numpy_dtype,
        )

    # The position encoding is a constant for one input size, so it is built
    # on the host and added to the tokens as a map before they are flattened.
    encoding = fourier_position_encoding(
        rows,
        columns,
        weights["pos_embed.weight"].shape[1] // 2,
        weights["pos_embed.weight"],
        weights["pos_embed.bias"],
    ).astype(numpy_dtype)
    hidden = graph.add(
        network, hidden, graph.constant(network, encoding, dtype=numpy_dtype, like=hidden)
    )
    tokens = graph.permute(network, hidden, (1, embed, count), (0, 2, 1), (1, count, embed))

    for index in range(layout["depth"]):
        tokens = _block(
            network, tokens, weights, f"blocks.{index}", layout["heads"], rows, columns, numpy_dtype
        )

    cls_token = graph.constant(
        network, weights["cls_token"].reshape(1, 1, embed), dtype=numpy_dtype, like=tokens
    )
    tokens = graph.concatenate_tokens(network, [cls_token, tokens])
    for index in range(layout["class_depth"]):
        tokens = _class_block(
            network,
            tokens,
            weights,
            f"cls_attn_blocks.{index}",
            layout["heads"],
            numpy_dtype,
            tokens_norm=tokens_norm,
        )

    tokens = _norm(network, tokens, weights, "norm", numpy_dtype)
    logits = graph.classifier(
        network,
        graph.slice_tokens(network, tokens, 0, 1),
        weights["head.weight"],
        weights["head.bias"],
        dtype=numpy_dtype,
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the XCiT output cast")
        logits = cast.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm XCiT engine build failed")
    return bytes(plan), config


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm XCiT image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_xcit does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_xcit does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_xcit does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_xcit does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_xcit does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_xcit does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_xcit does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_xcit supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_xcit does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_xcit does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_xcit supports only max_sequence_length=1")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw, Checkpoint.open(model_dir), str(request.precision).lower(), bool(request.verbose)
    )
    writer.set_header(family="timm_xcit", task=request.task, backend=request.backend)
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
