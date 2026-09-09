# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for timm CrossViT classifiers.

CrossViT is a convolutional network shaped like a transformer: a large-kernel
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


_LAYER_NORM_EPSILON = 1e-6

# Head count is not recoverable from the weights: every projection is square, so
# the head split leaves no trace in any tensor shape.
_NUM_HEADS = 4
_BLOCK = re.compile(r"^blocks\.(\d+)\.blocks\.(\d+)\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"CrossViT model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CrossViT config.json must contain one object")
    identity = value.get("model_type") or value.get("architecture")
    if identity is None:
        architectures = value.get("architectures")
        identity = architectures[0] if isinstance(architectures, list) and architectures else None
    if not isinstance(identity, str) or not identity.lower().startswith("crossvit"):
        raise ValueError(f"unsupported timm CrossViT model identity: {identity!r}")
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
        raise ValueError("CrossViT pretrained input_size must be [3, height, width]")
    mean = source.get("mean", [0.485, 0.456, 0.406])
    std = source.get("std", [0.229, 0.224, 0.225])
    if not isinstance(mean, list) or len(mean) != 3:
        raise ValueError("CrossViT image mean must contain three values")
    if not isinstance(std, list) or len(std) != 3:
        raise ValueError("CrossViT image std must contain three values")
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
        raise ValueError("CrossViT preprocessing or classifier config is invalid")
    return result


def _layout(checkpoint: Checkpoint) -> dict[str, Any]:
    names = checkpoint.names
    branches = len(
        {
            int(match.group(1))
            for match in (re.match(r"^patch_embed\.(\d+)\.", name) for name in names)
            if match
        }
    )
    if branches < 2:
        raise ValueError("CrossViT needs at least two branches")
    depths: dict[tuple[int, int], set[int]] = {}
    for name in names:
        match = _BLOCK.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            depths.setdefault(key, set()).add(int(match.group(3)))
    if not depths:
        raise ValueError("CrossViT checkpoint has no multi-scale blocks")
    stages = sorted({stage for stage, _ in depths})
    if stages != list(range(len(stages))):
        raise ValueError("CrossViT stage indices are not contiguous")
    result: list[list[int]] = []
    for stage in stages:
        per_branch: list[int] = []
        for branch in range(branches):
            indices = sorted(depths.get((stage, branch), set()))
            if not indices:
                raise ValueError(f"CrossViT blocks.{stage}.blocks.{branch} is empty")
            if indices != list(range(len(indices))):
                raise ValueError(f"CrossViT blocks.{stage}.blocks.{branch} is not contiguous")
            per_branch.append(len(indices))
        result.append(per_branch)
    return {"branches": branches, "depths": result}


def _weights(checkpoint: Checkpoint, layout: dict[str, Any], dtype: np.dtype):
    def take(name: str) -> np.ndarray:
        return checkpoint.tensor(name).astype(dtype)

    result: dict[str, np.ndarray] = {}
    for branch in range(layout["branches"]):
        result[f"patch.{branch}.weight"] = take(f"patch_embed.{branch}.proj.weight")
        result[f"patch.{branch}.bias"] = take(f"patch_embed.{branch}.proj.bias")
        result[f"cls.{branch}"] = take(f"cls_token_{branch}")
        result[f"pos.{branch}"] = take(f"pos_embed_{branch}")
        result[f"norm.{branch}.weight"] = take(f"norm.{branch}.weight")
        result[f"norm.{branch}.bias"] = take(f"norm.{branch}.bias")
        result[f"head.{branch}.weight"] = take(f"head.{branch}.weight")
        result[f"head.{branch}.bias"] = take(f"head.{branch}.bias")
    for stage, per_branch in enumerate(layout["depths"]):
        for branch, depth in enumerate(per_branch):
            for index in range(depth):
                prefix = f"blocks.{stage}.blocks.{branch}.{index}"
                for leaf in ("norm1", "norm2", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                    result[f"{prefix}.{leaf}.weight"] = take(f"{prefix}.{leaf}.weight")
                    result[f"{prefix}.{leaf}.bias"] = take(f"{prefix}.{leaf}.bias")
        for branch in range(layout["branches"]):
            for kind in ("projs", "revert_projs"):
                base = f"blocks.{stage}.{kind}.{branch}"
                for leaf in ("0", "2"):
                    result[f"{base}.{leaf}.weight"] = take(f"{base}.{leaf}.weight")
                    result[f"{base}.{leaf}.bias"] = take(f"{base}.{leaf}.bias")
            fusion = f"blocks.{stage}.fusion.{branch}"
            result[f"{fusion}.norm1.weight"] = take(f"{fusion}.norm1.weight")
            result[f"{fusion}.norm1.bias"] = take(f"{fusion}.norm1.bias")
            for leaf in ("wq", "wk", "wv", "proj"):
                result[f"{fusion}.{leaf}.weight"] = take(f"{fusion}.attn.{leaf}.weight")
                result[f"{fusion}.{leaf}.bias"] = take(f"{fusion}.attn.{leaf}.bias")
    if result["head.0.weight"].ndim != 2:
        raise ValueError("CrossViT classifier weights have incompatible shapes")
    return result


def _linear(network, tensor, weights, prefix: str, dtype):
    return graph.matmul_constant(
        network, tensor, weights[f"{prefix}.weight"], weights.get(f"{prefix}.bias"), dtype=dtype
    )


def _norm(network, tensor, weights, prefix: str, dtype):
    return graph.layer_norm(
        network, tensor, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
        epsilon=_LAYER_NORM_EPSILON, dtype=dtype,
    )


def _split_heads(network, tensor, tokens: int, width: int):
    return graph.permute(
        network, tensor, (1, tokens, _NUM_HEADS, width // _NUM_HEADS), (0, 2, 1, 3), None
    )


def _merge_heads(network, tensor, tokens: int, width: int):
    return graph.permute(network, tensor, None, (0, 2, 1, 3), (1, tokens, width))


def _self_attention(network, tensor, weights, prefix: str, dtype, *, tokens: int, width: int):
    head_dim = width // _NUM_HEADS
    qkv = _linear(network, tensor, weights, f"{prefix}.attn.qkv", dtype)
    qkv = graph.permute(
        network, qkv, (1, tokens, 3, _NUM_HEADS, head_dim), (2, 0, 3, 1, 4), None
    )
    parts = []
    for which in range(3):
        piece = graph.slice_tokens(network, qkv, which, 1, axis=0)
        parts.append(graph.reshape(network, piece, (1, _NUM_HEADS, tokens, head_dim)))
    context = graph.attention(network, parts[0], parts[1], parts[2], None, dtype=dtype)
    context = _merge_heads(network, context, tokens, width)
    return _linear(network, context, weights, f"{prefix}.attn.proj", dtype)


def _block(network, tensor, weights, prefix: str, dtype, *, tokens: int, width: int):
    hidden = _norm(network, tensor, weights, f"{prefix}.norm1", dtype)
    hidden = _self_attention(
        network, hidden, weights, prefix, dtype, tokens=tokens, width=width
    )
    tensor = graph.add(network, tensor, hidden)
    hidden = _norm(network, tensor, weights, f"{prefix}.norm2", dtype)
    hidden = _linear(network, hidden, weights, f"{prefix}.mlp.fc1", dtype)
    hidden = graph.gelu(network, hidden, dtype=dtype)
    hidden = _linear(network, hidden, weights, f"{prefix}.mlp.fc2", dtype)
    return graph.add(network, tensor, hidden)


def _projection(network, tensor, weights, prefix: str, dtype):
    """LayerNorm, GELU, Linear: the class-token bridge between branches."""
    hidden = _norm(network, tensor, weights, f"{prefix}.0", dtype)
    hidden = graph.gelu(network, hidden, dtype=dtype)
    return _linear(network, hidden, weights, f"{prefix}.2", dtype)


def _cross_attention(network, tensor, weights, prefix: str, dtype, *, tokens: int, width: int):
    """Attention with a single query: the class token attends, nothing else.

    Only the first token produces a query, so the output is one token wide.
    """
    query_token = graph.slice_tokens(network, tensor, 0, 1)
    query = _split_heads(network, _linear(network, query_token, weights, f"{prefix}.wq", dtype), 1, width)
    key = _split_heads(
        network, _linear(network, tensor, weights, f"{prefix}.wk", dtype), tokens, width
    )
    value = _split_heads(
        network, _linear(network, tensor, weights, f"{prefix}.wv", dtype), tokens, width
    )
    context = graph.attention(network, query, key, value, None, dtype=dtype)
    context = _merge_heads(network, context, 1, width)
    return _linear(network, context, weights, f"{prefix}.proj", dtype)


def _fusion(network, tensor, weights, prefix: str, dtype, *, tokens: int, width: int):
    """Cross-attention block whose residual keeps only the class token."""
    hidden = _norm(network, tensor, weights, f"{prefix}.norm1", dtype)
    hidden = _cross_attention(
        network, hidden, weights, prefix, dtype, tokens=tokens, width=width
    )
    return graph.add(network, graph.slice_tokens(network, tensor, 0, 1), hidden)


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
        raise ValueError(f"unsupported timm CrossViT precision: {precision}")
    config = _preprocess_config(raw)
    layout = _layout(checkpoint)
    weights = _weights(checkpoint, layout, numpy_dtype)
    if weights["head.0.weight"].shape[0] != config["num_classes"]:
        raise ValueError("CrossViT classifier dimensions do not match config.json")

    # Each branch's own input size follows from its patch size and its
    # positional table: the table has one entry per patch plus the class token,
    # and the patches tile a square.
    geometry = []
    for branch in range(layout["branches"]):
        patch_weight = weights[f"patch.{branch}.weight"]
        patch = int(patch_weight.shape[2])
        tokens = int(np.asarray(weights[f"pos.{branch}"]).shape[1])
        grid = int(round(np.sqrt(tokens - 1)))
        if grid * grid != tokens - 1:
            raise ValueError(f"CrossViT branch {branch}: patches do not tile a square")
        geometry.append(
            {
                "patch": patch,
                "width": int(patch_weight.shape[0]),
                "tokens": tokens,
                "size": grid * patch,
            }
        )

    height = config["image_height"]
    width = config["image_width"]
    if verbose:
        print(
            "[trtmc build] timm_crossvit: "
            f"image={height}x{width}, "
            f"branches={[(item['size'], item['patch'], item['width']) for item in geometry]}, "
            f"depths={layout['depths']}, classes={config['num_classes']}, "
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

    pixels = network.add_input("pixel_values", trt.float32, (1, 3, height, width))
    if pixels is None:
        raise RuntimeError("TensorRT rejected the CrossViT input")
    image = pixels
    if image.dtype != tensor_dtype:
        cast = network.add_cast(image, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the CrossViT input cast")
        image = cast.get_output(0)

    states = []
    for branch, item in enumerate(geometry):
        branch_image = image
        if item["size"] != height or item["size"] != width:
            # timm rescales rather than crops for these checkpoints.
            branch_image = graph.bicubic_resize(network, image, (item["size"], item["size"]))
        patches = graph.patch_convolution(
            network, branch_image, weights[f"patch.{branch}.weight"],
            weights[f"patch.{branch}.bias"], patch=item["patch"], dtype=numpy_dtype,
        )
        grid = item["size"] // item["patch"]
        patches = graph.permute(
            network, patches, None, (0, 2, 3, 1), (1, grid * grid, item["width"])
        )
        cls_token = graph.constant(
            network,
            np.asarray(weights[f"cls.{branch}"]).reshape(1, 1, item["width"]),
            dtype=numpy_dtype, like=patches,
        )
        tokens = graph.concatenate_tokens(network, [cls_token, patches])
        position = graph.constant(
            network, np.asarray(weights[f"pos.{branch}"]), dtype=numpy_dtype, like=tokens
        )
        states.append(graph.add(network, tokens, position))

    for stage, per_branch in enumerate(layout["depths"]):
        encoded = []
        for branch, depth in enumerate(per_branch):
            hidden = states[branch]
            for index in range(depth):
                hidden = _block(
                    network, hidden, weights, f"blocks.{stage}.blocks.{branch}.{index}",
                    numpy_dtype,
                    tokens=geometry[branch]["tokens"], width=geometry[branch]["width"],
                )
            encoded.append(hidden)

        fused = []
        for branch in range(layout["branches"]):
            other = (branch + 1) % layout["branches"]
            other_width = geometry[other]["width"]
            # This branch's class token, projected into the other branch's width
            # and placed at the front of the other branch's patches.
            own_cls = graph.slice_tokens(network, encoded[branch], 0, 1)
            projected = _projection(
                network, own_cls, weights, f"blocks.{stage}.projs.{branch}", numpy_dtype
            )
            other_patches = graph.slice_tokens(
                network, encoded[other], 1, geometry[other]["tokens"] - 1
            )
            merged = graph.concatenate_tokens(network, [projected, other_patches])
            attended = _fusion(
                network, merged, weights, f"blocks.{stage}.fusion.{branch}", numpy_dtype,
                tokens=geometry[other]["tokens"], width=other_width,
            )
            reverted = _projection(
                network, attended, weights, f"blocks.{stage}.revert_projs.{branch}", numpy_dtype
            )
            own_patches = graph.slice_tokens(
                network, encoded[branch], 1, geometry[branch]["tokens"] - 1
            )
            fused.append(graph.concatenate_tokens(network, [reverted, own_patches]))
        states = fused

    # Each branch classifies from its own class token; the logits are averaged,
    # so neither branch alone decides the answer.
    branch_logits = []
    for branch in range(layout["branches"]):
        hidden = _norm(network, states[branch], weights, f"norm.{branch}", numpy_dtype)
        hidden = graph.slice_tokens(network, hidden, 0, 1)
        hidden = graph.reshape(network, hidden, (1, geometry[branch]["width"]))
        branch_logits.append(_linear(network, hidden, weights, f"head.{branch}", numpy_dtype))

    logits = graph.scaled_sum(
        network, branch_logits, 1.0 / len(branch_logits), dtype=numpy_dtype
    )
    if logits.dtype != trt.float32:
        cast = network.add_cast(logits, trt.float32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the CrossViT output cast")
        logits = cast.get_output(0)
    logits = graph.reshape(network, logits, (1, config["num_classes"]))
    logits.name = "logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT timm CrossViT engine build failed")
    return bytes(plan), config


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm CrossViT image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_crossvit does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_crossvit does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_crossvit does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_crossvit does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_crossvit does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_crossvit does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_crossvit does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_crossvit supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_crossvit does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_crossvit does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    plan, runtime = _build_engine(
        raw,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="timm_crossvit", task=request.task, backend=request.backend)
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
