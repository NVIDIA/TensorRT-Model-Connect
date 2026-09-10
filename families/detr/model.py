# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain TensorRT build entrypoint for DETR object detection.

DETR predicts a fixed set of boxes directly. A ResNet backbone produces one
feature map, a transformer encoder attends over its pixels, and a transformer
decoder turns a fixed set of learned queries into one box and one label each.
There is no anchor grid and no non-maximum suppression: the set is the output.

The engine is built for one fixed input size, so the sine position embedding is
a constant and is computed on the host rather than in the graph.

The layout comes from the checkpoint. The backbone stage depths are read from
the `layer<stage>.<block>` keys, the encoder and decoder depths from their
`layers.<index>` keys, the head count from the config, and the class count from
the classifier weight.
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


# DetrFrozenBatchNorm2d hard-codes this; it is not read from the config.
_FROZEN_BATCH_NORM_EPSILON = 1e-5
# torch.nn.LayerNorm's default, which DetrConfig never overrides.
_LAYER_NORM_EPSILON = 1e-5
_POSITION_TEMPERATURE = 10000.0
# The backbone feature map DETR projects from is stride 32.
_BACKBONE_STRIDE = 32
_BACKBONE = "model.backbone.conv_encoder.model"
_BLOCK = re.compile(r"^layer(\d+)\.(\d+)\.")


def _read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"DETR model config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("DETR config.json must contain one object")
    identity = value.get("model_type")
    if identity != "detr":
        raise ValueError(f"unsupported DETR model identity: {identity!r}")
    architectures = value.get("architectures")
    if not isinstance(architectures, list) or "DetrForObjectDetection" not in architectures:
        raise ValueError("DETR config.json must declare DetrForObjectDetection")
    if value.get("dilation"):
        raise NotImplementedError("DETR dilated backbones are not supported")
    if value.get("position_embedding_type", "sine") != "sine":
        raise NotImplementedError("DETR supports only the sine position embedding")
    if value.get("activation_function", "relu") != "relu":
        raise NotImplementedError("DETR supports only the relu activation")
    return value


def _preprocess_config(raw: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    """Resolve the fixed input size and the image normalisation."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    path = model_dir / "preprocessor_config.json"
    if path.is_file():
        source = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(source, dict):
            if isinstance(source.get("image_mean"), list):
                mean = source["image_mean"]
            if isinstance(source.get("image_std"), list):
                std = source["image_std"]
    if len(mean) != 3 or len(std) != 3 or any(float(value) == 0.0 for value in std):
        raise ValueError("DETR image mean and std must each hold three non-zero values")

    height = int(raw.get("trtmc_input_height", 800))
    width = int(raw.get("trtmc_input_width", 800))
    if height <= 0 or width <= 0:
        raise ValueError("DETR input size must be positive")
    if height % _BACKBONE_STRIDE or width % _BACKBONE_STRIDE:
        raise ValueError(f"DETR input {height}x{width} must be divisible by {_BACKBONE_STRIDE}")
    return {
        "image_height": height,
        "image_width": width,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }


def _backbone_layout(checkpoint: Checkpoint) -> list[int]:
    """Blocks per ResNet stage, read from the checkpoint keys."""
    blocks: dict[int, set[int]] = {}
    prefix = f"{_BACKBONE}."
    for name in checkpoint.names:
        if not name.startswith(prefix):
            continue
        match = _BLOCK.match(name[len(prefix) :])
        if match:
            blocks.setdefault(int(match.group(1)), set()).add(int(match.group(2)))
    if not blocks:
        raise ValueError("DETR checkpoint has no backbone layer<stage>.<block> tensors")
    stages = sorted(blocks)
    if stages != list(range(1, len(stages) + 1)):
        raise ValueError("DETR backbone stage indices are not contiguous from 1")
    depths = []
    for stage in stages:
        indices = sorted(blocks[stage])
        if indices != list(range(len(indices))):
            raise ValueError(f"DETR backbone stage {stage} block indices are not contiguous")
        depths.append(len(indices))
    return depths


def _transformer_depth(checkpoint: Checkpoint, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.layers\.(\d+)\.")
    indices = {int(match.group(1)) for match in map(pattern.match, checkpoint.names) if match}
    if not indices:
        raise ValueError(f"DETR checkpoint has no {prefix}.layers tensors")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"DETR {prefix} layer indices are not contiguous from 0")
    return len(indices)


def _fold(
    checkpoint: Checkpoint, conv: str, norm: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a convolution and the frozen norm that follows it into one layer.

    The statistics stay float32 through the division; only the result is cast,
    so a small running variance does not lose precision in fp16.
    """
    weight = checkpoint.tensor(conv)
    gamma = checkpoint.tensor(f"{norm}.weight")
    beta = checkpoint.tensor(f"{norm}.bias")
    mean = checkpoint.tensor(f"{norm}.running_mean")
    variance = checkpoint.tensor(f"{norm}.running_var")
    if not (gamma.shape == beta.shape == mean.shape == variance.shape):
        raise ValueError(f"DETR norm {norm} has mismatched parameter shapes")
    if gamma.shape != (weight.shape[0],):
        raise ValueError(f"DETR norm {norm} does not match its convolution")
    if np.any(variance + _FROZEN_BATCH_NORM_EPSILON <= 0.0):
        raise ValueError(f"DETR norm {norm} has a non-positive running variance")
    scale = gamma / np.sqrt(variance + _FROZEN_BATCH_NORM_EPSILON)
    return (weight * scale.reshape(-1, 1, 1, 1)).astype(dtype), (beta - mean * scale).astype(dtype)


def sine_position_embedding(height: int, width: int, channels: int) -> np.ndarray:
    """The fixed sine position embedding for a fully valid feature map.

    This mirrors DetrSinePositionEmbedding for an all-ones pixel mask. The
    engine has one input size, so the result is a constant and is built here
    instead of in the graph.
    """
    if channels % 2:
        raise ValueError("DETR position embedding needs an even channel count")
    half = channels // 2
    # The mask is all ones, so its cumulative sums are just the 1-based
    # row and column indices.
    y_embed = np.repeat(np.arange(1, height + 1, dtype=np.float64)[:, None], width, axis=1)
    x_embed = np.repeat(np.arange(1, width + 1, dtype=np.float64)[None, :], height, axis=0)
    scale = 2.0 * np.pi
    y_embed = y_embed / (y_embed[-1:, :] + 1e-6) * scale
    x_embed = x_embed / (x_embed[:, -1:] + 1e-6) * scale

    index = np.arange(half, dtype=np.float64)
    dim_t = _POSITION_TEMPERATURE ** (2.0 * np.floor(index / 2.0) / half)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    # Interleave sin over the even lanes with cos over the odd ones.
    pos_x = np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])), axis=3).reshape(
        height, width, half
    )
    pos_y = np.stack((np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])), axis=3).reshape(
        height, width, half
    )
    # DetrSinePositionEmbedding puts the row lanes first.
    return np.concatenate((pos_y, pos_x), axis=2).reshape(1, height * width, channels)


def _bottleneck(
    network, tensor, weights: dict[str, np.ndarray], prefix: str, stride: int, dtype: np.dtype
):
    """One ResNet bottleneck: 1x1, 3x3 at the stage stride, 1x1, then a sum."""
    identity = tensor
    out = graph.convolution(
        network,
        tensor,
        weights[f"{prefix}.conv1.weight"],
        weights[f"{prefix}.conv1.bias"],
        dtype=dtype,
    )
    out = graph.relu(network, out)
    out = graph.convolution(
        network,
        out,
        weights[f"{prefix}.conv2.weight"],
        weights[f"{prefix}.conv2.bias"],
        stride=stride,
        padding=1,
        dtype=dtype,
    )
    out = graph.relu(network, out)
    out = graph.convolution(
        network,
        out,
        weights[f"{prefix}.conv3.weight"],
        weights[f"{prefix}.conv3.bias"],
        dtype=dtype,
    )
    if f"{prefix}.downsample.weight" in weights:
        identity = graph.convolution(
            network,
            identity,
            weights[f"{prefix}.downsample.weight"],
            weights[f"{prefix}.downsample.bias"],
            stride=stride,
            dtype=dtype,
        )
    return graph.relu(network, graph.add(network, out, identity))


def _heads(network, tensor, count: int, head_dim: int):
    """[1, tokens, channels] -> [1, heads, tokens, head_dim]."""
    tokens = int(tensor.shape[1])
    return graph.permute(
        network, tensor, (1, tokens, count, head_dim), (0, 2, 1, 3), (1, count, tokens, head_dim)
    )


def _merge_heads(network, tensor, channels: int):
    """[1, heads, tokens, head_dim] -> [1, tokens, channels]."""
    tokens = int(tensor.shape[2])
    return graph.permute(network, tensor, None, (0, 2, 1, 3), (1, tokens, channels))


def _multi_head_attention(
    network,
    query_source,
    key_source,
    value_source,
    weights: dict[str, np.ndarray],
    prefix: str,
    heads: int,
    dtype: np.dtype,
):
    """DETR attention. Position is added to the query and key, never the value."""
    channels = int(query_source.shape[-1])
    head_dim = channels // heads
    query = graph.linear(
        network,
        query_source,
        weights[f"{prefix}.q_proj.weight"],
        weights[f"{prefix}.q_proj.bias"],
        dtype=dtype,
    )
    key = graph.linear(
        network,
        key_source,
        weights[f"{prefix}.k_proj.weight"],
        weights[f"{prefix}.k_proj.bias"],
        dtype=dtype,
    )
    value = graph.linear(
        network,
        value_source,
        weights[f"{prefix}.v_proj.weight"],
        weights[f"{prefix}.v_proj.bias"],
        dtype=dtype,
    )
    context = graph.attention(
        network,
        _heads(network, query, heads, head_dim),
        _heads(network, key, heads, head_dim),
        _heads(network, value, heads, head_dim),
        dtype=dtype,
    )
    merged = _merge_heads(network, context, channels)
    return graph.linear(
        network,
        merged,
        weights[f"{prefix}.out_proj.weight"],
        weights[f"{prefix}.out_proj.bias"],
        dtype=dtype,
    )


def _feed_forward(network, tensor, weights: dict[str, np.ndarray], prefix: str, dtype: np.dtype):
    hidden = graph.linear(
        network, tensor, weights[f"{prefix}.fc1.weight"], weights[f"{prefix}.fc1.bias"], dtype=dtype
    )
    hidden = graph.relu(network, hidden)
    return graph.linear(
        network, hidden, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"], dtype=dtype
    )


def _norm(network, tensor, weights: dict[str, np.ndarray], prefix: str, dtype: np.dtype):
    return graph.layer_norm(
        network,
        tensor,
        weights[f"{prefix}.weight"],
        weights[f"{prefix}.bias"],
        epsilon=_LAYER_NORM_EPSILON,
        dtype=dtype,
    )


def _weights(
    checkpoint: Checkpoint,
    depths: list[int],
    encoder_layers: int,
    decoder_layers: int,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}

    weight, bias = _fold(checkpoint, f"{_BACKBONE}.conv1.weight", f"{_BACKBONE}.bn1", dtype)
    result["backbone.stem.weight"], result["backbone.stem.bias"] = weight, bias
    for stage, count in enumerate(depths, start=1):
        for index in range(count):
            source = f"{_BACKBONE}.layer{stage}.{index}"
            target = f"backbone.layer{stage}.{index}"
            for leaf in ("conv1", "conv2", "conv3"):
                norm = leaf.replace("conv", "bn")
                weight, bias = _fold(
                    checkpoint, f"{source}.{leaf}.weight", f"{source}.{norm}", dtype
                )
                result[f"{target}.{leaf}.weight"] = weight
                result[f"{target}.{leaf}.bias"] = bias
            if f"{source}.downsample.0.weight" in checkpoint.names:
                weight, bias = _fold(
                    checkpoint, f"{source}.downsample.0.weight", f"{source}.downsample.1", dtype
                )
                result[f"{target}.downsample.weight"] = weight
                result[f"{target}.downsample.bias"] = bias

    def plain(name: str, target: str | None = None) -> None:
        result[target or name] = checkpoint.tensor(name).astype(dtype)

    plain("model.input_projection.weight", "input_projection.weight")
    plain("model.input_projection.bias", "input_projection.bias")
    plain("model.query_position_embeddings.weight", "query_position_embeddings")

    for index in range(encoder_layers):
        source = f"model.encoder.layers.{index}"
        for leaf in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
            "self_attn_layer_norm",
            "final_layer_norm",
        ):
            plain(f"{source}.{leaf}.weight", f"encoder.{index}.{leaf}.weight")
            plain(f"{source}.{leaf}.bias", f"encoder.{index}.{leaf}.bias")

    for index in range(decoder_layers):
        source = f"model.decoder.layers.{index}"
        for leaf in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "encoder_attn.q_proj",
            "encoder_attn.k_proj",
            "encoder_attn.v_proj",
            "encoder_attn.out_proj",
            "fc1",
            "fc2",
            "self_attn_layer_norm",
            "encoder_attn_layer_norm",
            "final_layer_norm",
        ):
            plain(f"{source}.{leaf}.weight", f"decoder.{index}.{leaf}.weight")
            plain(f"{source}.{leaf}.bias", f"decoder.{index}.{leaf}.bias")

    plain("model.decoder.layernorm.weight", "decoder.norm.weight")
    plain("model.decoder.layernorm.bias", "decoder.norm.bias")
    plain("class_labels_classifier.weight", "classifier.weight")
    plain("class_labels_classifier.bias", "classifier.bias")
    for index in range(3):
        plain(f"bbox_predictor.layers.{index}.weight", f"bbox.{index}.weight")
        plain(f"bbox_predictor.layers.{index}.bias", f"bbox.{index}.bias")
    if result["bbox.2.weight"].shape[0] != 4:
        raise ValueError("DETR box head must end in four coordinates")
    return result


def _encoder(network, tensor, position, weights, layers: int, heads: int, dtype: np.dtype):
    for index in range(layers):
        prefix = f"encoder.{index}"
        # Position is added to the query and key, never to the value.
        located = graph.add(network, tensor, position)
        attended = _multi_head_attention(
            network, located, located, tensor, weights, f"{prefix}.self_attn", heads, dtype
        )
        tensor = _norm(
            network,
            graph.add(network, tensor, attended),
            weights,
            f"{prefix}.self_attn_layer_norm",
            dtype,
        )
        forwarded = _feed_forward(network, tensor, weights, prefix, dtype)
        tensor = _norm(
            network,
            graph.add(network, tensor, forwarded),
            weights,
            f"{prefix}.final_layer_norm",
            dtype,
        )
    return tensor


def _decoder(
    network, tensor, queries, memory, position, weights, layers: int, heads: int, dtype: np.dtype
):
    for index in range(layers):
        prefix = f"decoder.{index}"
        located = graph.add(network, tensor, queries)
        attended = _multi_head_attention(
            network, located, located, tensor, weights, f"{prefix}.self_attn", heads, dtype
        )
        tensor = _norm(
            network,
            graph.add(network, tensor, attended),
            weights,
            f"{prefix}.self_attn_layer_norm",
            dtype,
        )
        # The cross-attention key carries the image position, the value does not.
        attended = _multi_head_attention(
            network,
            graph.add(network, tensor, queries),
            graph.add(network, memory, position),
            memory,
            weights,
            f"{prefix}.encoder_attn",
            heads,
            dtype,
        )
        tensor = _norm(
            network,
            graph.add(network, tensor, attended),
            weights,
            f"{prefix}.encoder_attn_layer_norm",
            dtype,
        )
        forwarded = _feed_forward(network, tensor, weights, prefix, dtype)
        tensor = _norm(
            network,
            graph.add(network, tensor, forwarded),
            weights,
            f"{prefix}.final_layer_norm",
            dtype,
        )
    return _norm(network, tensor, weights, "decoder.norm", dtype)


def _decode(network, hidden, weights, num_queries: int, dtype: np.dtype):
    """Turn the decoder output into boxes, scores and labels, ranked by score."""
    logits = graph.linear(
        network, hidden, weights["classifier.weight"], weights["classifier.bias"], dtype=dtype
    )
    probabilities = graph.softmax(network, logits, axis=2)
    classes = int(weights["classifier.weight"].shape[0]) - 1
    # The final logit is the "no object" class and never names a detection.
    probabilities = graph.slice_axis(network, probabilities, 0, classes, axis=2)
    best, label = graph.top_k(network, probabilities, k=1, axis=2)
    scores = graph.reshape(network, best, (1, num_queries))
    labels = graph.reshape(network, label, (1, num_queries))

    boxes = hidden
    for index in range(3):
        boxes = graph.linear(
            network,
            boxes,
            weights[f"bbox.{index}.weight"],
            weights[f"bbox.{index}.bias"],
            dtype=dtype,
        )
        if index < 2:
            boxes = graph.relu(network, boxes)
    boxes = graph.sigmoid(network, boxes)
    boxes = graph.reshape(network, boxes, (num_queries, 4))

    # Centre form to corner form, still normalised to the network input.
    centre = graph.slice_axis(network, boxes, 0, 2, axis=1)
    size = graph.slice_axis(network, boxes, 2, 2, axis=1)
    half = graph.constant(network, np.full((1, 2), 0.5, dtype=dtype), dtype=dtype, like=boxes)
    offset = graph.multiply(network, size, half)
    corners = graph.concatenate(
        network,
        [graph.subtract(network, centre, offset), graph.add(network, centre, offset)],
        axis=1,
    )

    # Rank the fixed set so the runtime can stop at the first weak slot.
    ranked_scores, order = graph.top_k(network, scores, k=num_queries, axis=1)
    index = graph.reshape(network, order, (num_queries,))
    return (
        graph.gather(network, corners, index, axis=0),
        graph.reshape(network, ranked_scores, (num_queries,)),
        graph.gather(network, graph.reshape(network, labels, (num_queries,)), index, axis=0),
    )


def _build_engine(
    raw: dict[str, Any],
    model_dir: Path,
    checkpoint: Checkpoint,
    precision: str,
    verbose: bool,
) -> tuple[bytes, dict[str, Any]]:
    if precision == "fp16":
        numpy_dtype, tensor_dtype = np.float16, trt.float16
    elif precision == "fp32":
        numpy_dtype, tensor_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"unsupported DETR precision: {precision}")

    config = _preprocess_config(raw, model_dir)
    depths = _backbone_layout(checkpoint)
    encoder_layers = _transformer_depth(checkpoint, "model.encoder")
    decoder_layers = _transformer_depth(checkpoint, "model.decoder")
    weights = _weights(checkpoint, depths, encoder_layers, decoder_layers, numpy_dtype)

    channels = int(weights["input_projection.weight"].shape[0])
    num_queries = int(weights["query_position_embeddings"].shape[0])
    encoder_heads = int(raw.get("encoder_attention_heads", 8))
    decoder_heads = int(raw.get("decoder_attention_heads", 8))
    if channels % encoder_heads or channels % decoder_heads:
        raise ValueError("DETR head count must divide the model width")

    height, width = config["image_height"], config["image_width"]
    feature_h, feature_w = height // _BACKBONE_STRIDE, width // _BACKBONE_STRIDE
    tokens = feature_h * feature_w
    config["num_classes"] = int(weights["classifier.weight"].shape[0]) - 1
    config["num_queries"] = num_queries

    if verbose:
        print(
            "[trtmc build] detr: "
            f"image={height}x{width}, feature={feature_h}x{feature_w}, "
            f"backbone={depths}, encoder={encoder_layers}, decoder={decoder_layers}, "
            f"queries={num_queries}, classes={config['num_classes']}, precision={precision}",
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
        raise RuntimeError("TensorRT rejected the DETR input")
    hidden = pixels
    if hidden.dtype != tensor_dtype:
        cast = network.add_cast(hidden, tensor_dtype)
        if cast is None:
            raise RuntimeError("TensorRT rejected the DETR input cast")
        hidden = cast.get_output(0)

    hidden = graph.convolution(
        network,
        hidden,
        weights["backbone.stem.weight"],
        weights["backbone.stem.bias"],
        stride=2,
        padding=3,
        dtype=numpy_dtype,
    )
    hidden = graph.relu(network, hidden)
    hidden = graph.max_pool(network, hidden, 3, 2, 1)
    for stage, count in enumerate(depths, start=1):
        for index in range(count):
            # The stem has already halved twice; every stage after the first
            # halves once more at its head.
            stride = 2 if (stage > 1 and index == 0) else 1
            hidden = _bottleneck(
                network, hidden, weights, f"backbone.layer{stage}.{index}", stride, numpy_dtype
            )

    hidden = graph.convolution(
        network,
        hidden,
        weights["input_projection.weight"],
        weights["input_projection.bias"],
        dtype=numpy_dtype,
    )
    # [1, channels, h, w] -> [1, tokens, channels]
    memory = graph.permute(network, hidden, (1, channels, tokens), (0, 2, 1), (1, tokens, channels))
    position = graph.constant(
        network,
        sine_position_embedding(feature_h, feature_w, channels).astype(numpy_dtype),
        dtype=numpy_dtype,
        like=memory,
    )
    memory = _encoder(
        network, memory, position, weights, encoder_layers, encoder_heads, numpy_dtype
    )

    queries = graph.constant(
        network,
        weights["query_position_embeddings"].reshape(1, num_queries, channels),
        dtype=numpy_dtype,
        like=memory,
    )
    # The decoder starts from zeros; every query is carried by its embedding.
    hidden = graph.constant(
        network,
        np.zeros((1, num_queries, channels), dtype=numpy_dtype),
        dtype=numpy_dtype,
        like=memory,
    )
    hidden = _decoder(
        network,
        hidden,
        queries,
        memory,
        position,
        weights,
        decoder_layers,
        decoder_heads,
        numpy_dtype,
    )

    boxes, scores, classes = _decode(network, hidden, weights, num_queries, numpy_dtype)
    for tensor, name in ((boxes, "boxes"), (scores, "scores")):
        if tensor.dtype != trt.float32:
            cast = network.add_cast(tensor, trt.float32)
            if cast is None:
                raise RuntimeError(f"TensorRT rejected the DETR {name} cast")
            tensor = cast.get_output(0)
        tensor.name = name
        network.mark_output(tensor)
    if classes.dtype != trt.int32:
        cast = network.add_cast(classes, trt.int32)
        if cast is None:
            raise RuntimeError("TensorRT rejected the DETR classes cast")
        classes = cast.get_output(0)
    classes.name = "classes"
    network.mark_output(classes)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT DETR engine build failed")
    return bytes(plan), config


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one DETR object-detection bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("detr does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("detr does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("detr does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("detr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("detr does not support context parallelism")
    if request.task != "object_detection":
        raise ValueError("detr supports only task=object_detection")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("detr does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("detr does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("detr supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    # The engine has one fixed input size; the request may name it.
    if request.image_height is not None:
        raw["trtmc_input_height"] = int(request.image_height)
    if request.image_width is not None:
        raw["trtmc_input_width"] = int(request.image_width)
    plan, runtime = _build_engine(
        raw,
        model_dir,
        Checkpoint.open(model_dir),
        str(request.precision).lower(),
        bool(request.verbose),
    )
    writer.set_header(family="detr", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": runtime["image_height"],
            "input_image_w": runtime["image_width"],
            "image_mean": runtime["mean"],
            "image_std": runtime["std"],
            "num_classes": runtime["num_classes"],
            "num_queries": runtime["num_queries"],
        },
    )
