# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one YOLOS object-detection engine.

YOLOS is a plain ViT that detects by appending ``num_detection_tokens`` learned
tokens to the patch sequence and reading two MLP heads off them. There is no
FPN, no decoder and no NMS.

Two details are taken from the reference rather than inferred, because either one
is silently wrong-but-plausible if guessed:

* the sequence is ``[cls] + patches + detection`` in that order, so the heads read
  the *last* ``num_detection_tokens`` rows;
* ``mid_position_embeddings[i]`` is added **after** encoder layer ``i`` for
  ``i < num_hidden_layers - 1``, not before it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module
from . import graph as graph_ops

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_tensors(model_dir: Path) -> dict:
    paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors under {model_dir}")
    tensors: dict = {}
    for path in paths:
        with safe_open(str(path), framework="numpy") as reader:
            for key in reader.keys():
                tensors[key] = reader.get_tensor(key)
    return tensors


def _require(tensors: dict, name: str) -> np.ndarray:
    if name not in tensors:
        raise KeyError(f"YOLOS checkpoint is missing {name}")
    return np.asarray(tensors[name], dtype=np.float32)


def _build_network(network, tensors: dict, cfg: dict, work_np, work_trt):
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    layers = cfg["num_hidden_layers"]
    eps = cfg["layer_norm_eps"]
    detect = cfg["num_detection_tokens"]
    patch = cfg["patch_size"]
    tokens = 1 + cfg["num_patches"] + detect

    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, cfg["image_height"], cfg["image_width"])
    )
    # The network is strongly typed, so precision is carried by the tensors
    # rather than a builder flag. The runtime always feeds float32.
    image = pixel_values
    if image.dtype != work_trt:
        image = network.add_cast(image, work_trt).get_output(0)

    # Patch embedding: a stride-p convolution, then flatten to [1, patches, hidden].
    proj_w = _require(tensors, "vit.embeddings.patch_embeddings.projection.weight")
    proj_b = _require(tensors, "vit.embeddings.patch_embeddings.projection.bias")
    conv = network.add_convolution_nd(
        image, hidden, (patch, patch),
        trt.Weights(np.ascontiguousarray(proj_w, dtype=work_np)),
        trt.Weights(np.ascontiguousarray(proj_b, dtype=work_np)),
    )
    conv.stride_nd = (patch, patch)
    flat = network.add_shuffle(conv.get_output(0))
    flat.reshape_dims = (1, hidden, cfg["num_patches"])
    flat.second_transpose = (0, 2, 1)
    patches = flat.get_output(0)

    cls_token = graph_ops.add_constant(
        network, (1, 1, hidden), _require(tensors, "vit.embeddings.cls_token").reshape(1, 1, hidden), dtype=work_np
    )
    detection_tokens = graph_ops.add_constant(
        network, (1, detect, hidden),
        _require(tensors, "vit.embeddings.detection_tokens").reshape(1, detect, hidden),
        dtype=work_np,
    )
    # Order is load-bearing: the heads read the trailing detection rows.
    sequence = network.add_concatenation([cls_token, patches, detection_tokens])
    sequence.axis = 1
    hidden_states = sequence.get_output(0)

    position = _require(tensors, "vit.embeddings.position_embeddings").reshape(1, -1, hidden)
    if position.shape[1] != tokens:
        raise ValueError(
            f"YOLOS position embeddings cover {position.shape[1]} tokens but the "
            f"configured image size needs {tokens}; this family builds the checkpoint's "
            "native resolution only"
        )
    hidden_states = graph_ops.add_sum(
        network, hidden_states, graph_ops.add_constant(network, (1, tokens, hidden), position, dtype=work_np)
    )

    mid_position = None
    if cfg["use_mid_position_embeddings"]:
        mid_position = _require(tensors, "vit.encoder.mid_position_embeddings")
        if mid_position.shape[0] != layers - 1:
            raise ValueError(
                f"YOLOS mid position embeddings cover {mid_position.shape[0]} layers but "
                f"the config declares {layers - 1}"
            )

    scale = 1.0 / math.sqrt(head_dim)
    for index in range(layers):
        prefix = f"vit.encoder.layer.{index}"
        residual = hidden_states
        normed = graph_ops.add_layer_norm(
            network, hidden_states,
            _require(tensors, f"{prefix}.layernorm_before.weight"),
            _require(tensors, f"{prefix}.layernorm_before.bias"), eps, dtype=work_np,
        )
        attention = f"{prefix}.attention.attention"
        query = graph_ops.split_heads(
            network,
            graph_ops.add_linear(network, normed, _require(tensors, f"{attention}.query.weight"),
                                 _require(tensors, f"{attention}.query.bias"), dtype=work_np),
            tokens, heads, head_dim)
        key = graph_ops.split_heads(
            network,
            graph_ops.add_linear(network, normed, _require(tensors, f"{attention}.key.weight"),
                                 _require(tensors, f"{attention}.key.bias"), dtype=work_np),
            tokens, heads, head_dim)
        value = graph_ops.split_heads(
            network,
            graph_ops.add_linear(network, normed, _require(tensors, f"{attention}.value.weight"),
                                 _require(tensors, f"{attention}.value.bias"), dtype=work_np),
            tokens, heads, head_dim)
        context = graph_ops.merge_heads(
            network, graph_ops.add_attention(network, query, key, value, scale), tokens, hidden)
        projected = graph_ops.add_linear(
            network, context,
            _require(tensors, f"{prefix}.attention.output.dense.weight"),
            _require(tensors, f"{prefix}.attention.output.dense.bias"), dtype=work_np)
        hidden_states = graph_ops.add_sum(network, projected, residual)

        residual = hidden_states
        normed = graph_ops.add_layer_norm(
            network, hidden_states,
            _require(tensors, f"{prefix}.layernorm_after.weight"),
            _require(tensors, f"{prefix}.layernorm_after.bias"), eps, dtype=work_np,
        )
        inner = graph_ops.add_gelu(
            network,
            graph_ops.add_linear(network, normed,
                                 _require(tensors, f"{prefix}.intermediate.dense.weight"),
                                 _require(tensors, f"{prefix}.intermediate.dense.bias"), dtype=work_np))
        outer = graph_ops.add_linear(
            network, inner, _require(tensors, f"{prefix}.output.dense.weight"),
            _require(tensors, f"{prefix}.output.dense.bias"), dtype=work_np)
        hidden_states = graph_ops.add_sum(network, outer, residual)

        # After the layer, and never on the last one.
        if mid_position is not None and index < layers - 1:
            hidden_states = graph_ops.add_sum(
                network, hidden_states,
                graph_ops.add_constant(
                    network, (1, tokens, hidden),
                    mid_position[index].reshape(1, tokens, hidden), dtype=work_np))

    hidden_states = graph_ops.add_layer_norm(
        network, hidden_states,
        _require(tensors, "vit.layernorm.weight"),
        _require(tensors, "vit.layernorm.bias"), eps, dtype=work_np,
    )

    trailing = network.add_slice(
        hidden_states, (0, tokens - detect, 0), (1, detect, hidden), (1, 1, 1))
    detected = trailing.get_output(0)

    logits = _mlp_head(network, detected, tensors, "class_labels_classifier", work_np)
    boxes = graph_ops.add_sigmoid(
        network, _mlp_head(network, detected, tensors, "bbox_predictor", work_np))

    # Always publish float32 so the runtime contract does not vary with precision.
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    if boxes.dtype != trt.float32:
        boxes = network.add_cast(boxes, trt.float32).get_output(0)
    logits.name = "logits"
    boxes.name = "pred_boxes"
    network.mark_output(logits)
    network.mark_output(boxes)


def _mlp_head(network, inp, tensors: dict, prefix: str, work_np):
    """Three linear layers with ReLU between them, matching YolosMLPPredictionHead."""
    depth = 0
    while f"{prefix}.layers.{depth}.weight" in tensors:
        depth += 1
    if depth == 0:
        raise KeyError(f"YOLOS checkpoint is missing {prefix}.layers.0.weight")
    x = inp
    for index in range(depth):
        x = graph_ops.add_linear(
            network, x,
            _require(tensors, f"{prefix}.layers.{index}.weight"),
            _require(tensors, f"{prefix}.layers.{index}.bias"), dtype=work_np)
        if index < depth - 1:
            x = graph_ops.add_relu(network, x)
    return x


def build_engine(tensors: dict, cfg: dict, *, precision: str, verbose: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if precision == "fp16":
        work_np, work_trt = np.float16, trt.float16
    else:
        work_np, work_trt = np.float32, trt.float32
    _build_network(network, tensors, cfg, work_np, work_trt)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("YOLOS TensorRT engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one YOLOS object-detection bundle."""
    if request.task != "object_detection":
        raise ValueError("yolos supports only task=object_detection")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("yolos supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("yolos does not support dynamic_kv_cache")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("yolos supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("yolos does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("yolos does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("yolos does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("yolos does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("yolos does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("yolos does not support mixed-precision layers")

    model_dir = Path(request.model_dir)
    model_config = config_module.ModelConfig.from_dir(model_dir)
    if model_config.model_type.lower() != "yolos":
        raise ValueError(f"yolos does not support model_type={model_config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported yolos precision: {precision}")

    cfg = config_module.resolve(model_config.raw)
    if request.image_height is not None and int(request.image_height) != cfg["image_height"]:
        raise NotImplementedError(
            "yolos builds the checkpoint's native image height only; its position "
            "embeddings are not interpolated")
    if request.image_width is not None and int(request.image_width) != cfg["image_width"]:
        raise NotImplementedError(
            "yolos builds the checkpoint's native image width only; its position "
            "embeddings are not interpolated")

    tensors = _load_tensors(model_dir)
    plan = build_engine(tensors, cfg, precision=precision, verbose=bool(request.verbose))

    writer.set_header(family="yolos", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": cfg["image_height"],
            "input_image_w": cfg["image_width"],
            "image_mean": list(_IMAGENET_MEAN),
            "image_std": list(_IMAGENET_STD),
            "num_detection_tokens": cfg["num_detection_tokens"],
        },
    )
