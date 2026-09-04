# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm DenseNet image-classification family model.

Supports timm DenseNet classifiers stored in HF Hub format. The initial target
is:
  timm/densenet121.ra_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

The layout is fully recovered from the checkpoint: the number of dense blocks
and the number of layers in each come from the `features.denseblockN.denselayerM`
keys, and the growth rate follows from the layer convolution shapes. Nothing
here needs an architecture table, so densenet121/161/169/201 all build from one
code path.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from .graph import model as graph_ops
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

_BN_EPS = 1e-5


def _pretrained_cfg(raw: dict) -> dict:
    nested = raw.get("pretrained_cfg")
    return nested if isinstance(nested, dict) else raw


def _resolve_config(raw: dict) -> dict:
    pcfg = _pretrained_cfg(raw)
    input_size = pcfg.get("input_size", [3, 224, 224])
    if isinstance(input_size, int):
        image_h = image_w = int(input_size)
    else:
        image_h, image_w = int(input_size[-2]), int(input_size[-1])
    return {
        "image_size_h": image_h,
        "image_size_w": image_w,
        "num_classes": int(raw.get("num_classes", pcfg.get("num_classes", 1000))),
        "num_features": int(raw.get("num_features", 1024)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _discover_layout(readers) -> dict:
    """Recover the dense block and transition structure from the keys."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    pattern = re.compile(r"^features\.denseblock(\d+)\.denselayer(\d+)\.")
    layers: dict[int, set[int]] = {}
    for name in names:
        match = pattern.match(name)
        if match:
            block = int(match.group(1))
            layers.setdefault(block, set()).add(int(match.group(2)))
    if not layers:
        raise ValueError("Checkpoint has no features.denseblockN.denselayerM entries")

    blocks = sorted(layers)
    if blocks != list(range(1, len(blocks) + 1)):
        raise ValueError("Dense block indices are not contiguous from 1")

    block_layers = []
    for block in blocks:
        indices = sorted(layers[block])
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"denseblock{block} layer indices are not contiguous from 1")
        block_layers.append(len(indices))

    # A transition sits between consecutive dense blocks, never after the last.
    transitions = [
        index
        for index in range(1, len(blocks))
        if any(n.startswith(f"features.transition{index}.") for n in names)
    ]
    if transitions != list(range(1, len(blocks))):
        raise ValueError("Transition layers do not sit between every dense block")

    return {"block_layers": block_layers, "num_blocks": len(blocks)}


class _TimmDensenetModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        raw = config.raw
        cfg = _resolve_config(raw)
        layout = _discover_layout(readers)
        cfg.update(layout)
        raw["_timm_densenet_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def bn(prefix: str) -> None:
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        conv("features.conv0.weight")
        bn("features.norm0")

        for block_index, count in enumerate(layout["block_layers"], start=1):
            for layer_index in range(1, count + 1):
                prefix = f"features.denseblock{block_index}.denselayer{layer_index}"
                bn(f"{prefix}.norm1")
                conv(f"{prefix}.conv1.weight")
                bn(f"{prefix}.norm2")
                conv(f"{prefix}.conv2.weight")
            if block_index < layout["num_blocks"]:
                bn(f"features.transition{block_index}.norm")
                conv(f"features.transition{block_index}.conv.weight")

        bn("features.norm5")
        for key in ("classifier.weight", "classifier.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _bn(self, network, x, weights, prefix, dtype):
        return graph_ops.add_batch_norm(
            network,
            x,
            weights[f"{prefix}.weight"],
            weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"],
            weights[f"{prefix}.running_var"],
            _BN_EPS,
            dtype=dtype,
        )

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        parallel_config=None,
    ) -> bytes:
        del max_cache_length
        if quant_ctx is not None:
            raise NotImplementedError("timm_densenet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_densenet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_densenet precision: {precision}")

        cfg = config.raw.get("_timm_densenet_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        block_layers = cfg["block_layers"]

        # Stem halves twice, then one halving per transition.
        divisor = 4 * (1 << (len(block_layers) - 1))
        if image_h % divisor != 0 or image_w % divisor != 0:
            raise ValueError(
                f"timm_densenet input {image_h}x{image_w} must be divisible by {divisor}"
            )
        feat_h, feat_w = image_h // divisor, image_w // divisor

        if verbose:
            print(
                "[trtmc build] timm_densenet: "
                f"image={image_h}x{image_w}, blocks={block_layers}, "
                f"classes={num_classes}, precision={precision}",
                file=sys.stderr,
            )

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.avg_timing_iterations = 8
        trt_config.max_aux_streams = 0
        trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_h, image_w))
        hidden = pixel_values
        if hidden.dtype != work_trt_dtype:
            hidden = network.add_cast(hidden, work_trt_dtype).get_output(0)

        stem_w = weights["features.conv0.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            stem_w,
            None,
            int(stem_w.shape[0]),
            (7, 7),
            stride=(2, 2),
            padding=(3, 3),
            dtype=work_np_dtype,
        )
        hidden = self._bn(network, hidden, weights, "features.norm0", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = graph_ops.add_max_pool2d(network, hidden, 3, 2, 1)

        for block_index, count in enumerate(block_layers, start=1):
            # Each layer sees every earlier output, so keep the running stack
            # and concatenate once per layer.
            stack = [hidden]
            for layer_index in range(1, count + 1):
                prefix = f"features.denseblock{block_index}.denselayer{layer_index}"
                inputs = stack[0] if len(stack) == 1 else graph_ops.add_concat(network, stack)

                # DenseNet is pre-activation: norm and ReLU come before the conv.
                out = self._bn(network, inputs, weights, f"{prefix}.norm1", work_np_dtype)
                out = graph_ops.add_relu(network, out)
                w1 = weights[f"{prefix}.conv1.weight"]
                out = graph_ops.add_conv2d(
                    network, out, w1, None, int(w1.shape[0]), (1, 1), dtype=work_np_dtype
                )

                out = self._bn(network, out, weights, f"{prefix}.norm2", work_np_dtype)
                out = graph_ops.add_relu(network, out)
                w2 = weights[f"{prefix}.conv2.weight"]
                out = graph_ops.add_conv2d(
                    network,
                    out,
                    w2,
                    None,
                    int(w2.shape[0]),
                    (3, 3),
                    padding=(1, 1),
                    dtype=work_np_dtype,
                )
                stack.append(out)

            hidden = graph_ops.add_concat(network, stack)

            if block_index < len(block_layers):
                t = f"features.transition{block_index}"
                hidden = self._bn(network, hidden, weights, f"{t}.norm", work_np_dtype)
                hidden = graph_ops.add_relu(network, hidden)
                tw = weights[f"{t}.conv.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, tw, None, int(tw.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = graph_ops.add_avg_pool2d(network, hidden, 2, 2)

        hidden = self._bn(network, hidden, weights, "features.norm5", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = graph_ops.add_global_avg_pool(network, hidden, (feat_h, feat_w))

        cls_w = weights["classifier.weight"]
        logits = graph_ops.add_fc(
            network,
            hidden,
            int(cls_w.shape[1]),
            num_classes,
            cls_w,
            weights["classifier.bias"],
            dtype=work_np_dtype,
        )
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_densenet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_densenet_config") or _resolve_config(config.raw)
        return {
            "model_type": config.model_type,
            "input_image_h": cfg["image_size_h"],
            "input_image_w": cfg["image_size_w"],
            "num_classes": cfg["num_classes"],
            "image_mean": cfg["mean"],
            "image_std": cfg["std"],
            "crop_pct": cfg["crop_pct"],
            "interpolation": cfg["interpolation"],
        }


_QUALIFIED_ARCHITECTURES = {"densenet121", "densenet161", "densenet169", "densenet201"}


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm DenseNet image-classification bundle."""
    if request.image_height is not None:
        raise NotImplementedError("timm_densenet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_densenet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_densenet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_densenet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_densenet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_densenet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_densenet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_densenet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_densenet does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_densenet supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    architecture = str(config.raw.get("architecture") or config.model_type).lower()
    if architecture not in _QUALIFIED_ARCHITECTURES:
        raise ValueError(f"timm DenseNet does not support architecture={architecture!r}")
    precision = str(request.precision).lower()
    model = _TimmDensenetModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        1,
        precision=precision,
        quant_ctx=None,
        verbose=bool(request.verbose),
        parallel_config=None,
    )
    writer.set_header(family="timm_densenet", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    runtime = model.get_bundle_config_overrides(config)
    writer.add_json(
        "runtime.json",
        {
            key: runtime[key]
            for key in (
                "input_image_h",
                "input_image_w",
                "crop_pct",
                "interpolation",
                "image_mean",
                "image_std",
            )
        },
    )
