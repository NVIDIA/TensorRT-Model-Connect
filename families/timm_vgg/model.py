# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm VGG image-classification family plugin.

Supports timm VGG classifiers stored in HF Hub format. The initial target is:
  timm/vgg16.tv_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

timm stores VGG as a flat `features.<index>` Sequential, so the convolution and
pooling layout is recovered from the checkpoint key indices rather than from a
per-depth table. That covers vgg11/13/16/19 from one code path.
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
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

# timm's VGG stacks 3x3 convolutions with padding 1 and halves with 2x2 max
# pooling; both are fixed by the architecture rather than recorded in config.
_CONV_KERNEL = (3, 3)
_CONV_PADDING = (1, 1)
_POOL_KERNEL = 2
_POOL_STRIDE = 2
_SUPPORTED_MODEL_TYPES = frozenset({"timm_vgg", "vgg11", "vgg13", "vgg16", "vgg19"})


def _resolve_vgg_config(raw: dict) -> dict:
    pcfg = raw.get("pretrained_cfg")
    if not isinstance(pcfg, dict):
        raise ValueError("timm VGG config requires pretrained_cfg")
    required = ("input_size", "mean", "std", "crop_pct", "crop_mode", "interpolation")
    missing = [f"pretrained_cfg.{key}" for key in required if key not in pcfg]
    if "num_classes" not in raw:
        missing.append("num_classes")
    if missing:
        raise ValueError(f"timm VGG config is missing required fields: {missing}")
    input_size = pcfg["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 3 or int(input_size[0]) != 3:
        raise ValueError("timm VGG pretrained_cfg.input_size must be [3, height, width]")
    image_h, image_w = int(input_size[1]), int(input_size[2])
    if not isinstance(pcfg["mean"], list) or not isinstance(pcfg["std"], list):
        raise ValueError("timm VGG mean/std must be lists")
    mean = [float(value) for value in pcfg["mean"]]
    std = [float(value) for value in pcfg["std"]]
    crop_pct = float(pcfg["crop_pct"])
    interpolation = str(pcfg["interpolation"])
    num_classes = int(raw["num_classes"])
    if image_h <= 0 or image_w <= 0 or num_classes <= 0:
        raise ValueError("timm VGG image dimensions and num_classes must be positive")
    if len(mean) != 3 or len(std) != 3 or any(value == 0.0 for value in std):
        raise ValueError("timm VGG mean/std must contain three channels with non-zero std")
    if not 0.0 < crop_pct <= 1.0 or pcfg["crop_mode"] != "center":
        raise ValueError("timm VGG requires a center crop with crop_pct in (0, 1]")
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("timm VGG supports only bilinear or bicubic interpolation")
    return {
        "image_size_h": image_h,
        "image_size_w": image_w,
        "num_classes": num_classes,
        "mean": mean,
        "std": std,
        "crop_pct": crop_pct,
        "interpolation": interpolation,
    }


def _discover_layout(readers) -> dict:
    """Recover the conv/pool sequence from the features.<index> keys.

    timm keeps the torchvision Sequential indices, so a convolution is followed
    by a ReLU at index+1. When the next convolution is more than two indices
    away, the gap is a max pool.
    """
    names = set(readers.tensor_map)

    pattern = re.compile(r"^features\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for m in map(pattern.match, names) if m)
    if not indices:
        raise ValueError("Checkpoint has no features.<index>.weight convolutions")

    layers: list[tuple[str, int]] = []
    for position, index in enumerate(indices):
        layers.append(("conv", index))
        if position + 1 < len(indices) and indices[position + 1] - index > 2:
            layers.append(("pool", -1))
    # VGG always closes the feature stack with a pool before the head.
    layers.append(("pool", -1))

    pools = sum(1 for kind, _ in layers if kind == "pool")
    return {"layers": layers, "num_pools": pools, "conv_indices": indices}


class _TimmVggModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str,
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        raw = config.raw
        vgg_cfg = _resolve_vgg_config(raw)
        layout = _discover_layout(readers)
        vgg_cfg.update(layout)
        raw["_timm_vgg_config"] = vgg_cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        for index in layout["conv_indices"]:
            for suffix in ("weight", "bias"):
                key = f"features.{index}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(target_dtype)

        for key in (
            "pre_logits.fc1.weight",
            "pre_logits.fc1.bias",
            "pre_logits.fc2.weight",
            "pre_logits.fc2.bias",
            "head.fc.weight",
            "head.fc.bias",
        ):
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str,
        verbose: bool = False,
    ) -> bytes:
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_vgg precision: {precision}")

        cfg = config.raw.get("_timm_vgg_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        layers = cfg["layers"]
        num_pools = cfg["num_pools"]

        divisor = 1 << num_pools
        if image_h % divisor != 0 or image_w % divisor != 0:
            raise ValueError(f"timm_vgg input {image_h}x{image_w} must be divisible by {divisor}")
        if verbose:
            print(
                "[trtmc build] timm_vgg: "
                f"image={image_h}x{image_w}, convs={len(cfg['conv_indices'])}, "
                f"pools={num_pools}, classes={num_classes}, precision={precision}",
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

        for kind, index in layers:
            if kind == "pool":
                hidden = graph_ops.add_max_pool2d(network, hidden, _POOL_KERNEL, _POOL_STRIDE, 0)
                continue
            w = weights[f"features.{index}.weight"]
            hidden = graph_ops.add_conv2d(
                network,
                hidden,
                w,
                weights[f"features.{index}.bias"],
                int(w.shape[0]),
                _CONV_KERNEL,
                padding=_CONV_PADDING,
                dtype=work_np_dtype,
            )
            hidden = graph_ops.add_relu(network, hidden)

        # timm implements the VGG head as convolutions: fc1 is a feat_h x feat_w
        # convolution and fc2 is 1x1, so the head runs on the feature map.
        fc1_w = weights["pre_logits.fc1.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            fc1_w,
            weights["pre_logits.fc1.bias"],
            int(fc1_w.shape[0]),
            (int(fc1_w.shape[2]), int(fc1_w.shape[3])),
            dtype=work_np_dtype,
        )
        hidden = graph_ops.add_relu(network, hidden)

        fc2_w = weights["pre_logits.fc2.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            fc2_w,
            weights["pre_logits.fc2.bias"],
            int(fc2_w.shape[0]),
            (1, 1),
            dtype=work_np_dtype,
        )
        hidden = graph_ops.add_relu(network, hidden)

        head_w = weights["head.fc.weight"]
        logits = graph_ops.add_fc(
            network,
            hidden,
            int(head_w.shape[1]),
            num_classes,
            head_w,
            weights["head.fc.bias"],
            dtype=work_np_dtype,
        )
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_vgg engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_vgg_config") or _resolve_vgg_config(config.raw)
        return {
            "input_image_h": cfg["image_size_h"],
            "input_image_w": cfg["image_size_w"],
            "num_classes": cfg["num_classes"],
            "image_mean": cfg["mean"],
            "image_std": cfg["std"],
            "crop_pct": cfg["crop_pct"],
            "interpolation": cfg["interpolation"],
        }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm VGG image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_vgg does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("timm_vgg does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_vgg does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_vgg does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_vgg does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_vgg does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_vgg does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_vgg supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_vgg does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_vgg does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_vgg supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    model_type = str(config.model_type).lower()
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(f"timm VGG does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    model = _TimmVggModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="timm_vgg", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    runtime_source = model.get_bundle_config_overrides(config)
    writer.add_json(
        "runtime.json",
        {
            key: runtime_source[key]
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
