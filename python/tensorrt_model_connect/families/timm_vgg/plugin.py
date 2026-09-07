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

import numpy as np
from tensorrt_model_connect import trt_compat

from .model import model as graph_ops
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig


trt = trt_compat.get_trt()

# timm's VGG stacks 3x3 convolutions with padding 1 and halves with 2x2 max
# pooling; both are fixed by the architecture rather than recorded in config.
_CONV_KERNEL = (3, 3)
_CONV_PADDING = (1, 1)
_POOL_KERNEL = 2
_POOL_STRIDE = 2


def _pretrained_cfg(raw: dict) -> dict:
    """timm nests preprocessing under pretrained_cfg; older exports inline it."""
    nested = raw.get("pretrained_cfg")
    return nested if isinstance(nested, dict) else raw


def _resolve_vgg_config(raw: dict) -> dict:
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
        "num_features": int(raw.get("num_features", 4096)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bilinear")),
    }


def _discover_layout(readers) -> dict:
    """Recover the conv/pool sequence from the features.<index> keys.

    timm keeps the torchvision Sequential indices, so a convolution is followed
    by a ReLU at index+1. When the next convolution is more than two indices
    away, the gap is a max pool.
    """
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

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


class TimmVggPlugin:
    name = "timm_vgg"
    runtime_strategy = "timm_vgg_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_vgg":
            return True
        # timm config.json has no model_type; ModelConfig falls back to the
        # "architecture" field, e.g. "vgg16" or "vgg16_bn".
        return mt.startswith("vgg")

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
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
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

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
            raise NotImplementedError("timm_vgg does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_vgg does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_vgg precision: {precision}")

        cfg = config.raw.get("_timm_vgg_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        layers = cfg["layers"]
        num_pools = cfg["num_pools"]

        divisor = 1 << num_pools
        if image_h % divisor != 0 or image_w % divisor != 0:
            raise ValueError(
                f"timm_vgg input {image_h}x{image_w} must be divisible by {divisor}")
        feat_h, feat_w = image_h // divisor, image_w // divisor

        if verbose:
            print(
                "[trtmc build] timm_vgg: "
                f"image={image_h}x{image_w}, convs={len(cfg['conv_indices'])}, "
                f"pools={num_pools}, classes={num_classes}, precision={precision}",
                file=sys.stderr,
            )

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            trt_compat.network_creation_flags(
                strongly_typed=True,
                explicit_batch=True,
            )
        )
        trt_config = builder.create_builder_config()
        trt_config.avg_timing_iterations = 8
        trt_config.max_aux_streams = 0
        trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        pixel_values = network.add_input(
            "pixel_values", trt.float32, (1, 3, image_h, image_w))
        hidden = pixel_values
        if hidden.dtype != work_trt_dtype:
            hidden = network.add_cast(hidden, work_trt_dtype).get_output(0)

        for kind, index in layers:
            if kind == "pool":
                hidden = graph_ops.add_max_pool2d(
                    network, hidden, _POOL_KERNEL, _POOL_STRIDE, 0)
                continue
            w = weights[f"features.{index}.weight"]
            hidden = graph_ops.add_conv2d(
                network, hidden, w, weights[f"features.{index}.bias"],
                int(w.shape[0]), _CONV_KERNEL,
                padding=_CONV_PADDING, dtype=work_np_dtype)
            hidden = graph_ops.add_relu(network, hidden)

        # timm implements the VGG head as convolutions: fc1 is a feat_h x feat_w
        # convolution and fc2 is 1x1, so the head runs on the feature map.
        fc1_w = weights["pre_logits.fc1.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, fc1_w, weights["pre_logits.fc1.bias"],
            int(fc1_w.shape[0]), (int(fc1_w.shape[2]), int(fc1_w.shape[3])),
            dtype=work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        fc2_w = weights["pre_logits.fc2.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, fc2_w, weights["pre_logits.fc2.bias"],
            int(fc2_w.shape[0]), (1, 1), dtype=work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        head_w = weights["head.fc.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(head_w.shape[1]), num_classes,
            head_w, weights["head.fc.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        del feat_h, feat_w
        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_vgg engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_vgg_config") or _resolve_vgg_config(config.raw)
        return {
            "model_type": config.model_type,
            "runtime_strategy": self.runtime_strategy,
            "input_image_h": cfg["image_size_h"],
            "input_image_w": cfg["image_size_w"],
            "num_classes": cfg["num_classes"],
            "image_mean": cfg["mean"],
            "image_std": cfg["std"],
            "crop_pct": cfg["crop_pct"],
            "interpolation": cfg["interpolation"],
        }


plugin = TimmVggPlugin()
