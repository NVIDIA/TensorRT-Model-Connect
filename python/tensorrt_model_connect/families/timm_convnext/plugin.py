# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ConvNeXt image-classification family plugin.

Supports timm ConvNeXt classifiers stored in HF Hub format. The initial target
is:
  timm/convnext_tiny.in12k_ft_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

ConvNeXt is a convolutional network shaped like a transformer: a large-kernel
depthwise convolution stands in for attention, and it is followed by a
LayerNorm and a two-layer MLP with a single GELU. timm applies that MLP by
transposing to channels-last and back; here it is emitted as two 1x1
convolutions instead, which is the same arithmetic on NCHW and avoids two
transposes per block.
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

# ConvNeXt uses a looser LayerNorm epsilon than the transformer families.
_LN_EPS = 1e-6


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
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.95)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _discover_layout(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    stage_regex = re.compile(r"^stages\.(\d+)\.")
    stages = {int(m.group(1)) for m in map(stage_regex.match, names) if m}
    if not stages:
        raise ValueError("Checkpoint has no stages")
    if sorted(stages) != list(range(len(stages))):
        raise ValueError("Stage indices are not contiguous")

    depths = []
    for stage in range(len(stages)):
        block_regex = re.compile(rf"^stages\.{stage}\.blocks\.(\d+)\.")
        blocks = {int(m.group(1)) for m in map(block_regex.match, names) if m}
        if not blocks:
            raise ValueError(f"Checkpoint has no stages.{stage} blocks")
        if sorted(blocks) != list(range(len(blocks))):
            raise ValueError(f"stages.{stage} block indices are not contiguous")
        depths.append(len(blocks))

    return {"depths": depths}


class TimmConvnextPlugin:
    name = "timm_convnext"
    runtime_strategy = "timm_convnext_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_convnext" or mt.startswith("convnext")

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
        cfg.update(_discover_layout(readers))
        raw["_timm_convnext_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for name in sorted(names):
            weights[name] = _load_tensor(readers, name).astype(target_dtype)

        for key in ("head.fc.weight", "head.fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")

        return weights

    def _conv(self, network, x, weights, prefix, dtype, *, stride=1, depthwise=False):
        w = weights[f"{prefix}.weight"]
        kernel = (int(w.shape[2]), int(w.shape[3]))
        out_channels = int(w.shape[0])
        return graph_ops.add_conv2d(
            network, x, w, weights.get(f"{prefix}.bias"), out_channels, kernel,
            stride=(stride, stride),
            padding=(kernel[0] // 2, kernel[1] // 2) if stride == 1 else (0, 0),
            groups=out_channels if depthwise else 1, dtype=dtype)

    def _norm(self, network, x, weights, prefix, channels, dtype):
        return graph_ops.add_layer_norm_channels(
            network, x, channels,
            np.asarray(weights[f"{prefix}.weight"], dtype=np.float32),
            np.asarray(weights[f"{prefix}.bias"], dtype=np.float32),
            _LN_EPS, dtype=dtype)

    def _linear_as_conv(self, network, x, weights, prefix, dtype):
        """A Linear over the channel axis, emitted as a 1x1 convolution."""
        w = np.asarray(weights[f"{prefix}.weight"])
        out_features, in_features = int(w.shape[0]), int(w.shape[1])
        return graph_ops.add_conv2d(
            network, x, w.reshape(out_features, in_features, 1, 1),
            weights[f"{prefix}.bias"], out_features, (1, 1), dtype=dtype)

    def _block(self, network, x, weights, prefix, channels, dtype):
        shortcut = x
        out = self._conv(network, x, weights, f"{prefix}.conv_dw", dtype, depthwise=True)
        out = self._norm(network, out, weights, f"{prefix}.norm", channels, dtype)
        out = self._linear_as_conv(network, out, weights, f"{prefix}.mlp.fc1", dtype)
        out = graph_ops.add_gelu_erf(network, out, dtype=dtype)
        out = self._linear_as_conv(network, out, weights, f"{prefix}.mlp.fc2", dtype)
        if f"{prefix}.gamma" in weights:
            # Layer scale: a learned per-channel gain on the residual branch.
            out = graph_ops.add_channel_scale(
                network, out, channels, weights[f"{prefix}.gamma"], dtype=dtype)
        return graph_ops.add_sum(network, out, shortcut)

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
            raise NotImplementedError("timm_convnext does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_convnext does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_convnext precision: {precision}")

        cfg = config.raw.get("_timm_convnext_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        depths = cfg["depths"]

        if verbose:
            print(
                "[trtmc build] timm_convnext: "
                f"image={image_h}x{image_w}, depths={depths}, "
                f"classes={num_classes}, precision={precision}",
                file=sys.stderr,
            )

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True))
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

        # Stem: a non-overlapping patch convolution, then a norm.
        stem_w = weights["stem.0.weight"]
        channels = int(stem_w.shape[0])
        patch = int(stem_w.shape[2])
        hidden = self._conv(network, hidden, weights, "stem.0", work_np_dtype, stride=patch)
        hidden = self._norm(network, hidden, weights, "stem.1", channels, work_np_dtype)

        for stage, depth in enumerate(depths):
            if f"stages.{stage}.downsample.1.weight" in weights:
                # The norm comes before the strided convolution here, unlike
                # every other downsample in this repo.
                hidden = self._norm(
                    network, hidden, weights, f"stages.{stage}.downsample.0",
                    channels, work_np_dtype)
                stride = int(weights[f"stages.{stage}.downsample.1.weight"].shape[2])
                hidden = self._conv(
                    network, hidden, weights, f"stages.{stage}.downsample.1",
                    work_np_dtype, stride=stride)
                channels = int(weights[f"stages.{stage}.downsample.1.weight"].shape[0])
            for block in range(depth):
                hidden = self._block(
                    network, hidden, weights, f"stages.{stage}.blocks.{block}",
                    channels, work_np_dtype)

        # The head pools first and normalises after, so the norm sees one
        # value per channel rather than the whole feature map.
        hidden = graph_ops.add_mean_spatial(network, hidden)
        hidden = self._norm(network, hidden, weights, "head.norm", channels, work_np_dtype)

        fc_w = weights["head.fc.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["head.fc.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_convnext engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_convnext_config") or _resolve_config(config.raw)
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


plugin = TimmConvnextPlugin()
