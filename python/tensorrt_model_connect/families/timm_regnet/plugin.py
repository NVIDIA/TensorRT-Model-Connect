# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm RegNet image-classification family plugin.

Supports timm RegNet classifiers stored in HF Hub format. The initial target is:
  timm/regnety_040.ra3_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

The layout is recovered from the checkpoint: stages and their block counts come
from the `s<stage>.b<block>` keys, the convolution group count from the 3x3
weight shape, and the squeeze-excite gate and downsample path from the keys that
are present. RegNet halves the resolution in the first block of every stage, so
the stride follows a uniform rule rather than a per-model table.
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
        "num_features": int(raw.get("num_features", 1088)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _discover_layout(readers) -> dict:
    """Recover the stage and block structure from the s<N>.b<M> keys."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    leaves: dict[tuple[int, int], set[str]] = {}
    pattern = re.compile(r"^s(\d+)\.b(\d+)\.(.+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".")[0])
    if not leaves:
        raise ValueError("Checkpoint has no s<stage>.b<block> entries")

    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(1, len(stages) + 1)):
        raise ValueError("Stage indices are not contiguous from 1")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"Stage {stage} block indices are not contiguous from 1")
        for index in indices:
            present = leaves[(stage, index)]
            blocks.append(
                {
                    "prefix": f"s{stage}.b{index}",
                    # RegNet halves the resolution at the head of every stage.
                    "stride": 2 if index == 1 else 1,
                    "has_se": "se" in present,
                    "has_downsample": "downsample" in present,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class TimmRegnetPlugin:
    name = "timm_regnet"
    runtime_strategy = "timm_regnet_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_regnet":
            return True
        return mt.startswith("regnet")

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
        raw["_timm_regnet_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv_bn(prefix: str) -> None:
            """timm wraps each convolution and its norm in a ConvNormAct."""
            weights[f"{prefix}.conv.weight"] = _load_tensor(
                readers, f"{prefix}.conv.weight").astype(target_dtype)
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.bn.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        conv_bn("stem")

        for block in layout["blocks"]:
            prefix = block["prefix"]
            for leaf in ("conv1", "conv2", "conv3"):
                conv_bn(f"{prefix}.{leaf}")
            if block["has_downsample"]:
                conv_bn(f"{prefix}.downsample")
            if block["has_se"]:
                for leaf in ("se.fc1", "se.fc2"):
                    for suffix in ("weight", "bias"):
                        key = f"{prefix}.{leaf}.{suffix}"
                        weights[key] = _load_tensor(readers, key).astype(target_dtype)

        for key in ("head.fc.weight", "head.fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _conv_bn(self, network, x, weights, prefix, kernel, dtype,
                 *, stride=1, padding=0, groups=1):
        w = weights[f"{prefix}.conv.weight"]
        out = graph_ops.add_conv2d(
            network, x, w, None, int(w.shape[0]), kernel,
            stride=(stride, stride), padding=(padding, padding),
            groups=groups, dtype=dtype)
        return graph_ops.add_batch_norm(
            network, out,
            weights[f"{prefix}.bn.weight"], weights[f"{prefix}.bn.bias"],
            weights[f"{prefix}.bn.running_mean"], weights[f"{prefix}.bn.running_var"],
            _BN_EPS, dtype=dtype)

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
            raise NotImplementedError("timm_regnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_regnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_regnet precision: {precision}")

        cfg = config.raw.get("_timm_regnet_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        total_stride = 2
        for block in blocks:
            total_stride *= block["stride"]
        if image_h % total_stride != 0 or image_w % total_stride != 0:
            raise ValueError(
                f"timm_regnet input {image_h}x{image_w} must be divisible by {total_stride}")

        if verbose:
            print(
                "[trtmc build] timm_regnet: "
                f"image={image_h}x{image_w}, stages={cfg['num_stages']}, "
                f"blocks={len(blocks)}, classes={num_classes}, precision={precision}",
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

        hidden = self._conv_bn(
            network, hidden, weights, "stem", (3, 3), work_np_dtype, stride=2, padding=1)
        hidden = graph_ops.add_relu(network, hidden)

        cur_h, cur_w = image_h // 2, image_w // 2
        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]
            identity = hidden

            out = self._conv_bn(
                network, hidden, weights, f"{prefix}.conv1", (1, 1), work_np_dtype)
            out = graph_ops.add_relu(network, out)

            # The 3x3 is grouped; the group count follows from its input channels.
            w2 = weights[f"{prefix}.conv2.conv.weight"]
            groups = max(1, int(w2.shape[0]) // int(w2.shape[1]))
            out = self._conv_bn(
                network, out, weights, f"{prefix}.conv2", (3, 3), work_np_dtype,
                stride=stride, padding=1, groups=groups)
            out = graph_ops.add_relu(network, out)
            cur_h, cur_w = cur_h // stride, cur_w // stride

            if block["has_se"]:
                out = graph_ops.add_squeeze_excite(
                    network, out, (cur_h, cur_w),
                    weights[f"{prefix}.se.fc1.weight"], weights[f"{prefix}.se.fc1.bias"],
                    weights[f"{prefix}.se.fc2.weight"], weights[f"{prefix}.se.fc2.bias"],
                    dtype=work_np_dtype)

            out = self._conv_bn(
                network, out, weights, f"{prefix}.conv3", (1, 1), work_np_dtype)

            if block["has_downsample"]:
                identity = self._conv_bn(
                    network, identity, weights, f"{prefix}.downsample", (1, 1),
                    work_np_dtype, stride=stride)

            hidden = graph_ops.add_sum(network, out, identity)
            hidden = graph_ops.add_relu(network, hidden)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))

        head_w = weights["head.fc.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(head_w.shape[1]), num_classes,
            head_w, weights["head.fc.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_regnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_regnet_config") or _resolve_config(config.raw)
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


plugin = TimmRegnetPlugin()
