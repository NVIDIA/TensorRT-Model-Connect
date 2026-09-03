# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm GhostNet image-classification family plugin.

Supports timm GhostNet classifiers stored in HF Hub format. The initial target
is:
  timm/ghostnet_100.in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

The whole layout is recovered from the checkpoint, including the stride: a
GhostNet bottleneck downsamples exactly when it carries a `conv_dw` depthwise
convolution, so no architecture table is needed. The squeeze-excite gate and the
projection shortcut are likewise detected from the keys.
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
        "num_features": int(raw.get("num_features", 1280)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
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

    leaves: dict[tuple[int, int], set[str]] = {}
    pattern = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".")[0])
    if not leaves:
        raise ValueError("Checkpoint has no blocks.<stage>.<index> entries")

    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("Block stage indices are not contiguous")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        for index in indices:
            present = leaves[(stage, index)]
            if "conv" in present:
                kind = "conv_bn_act"
            elif "ghost1" in present:
                kind = "ghost_bottleneck"
            else:
                raise ValueError(
                    f"blocks.{stage}.{index} is neither a Ghost bottleneck nor a "
                    "convolution block")
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    # A bottleneck downsamples exactly when it has a depthwise
                    # convolution between its two Ghost modules.
                    "stride": 2 if "conv_dw" in present else 1,
                    "has_se": "se" in present,
                    "has_shortcut": "shortcut" in present,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class TimmGhostnetPlugin:
    name = "timm_ghostnet"
    runtime_strategy = "timm_ghostnet_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_ghostnet":
            return True
        return mt.startswith("ghostnet")

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
        raw["_timm_ghostnet_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def bn(prefix: str) -> None:
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        def ghost(prefix: str) -> None:
            """A Ghost module is a pointwise branch plus a cheap depthwise one."""
            conv(f"{prefix}.primary_conv.0.weight")
            bn(f"{prefix}.primary_conv.1")
            conv(f"{prefix}.cheap_operation.0.weight")
            bn(f"{prefix}.cheap_operation.1")

        conv("conv_stem.weight")
        bn("bn1")

        for block in layout["blocks"]:
            prefix = block["prefix"]
            if block["kind"] == "conv_bn_act":
                conv(f"{prefix}.conv.weight")
                bn(f"{prefix}.bn1")
                continue
            ghost(f"{prefix}.ghost1")
            if block["stride"] != 1:
                conv(f"{prefix}.conv_dw.weight")
                bn(f"{prefix}.bn_dw")
            if block["has_se"]:
                for leaf in ("se.conv_reduce", "se.conv_expand"):
                    conv(f"{prefix}.{leaf}.weight")
                    weights[f"{prefix}.{leaf}.bias"] = _load_tensor(
                        readers, f"{prefix}.{leaf}.bias").astype(target_dtype)
            ghost(f"{prefix}.ghost2")
            if block["has_shortcut"]:
                # depthwise, norm, pointwise, norm
                conv(f"{prefix}.shortcut.0.weight")
                bn(f"{prefix}.shortcut.1")
                conv(f"{prefix}.shortcut.2.weight")
                bn(f"{prefix}.shortcut.3")

        for key in ("conv_head.weight", "conv_head.bias",
                    "classifier.weight", "classifier.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _bn(self, network, x, weights, prefix, dtype):
        return graph_ops.add_batch_norm(
            network, x,
            weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"], weights[f"{prefix}.running_var"],
            _BN_EPS, dtype=dtype)

    def _ghost(self, network, x, weights, prefix, dtype, *, relu: bool):
        """Ghost module: pointwise branch, cheap depthwise branch, concatenated."""
        pw = weights[f"{prefix}.primary_conv.0.weight"]
        primary = graph_ops.add_conv2d(
            network, x, pw, None, int(pw.shape[0]), (1, 1), dtype=dtype)
        primary = self._bn(network, primary, weights, f"{prefix}.primary_conv.1", dtype)
        if relu:
            primary = graph_ops.add_relu(network, primary)

        dw = weights[f"{prefix}.cheap_operation.0.weight"]
        k = int(dw.shape[2])
        cheap = graph_ops.add_conv2d(
            network, primary, dw, None, int(dw.shape[0]), (k, k),
            padding=(k // 2, k // 2), groups=int(dw.shape[0]), dtype=dtype)
        cheap = self._bn(network, cheap, weights, f"{prefix}.cheap_operation.1", dtype)
        if relu:
            cheap = graph_ops.add_relu(network, cheap)

        return graph_ops.add_concat(network, [primary, cheap])

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
            raise NotImplementedError("timm_ghostnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_ghostnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_ghostnet precision: {precision}")

        cfg = config.raw.get("_timm_ghostnet_config")
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
                f"timm_ghostnet input {image_h}x{image_w} must be divisible by {total_stride}")

        if verbose:
            print(
                "[trtmc build] timm_ghostnet: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, "
                f"se_blocks={sum(1 for b in blocks if b['has_se'])}, "
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

        stem_w = weights["conv_stem.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, stem_w, None, int(stem_w.shape[0]), (3, 3),
            stride=(2, 2), padding=(1, 1), dtype=work_np_dtype)
        hidden = self._bn(network, hidden, weights, "bn1", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        cur_h, cur_w = image_h // 2, image_w // 2
        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]

            if block["kind"] == "conv_bn_act":
                w = weights[f"{prefix}.conv.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, w, None, int(w.shape[0]), (1, 1), dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = graph_ops.add_relu(network, hidden)
                continue

            identity = hidden
            # The first Ghost module activates; the second does not.
            out = self._ghost(
                network, hidden, weights, f"{prefix}.ghost1", work_np_dtype, relu=True)

            if stride != 1:
                dw = weights[f"{prefix}.conv_dw.weight"]
                k = int(dw.shape[2])
                out = graph_ops.add_conv2d(
                    network, out, dw, None, int(dw.shape[0]), (k, k),
                    stride=(stride, stride), padding=(k // 2, k // 2),
                    groups=int(dw.shape[0]), dtype=work_np_dtype)
                out = self._bn(network, out, weights, f"{prefix}.bn_dw", work_np_dtype)
                cur_h, cur_w = cur_h // stride, cur_w // stride

            if block["has_se"]:
                out = graph_ops.add_squeeze_excite(
                    network, out, (cur_h, cur_w),
                    weights[f"{prefix}.se.conv_reduce.weight"],
                    weights[f"{prefix}.se.conv_reduce.bias"],
                    weights[f"{prefix}.se.conv_expand.weight"],
                    weights[f"{prefix}.se.conv_expand.bias"],
                    dtype=work_np_dtype)

            out = self._ghost(
                network, out, weights, f"{prefix}.ghost2", work_np_dtype, relu=False)

            if block["has_shortcut"]:
                sdw = weights[f"{prefix}.shortcut.0.weight"]
                k = int(sdw.shape[2])
                identity = graph_ops.add_conv2d(
                    network, identity, sdw, None, int(sdw.shape[0]), (k, k),
                    stride=(stride, stride), padding=(k // 2, k // 2),
                    groups=int(sdw.shape[0]), dtype=work_np_dtype)
                identity = self._bn(
                    network, identity, weights, f"{prefix}.shortcut.1", work_np_dtype)
                spw = weights[f"{prefix}.shortcut.2.weight"]
                identity = graph_ops.add_conv2d(
                    network, identity, spw, None, int(spw.shape[0]), (1, 1),
                    dtype=work_np_dtype)
                identity = self._bn(
                    network, identity, weights, f"{prefix}.shortcut.3", work_np_dtype)

            hidden = graph_ops.add_sum(network, out, identity)

        # GhostNet pools before the head convolution, like MobileNetV3.
        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))
        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, head_w, weights["conv_head.bias"],
            int(head_w.shape[0]), (1, 1), dtype=work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        cls_w = weights["classifier.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(cls_w.shape[1]), num_classes,
            cls_w, weights["classifier.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_ghostnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_ghostnet_config") or _resolve_config(config.raw)
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


plugin = TimmGhostnetPlugin()
