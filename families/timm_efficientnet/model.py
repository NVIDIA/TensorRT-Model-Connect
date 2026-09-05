# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm EfficientNet image-classification family model.

Supports timm EfficientNet classifiers stored in HF Hub format. The initial
target is:
  timm/efficientnet_b0.ra_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Block shape is recovered from the checkpoint: the block kind follows from which
convolutions are present and the depthwise kernel from its weight shape. The
activation is uniform SiLU, so unlike MobileNetV3 only the per-stage stride
comes from an architecture table.
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

_BN_EPS = 1e-5  # PyTorch default; the tf_efficientnet_* ports use 1e-3 instead.

# Per-stage stride for the 7-stage EfficientNet layout; the stride applies to
# the first block of each stage. The B0..B7 compound-scaled variants share this
# schedule and differ only in width and depth, which come from the checkpoint.
_STRIDES_7_STAGE = (1, 2, 2, 2, 1, 2, 1)

_STRIDE_SCHEDULES = {7: _STRIDES_7_STAGE}


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
    strides = _STRIDE_SCHEDULES.get(len(stages))
    if strides is None:
        raise ValueError(f"No EfficientNet stride schedule for {len(stages)} stages")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        for index in indices:
            present = leaves[(stage, index)]
            if "conv" in present:
                kind = "conv_bn_act"
            elif "conv_pwl" in present:
                kind = "inverted_residual"
            else:
                kind = "depthwise_separable"
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    "stride": strides[stage] if index == 0 else 1,
                    "has_se": "se" in present,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class _TimmEfficientnetModel:
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
        raw["_timm_efficientnet_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def bn(prefix: str) -> None:
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        conv("conv_stem.weight")
        bn("bn1")

        for block in layout["blocks"]:
            prefix = block["prefix"]
            if block["kind"] == "conv_bn_act":
                conv(f"{prefix}.conv.weight")
                bn(f"{prefix}.bn1")
            elif block["kind"] == "depthwise_separable":
                conv(f"{prefix}.conv_dw.weight")
                bn(f"{prefix}.bn1")
                conv(f"{prefix}.conv_pw.weight")
                bn(f"{prefix}.bn2")
            else:
                conv(f"{prefix}.conv_pw.weight")
                bn(f"{prefix}.bn1")
                conv(f"{prefix}.conv_dw.weight")
                bn(f"{prefix}.bn2")
                conv(f"{prefix}.conv_pwl.weight")
                bn(f"{prefix}.bn3")
            if block["has_se"]:
                for leaf in ("se.conv_reduce", "se.conv_expand"):
                    conv(f"{prefix}.{leaf}.weight")
                    weights[f"{prefix}.{leaf}.bias"] = _load_tensor(
                        readers, f"{prefix}.{leaf}.bias"
                    ).astype(target_dtype)

        conv("conv_head.weight")
        bn("bn2")
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
            raise NotImplementedError("timm_efficientnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_efficientnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_efficientnet precision: {precision}")

        cfg = config.raw.get("_timm_efficientnet_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        total_stride = 2
        for block in blocks:
            total_stride *= block["stride"]
        if image_h % total_stride != 0 or image_w % total_stride != 0:
            raise ValueError(
                f"timm_efficientnet input {image_h}x{image_w} must be divisible by {total_stride}"
            )

        if verbose:
            print(
                "[trtmc build] timm_efficientnet: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, "
                f"se_blocks={sum(1 for b in blocks if b['has_se'])}, "
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

        stem_w = weights["conv_stem.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            stem_w,
            None,
            int(stem_w.shape[0]),
            (3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=work_np_dtype,
        )
        hidden = self._bn(network, hidden, weights, "bn1", work_np_dtype)
        hidden = graph_ops.add_silu(network, hidden)

        cur_h, cur_w = image_h // 2, image_w // 2
        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]
            identity = hidden

            if block["kind"] == "conv_bn_act":
                w = weights[f"{prefix}.conv.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, w, None, int(w.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = graph_ops.add_silu(network, hidden)
                continue

            if block["kind"] == "depthwise_separable":
                dw = weights[f"{prefix}.conv_dw.weight"]
                k = int(dw.shape[2])
                hidden = graph_ops.add_conv2d(
                    network,
                    hidden,
                    dw,
                    None,
                    int(dw.shape[0]),
                    (k, k),
                    stride=(stride, stride),
                    padding=(k // 2, k // 2),
                    groups=int(dw.shape[0]),
                    dtype=work_np_dtype,
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = graph_ops.add_silu(network, hidden)
                cur_h, cur_w = cur_h // stride, cur_w // stride
                if block["has_se"]:
                    hidden = graph_ops.add_squeeze_excite(
                        network,
                        hidden,
                        (cur_h, cur_w),
                        weights[f"{prefix}.se.conv_reduce.weight"],
                        weights[f"{prefix}.se.conv_reduce.bias"],
                        weights[f"{prefix}.se.conv_expand.weight"],
                        weights[f"{prefix}.se.conv_expand.bias"],
                        dtype=work_np_dtype,
                    )
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn2", work_np_dtype)
            else:
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = graph_ops.add_silu(network, hidden)

                dw = weights[f"{prefix}.conv_dw.weight"]
                k = int(dw.shape[2])
                hidden = graph_ops.add_conv2d(
                    network,
                    hidden,
                    dw,
                    None,
                    int(dw.shape[0]),
                    (k, k),
                    stride=(stride, stride),
                    padding=(k // 2, k // 2),
                    groups=int(dw.shape[0]),
                    dtype=work_np_dtype,
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn2", work_np_dtype)
                hidden = graph_ops.add_silu(network, hidden)
                cur_h, cur_w = cur_h // stride, cur_w // stride

                if block["has_se"]:
                    hidden = graph_ops.add_squeeze_excite(
                        network,
                        hidden,
                        (cur_h, cur_w),
                        weights[f"{prefix}.se.conv_reduce.weight"],
                        weights[f"{prefix}.se.conv_reduce.bias"],
                        weights[f"{prefix}.se.conv_expand.weight"],
                        weights[f"{prefix}.se.conv_expand.bias"],
                        dtype=work_np_dtype,
                    )

                pwl = weights[f"{prefix}.conv_pwl.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pwl, None, int(pwl.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn3", work_np_dtype)

            in_ch = int(identity.shape[1])
            out_ch = int(hidden.shape[1])
            if stride == 1 and in_ch == out_ch:
                hidden = graph_ops.add_sum(network, hidden, identity)

        # EfficientNet runs the head convolution on the feature map and pools
        # afterwards, the opposite order to MobileNetV3.
        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, head_w, None, int(head_w.shape[0]), (1, 1), dtype=work_np_dtype
        )
        hidden = self._bn(network, hidden, weights, "bn2", work_np_dtype)
        hidden = graph_ops.add_silu(network, hidden)
        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))

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
            raise RuntimeError("TensorRT timm_efficientnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_efficientnet_config") or _resolve_config(config.raw)
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


_QUALIFIED_ARCHITECTURES = {
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
}


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm EfficientNet image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_efficientnet does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("timm_efficientnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_efficientnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_efficientnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_efficientnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_efficientnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_efficientnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_efficientnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_efficientnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_efficientnet does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_efficientnet supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    architecture = str(config.raw.get("architecture") or config.model_type).lower()
    if architecture not in _QUALIFIED_ARCHITECTURES:
        raise ValueError(f"timm EfficientNet does not support architecture={architecture!r}")
    precision = str(request.precision).lower()
    model = _TimmEfficientnetModel()
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
    writer.set_header(family="timm_efficientnet", task=request.task, backend=request.backend)
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
