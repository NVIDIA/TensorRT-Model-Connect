# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm MNASNet image-classification family plugin.

Supports timm MNASNet classifiers stored in HF Hub format. The initial target
is:
  timm/mnasnet_100.rmsp_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Block shape is recovered from the checkpoint: the block kind follows from which
convolutions are present and the depthwise kernel from its weight shape. The
activation is uniform ReLU, so only the per-stage stride comes from an
architecture table. MNASNet has no squeeze-excite gate; the gate is still
detected from the keys so a variant that adds one is rejected loudly rather
than silently ignored.
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

_BN_EPS = 1e-5
_ARCHITECTURE = "mnasnet_100"

# Per-stage stride for the 7-stage MNASNet layout; the stride applies to the
# first block of each stage.
_STRIDES_7_STAGE = (1, 2, 2, 2, 1, 2, 1)

_STRIDE_SCHEDULES = {7: _STRIDES_7_STAGE}


def _resolve_config(raw: dict) -> dict:
    pcfg = raw.get("pretrained_cfg")
    if not isinstance(pcfg, dict):
        raise ValueError("timm MNASNet config requires pretrained_cfg")
    required = ("input_size", "mean", "std", "crop_pct", "interpolation")
    missing = [f"pretrained_cfg.{key}" for key in required if key not in pcfg]
    if "num_classes" not in raw:
        missing.append("num_classes")
    if missing:
        raise ValueError(f"timm MNASNet config is missing required fields: {missing}")
    input_size = pcfg["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 3 or int(input_size[0]) != 3:
        raise ValueError("timm MNASNet pretrained_cfg.input_size must be [3, height, width]")
    image_h, image_w = int(input_size[1]), int(input_size[2])
    if not isinstance(pcfg["mean"], list) or not isinstance(pcfg["std"], list):
        raise ValueError("timm MNASNet mean/std must be lists")
    mean = [float(value) for value in pcfg["mean"]]
    std = [float(value) for value in pcfg["std"]]
    crop_pct = float(pcfg["crop_pct"])
    interpolation = str(pcfg["interpolation"])
    num_classes = int(raw["num_classes"])
    if image_h <= 0 or image_w <= 0 or num_classes <= 0:
        raise ValueError("timm MNASNet image dimensions and num_classes must be positive")
    if len(mean) != 3 or len(std) != 3 or any(value == 0.0 for value in std):
        raise ValueError("timm MNASNet mean/std must contain three channels with non-zero std")
    if not 0.0 < crop_pct <= 1.0 or pcfg.get("crop_mode", "center") != "center":
        raise ValueError("timm MNASNet requires a center crop with crop_pct in (0, 1]")
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("timm MNASNet supports only bilinear or bicubic interpolation")
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
    names = set(readers.keys())

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
        raise ValueError(f"No MNASNet stride schedule for {len(stages)} stages")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        for index in indices:
            present = leaves[(stage, index)]
            if "se" in present:
                raise ValueError(
                    f"blocks.{stage}.{index} has a squeeze-excite gate, which this "
                    "family does not build"
                )
            if "conv" in present:
                kind = "conv_bn_act"
                required = {"conv", "bn1"}
            elif "conv_pwl" in present:
                kind = "inverted_residual"
                required = {"conv_pw", "bn1", "conv_dw", "bn2", "conv_pwl", "bn3"}
            else:
                kind = "depthwise_separable"
                required = {"conv_dw", "bn1", "conv_pw", "bn2"}
            expected_kind = "depthwise_separable" if stage == 0 else "inverted_residual"
            if kind != expected_kind:
                raise ValueError(f"MNASNet stage {stage} requires {expected_kind}, found {kind}")
            if present != required:
                raise ValueError(f"Unexpected MNASNet block keys at stage {stage}: {present}")
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    "stride": strides[stage] if index == 0 else 1,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class _TimmMnasnetModel:
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
        raw["_timm_mnasnet_config"] = cfg
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

        conv("conv_head.weight")
        bn("bn2")
        for key in ("classifier.weight", "classifier.bias"):
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
        *,
        precision: str,
        verbose: bool = False,
    ) -> bytes:
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_mnasnet precision: {precision}")

        cfg = config.raw.get("_timm_mnasnet_config")
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
                f"timm_mnasnet input {image_h}x{image_w} must be divisible by {total_stride}"
            )

        if verbose:
            print(
                "[trtmc build] timm_mnasnet: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, "
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
        hidden = graph_ops.add_relu(network, hidden)

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
                hidden = graph_ops.add_relu(network, hidden)
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
                hidden = graph_ops.add_relu(network, hidden)
                cur_h, cur_w = cur_h // stride, cur_w // stride
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
                hidden = graph_ops.add_relu(network, hidden)

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
                hidden = graph_ops.add_relu(network, hidden)
                cur_h, cur_w = cur_h // stride, cur_w // stride

                pwl = weights[f"{prefix}.conv_pwl.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pwl, None, int(pwl.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn3", work_np_dtype)

            in_ch = int(identity.shape[1])
            out_ch = int(hidden.shape[1])
            if stride == 1 and in_ch == out_ch:
                hidden = graph_ops.add_sum(network, hidden, identity)

        # MNASNet runs the head convolution on the feature map and pools after,
        # the same order as EfficientNet.
        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, head_w, None, int(head_w.shape[0]), (1, 1), dtype=work_np_dtype
        )
        hidden = self._bn(network, hidden, weights, "bn2", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)
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
            raise RuntimeError("TensorRT timm_mnasnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_mnasnet_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before reading bundle config")
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
    """Build one timm MNASNet image-classification bundle."""
    if request.image_height is not None:
        raise NotImplementedError("timm_mnasnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_mnasnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_mnasnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_mnasnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_mnasnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_mnasnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_mnasnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_mnasnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_mnasnet does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_mnasnet supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if config.architecture != _ARCHITECTURE:
        raise ValueError(f"timm MNASNet does not support architecture={config.architecture!r}")
    precision = str(request.precision).lower()
    model = _TimmMnasnetModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="timm_mnasnet", task=request.task, backend="trt")
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
