# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm MobileNetV3 image-classification family plugin.

Supports timm MobileNetV3 classifiers stored in HF Hub format. The initial
target is:
  timm/mobilenetv3_large_100.ra_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Block shape is recovered from the checkpoint: the block type follows from which
convolutions are present, the depthwise kernel from its weight shape, and the
squeeze-excite gate from the `se.*` keys. Two things are not recorded in the
checkpoint and come from an architecture table instead: the per-stage stride and
the activation, because MobileNetV3 uses ReLU in the early stages and hard-swish
later. Numerical parity against the reference is what validates that table.
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

# Per-stage (stride, activation) for the MobileNetV3-Large layout. The stride
# applies to the first block of the stage; later blocks in the stage use 1.
_LARGE_SCHEDULE = (
    (1, "relu"),
    (2, "relu"),
    (2, "relu"),
    (2, "hard_swish"),
    (1, "hard_swish"),
    (2, "hard_swish"),
    (1, "hard_swish"),
)

_LARGE_BLOCK_COUNTS = (1, 2, 3, 4, 2, 3, 1)
_LARGE_SE_BLOCKS = frozenset({(2, 0), (2, 1), (2, 2), (4, 0), (4, 1), (5, 0), (5, 1), (5, 2)})

_ARCHITECTURE = "mobilenetv3_large_100"


def _resolve_config(raw: dict) -> dict:
    pcfg = raw.get("pretrained_cfg")
    if not isinstance(pcfg, dict):
        raise ValueError("timm MobileNetV3 config requires pretrained_cfg")
    required = ("input_size", "mean", "std", "crop_pct", "crop_mode", "interpolation")
    missing = [f"pretrained_cfg.{key}" for key in required if key not in pcfg]
    if "num_classes" not in raw:
        missing.append("num_classes")
    if missing:
        raise ValueError(f"timm MobileNetV3 config is missing required fields: {missing}")
    input_size = pcfg["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 3 or int(input_size[0]) != 3:
        raise ValueError("timm MobileNetV3 pretrained_cfg.input_size must be [3, height, width]")
    image_h, image_w = int(input_size[1]), int(input_size[2])
    if not isinstance(pcfg["mean"], list) or not isinstance(pcfg["std"], list):
        raise ValueError("timm MobileNetV3 mean/std must be lists")
    mean = [float(value) for value in pcfg["mean"]]
    std = [float(value) for value in pcfg["std"]]
    crop_pct = float(pcfg["crop_pct"])
    interpolation = str(pcfg["interpolation"])
    num_classes = int(raw["num_classes"])
    if image_h <= 0 or image_w <= 0 or num_classes <= 0:
        raise ValueError("timm MobileNetV3 image dimensions and num_classes must be positive")
    if len(mean) != 3 or len(std) != 3 or any(value == 0.0 for value in std):
        raise ValueError("timm MobileNetV3 mean/std must contain three channels with non-zero std")
    if not 0.0 < crop_pct <= 1.0 or pcfg["crop_mode"] != "center":
        raise ValueError("timm MobileNetV3 requires a center crop with crop_pct in (0, 1]")
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("timm MobileNetV3 supports only bilinear or bicubic interpolation")
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
    """Recover and validate the MobileNetV3-Large block layout."""
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
    if stages != list(range(len(_LARGE_SCHEDULE))):
        raise ValueError("MobileNetV3-Large requires exactly seven contiguous stages")

    blocks = []
    for stage, (stage_stride, activation) in enumerate(_LARGE_SCHEDULE):
        indices = sorted(index for owner, index in leaves if owner == stage)
        expected_count = _LARGE_BLOCK_COUNTS[stage]
        if indices != list(range(expected_count)):
            raise ValueError(
                f"MobileNetV3-Large stage {stage} requires {expected_count} contiguous blocks"
            )
        for index in indices:
            present = leaves[(stage, index)]
            has_se = "se" in present
            expected_se = (stage, index) in _LARGE_SE_BLOCKS
            if has_se != expected_se:
                raise ValueError(
                    f"MobileNetV3-Large stage {stage} block {index} SE topology mismatch"
                )
            if "conv" in present:
                kind = "conv_bn_act"
                required = {"conv", "bn1"}
            elif "conv_pwl" in present:
                kind = "inverted_residual"
                required = {"conv_pw", "bn1", "conv_dw", "bn2", "conv_pwl", "bn3"}
            elif "conv_dw" in present and "conv_pw" in present:
                kind = "depthwise_separable"
                required = {"conv_dw", "bn1", "conv_pw", "bn2"}
                if has_se:
                    raise ValueError("MobileNetV3-Large depthwise block must not contain SE")
            else:
                raise ValueError(f"Unsupported MobileNetV3 block keys at stage {stage}: {present}")
            expected_kind = (
                "depthwise_separable"
                if stage == 0
                else "conv_bn_act"
                if stage == 6
                else "inverted_residual"
            )
            if kind != expected_kind:
                raise ValueError(
                    f"MobileNetV3-Large stage {stage} requires {expected_kind}, found {kind}"
                )
            expected = required | ({"se"} if has_se else set())
            if present != expected:
                raise ValueError(f"Unexpected MobileNetV3 block keys at stage {stage}: {present}")
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    "stride": stage_stride if index == 0 else 1,
                    "activation": activation,
                    "has_se": has_se,
                }
            )
    return {"blocks": blocks}


class _TimmMobilenetv3Model:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str,
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        raw = config.raw
        cfg = _resolve_config(raw)
        layout = _discover_layout(readers)
        cfg.update(layout)
        raw["_timm_mobilenetv3_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def bn(prefix: str) -> None:
            # Kept fp32: the fold computes 1/sqrt(var + eps) on the host.
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
                continue
            if block["kind"] == "depthwise_separable":
                conv(f"{prefix}.conv_dw.weight")
                bn(f"{prefix}.bn1")
                conv(f"{prefix}.conv_pw.weight")
                bn(f"{prefix}.bn2")
            elif block["kind"] == "inverted_residual":
                conv(f"{prefix}.conv_pw.weight")
                bn(f"{prefix}.bn1")
                conv(f"{prefix}.conv_dw.weight")
                bn(f"{prefix}.bn2")
                conv(f"{prefix}.conv_pwl.weight")
                bn(f"{prefix}.bn3")
            else:
                raise ValueError(f"Unsupported MobileNetV3 block kind: {block['kind']}")
            if block["has_se"]:
                for leaf in ("se.conv_reduce", "se.conv_expand"):
                    conv(f"{prefix}.{leaf}.weight")
                    weights[f"{prefix}.{leaf}.bias"] = _load_tensor(
                        readers, f"{prefix}.{leaf}.bias"
                    ).astype(target_dtype)

        for key in ("conv_head.weight", "conv_head.bias", "classifier.weight", "classifier.bias"):
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

    def _act(self, network, x, kind, dtype):
        if kind == "relu":
            return graph_ops.add_relu(network, x)
        if kind == "hard_swish":
            return graph_ops.add_hard_swish(network, x, dtype=dtype)
        raise ValueError(f"Unsupported MobileNetV3 activation: {kind}")

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
            raise ValueError(f"Unsupported timm_mobilenetv3 precision: {precision}")

        cfg = config.raw.get("_timm_mobilenetv3_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        # The stem is stride 2; every block stride multiplies on top of it.
        total_stride = 2
        for block in blocks:
            total_stride *= block["stride"]
        if image_h % total_stride != 0 or image_w % total_stride != 0:
            raise ValueError(
                f"timm_mobilenetv3 input {image_h}x{image_w} must be divisible by {total_stride}"
            )

        if verbose:
            print(
                "[trtmc build] timm_mobilenetv3: "
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
        # MobileNetV3-Large uses hard-swish after its stem.
        hidden = graph_ops.add_hard_swish(network, hidden, dtype=work_np_dtype)

        cur_h, cur_w = image_h // 2, image_w // 2
        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]
            act = block["activation"]
            identity = hidden

            if block["kind"] == "conv_bn_act":
                w = weights[f"{prefix}.conv.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, w, None, int(w.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)
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
                hidden = self._act(network, hidden, act, work_np_dtype)
                cur_h, cur_w = cur_h // stride, cur_w // stride
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn2", work_np_dtype)
                # timm's depthwise-separable block has no activation after the
                # pointwise projection.
            elif block["kind"] == "inverted_residual":
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1), dtype=work_np_dtype
                )
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)

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
                hidden = self._act(network, hidden, act, work_np_dtype)
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
            else:
                raise ValueError(f"Unsupported MobileNetV3 block kind: {block['kind']}")

            # Residual only when the block keeps both shape and channel count.
            in_ch = int(identity.shape[1])
            out_ch = int(hidden.shape[1])
            if stride == 1 and in_ch == out_ch:
                hidden = graph_ops.add_sum(network, hidden, identity)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))

        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            head_w,
            weights["conv_head.bias"],
            int(head_w.shape[0]),
            (1, 1),
            dtype=work_np_dtype,
        )
        hidden = graph_ops.add_hard_swish(network, hidden, dtype=work_np_dtype)

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
            raise RuntimeError("TensorRT timm_mobilenetv3 engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_mobilenetv3_config")
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
    """Build one timm MobileNetV3 image-classification bundle."""
    if request.image_height is not None:
        raise NotImplementedError("timm_mobilenetv3 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_mobilenetv3 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_mobilenetv3 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_mobilenetv3 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_mobilenetv3 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_mobilenetv3 does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_mobilenetv3 supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_mobilenetv3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_mobilenetv3 does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_mobilenetv3 supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if config.architecture != _ARCHITECTURE:
        raise ValueError(f"timm MobileNetV3 does not support architecture={config.architecture!r}")
    precision = str(request.precision).lower()
    model = _TimmMobilenetv3Model()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="timm_mobilenetv3", task=request.task, backend="trt")
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
