# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm MobileNetV4 image-classification family plugin.

Supports the pure-convolution MobileNetV4 widths stored in HF Hub format:
`mobilenetv4_conv_small_050`, `mobilenetv4_conv_small`,
`mobilenetv4_conv_medium` and `mobilenetv4_conv_large`.

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Almost the whole layout is recovered from the checkpoint. Which of the three
block kinds a position holds follows from its leaf names, which do not overlap:
a universal inverted bottleneck carries `pw_exp`, an edge residual carries
`conv_exp`, and a plain convolution carries only `conv`. Kernel sizes, channel
counts and which of the optional depthwise convolutions are present all follow
from the weight shapes.

Three things are not recorded in the checkpoint and are stated here instead:
the per-stage stride, the activation after each norm, and which blocks add
their input back. Each was read off the reference across all four widths rather
than inferred, and each is covered by a test - see `_STRIDED_STAGES` and
`_skips_input` for what goes wrong when they are guessed.
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

# The widths this family claims. The `hybrid` widths add a Mobile MQA attention
# block and the `aa` and `blur` widths move their stride into an anti-aliasing
# blur pool, so neither is built here.
_ARCHITECTURES = (
    "mobilenetv4_conv_small_050",
    "mobilenetv4_conv_small",
    "mobilenetv4_conv_medium",
    "mobilenetv4_conv_large",
)

# There are five stages. The first block of each of the first four halves the
# resolution and every other block keeps it; together with the stride-2 stem
# that is the factor of 32 the published input sizes assume. Stride is not
# recorded in a safetensors checkpoint, so this is stated rather than read; it
# was checked against all four widths.
_STAGE_COUNT = 5
_STRIDED_STAGES = frozenset({0, 1, 2, 3})

# Activation after each normalisation, by block kind and leaf. MobileNetV4 uses
# ReLU throughout, but not after every norm: the depthwise convolution that
# opens a universal inverted bottleneck and the pointwise projection that closes
# it are both left linear. Read off all four widths.
_ACTIVATED = {
    ("universal_inverted", "dw_start"): False,
    ("universal_inverted", "pw_exp"): True,
    ("universal_inverted", "dw_mid"): True,
    ("universal_inverted", "pw_proj"): False,
    ("edge_residual", "bn1"): True,
    ("edge_residual", "bn2"): False,
    ("conv_bn_act", "bn1"): True,
}

# The leaves that identify each block kind. They do not overlap, so the kind is
# read from the checkpoint rather than tabulated.
_KIND_MARKERS = (
    ("universal_inverted", "pw_exp"),
    ("edge_residual", "conv_exp"),
    ("conv_bn_act", "conv"),
)

_REQUIRED_LEAVES = {
    "universal_inverted": {"pw_exp", "pw_proj"},
    "edge_residual": {"conv_exp", "bn1", "conv_pwl", "bn2"},
    "conv_bn_act": {"conv", "bn1"},
}

_OPTIONAL_LEAVES = {
    "universal_inverted": {"dw_start", "dw_mid", "dw_end"},
    "edge_residual": set(),
    "conv_bn_act": set(),
}


def _skips_input(kind: str, stride: int, in_channels: int, out_channels: int) -> bool:
    """Whether a block adds its input back.

    Shape alone is not the rule. A plain convolution never adds its input back,
    even where the channel count and the stride would allow it, and two of the
    published widths contain exactly such a block: `conv_small` and
    `conv_small_050` each hold one 32-to-32 stride-1 `ConvBnAct` whose input is
    not added. Building those with a residual still produces a working engine
    with quietly wrong logits, which is why this is a rule and not a shape test.
    """
    if kind == "conv_bn_act":
        return False
    return stride == 1 and in_channels == out_channels


def _resolve_config(raw: dict) -> dict:
    pcfg = raw.get("pretrained_cfg")
    if not isinstance(pcfg, dict):
        raise ValueError("timm MobileNetV4 config requires pretrained_cfg")
    required = ("input_size", "mean", "std", "crop_pct", "crop_mode", "interpolation")
    missing = [f"pretrained_cfg.{key}" for key in required if key not in pcfg]
    if "num_classes" not in raw:
        missing.append("num_classes")
    if missing:
        raise ValueError(f"timm MobileNetV4 config is missing required fields: {missing}")
    input_size = pcfg["input_size"]
    if not isinstance(input_size, list) or len(input_size) != 3 or int(input_size[0]) != 3:
        raise ValueError("timm MobileNetV4 pretrained_cfg.input_size must be [3, height, width]")
    image_h, image_w = int(input_size[1]), int(input_size[2])
    if not isinstance(pcfg["mean"], list) or not isinstance(pcfg["std"], list):
        raise ValueError("timm MobileNetV4 mean/std must be lists")
    mean = [float(value) for value in pcfg["mean"]]
    std = [float(value) for value in pcfg["std"]]
    crop_pct = float(pcfg["crop_pct"])
    interpolation = str(pcfg["interpolation"])
    num_classes = int(raw["num_classes"])
    if image_h <= 0 or image_w <= 0 or num_classes <= 0:
        raise ValueError("timm MobileNetV4 image dimensions and num_classes must be positive")
    if len(mean) != 3 or len(std) != 3 or any(value == 0.0 for value in std):
        raise ValueError("timm MobileNetV4 mean/std must contain three channels with non-zero std")
    if not 0.0 < crop_pct <= 1.0 or pcfg["crop_mode"] != "center":
        raise ValueError("timm MobileNetV4 requires a center crop with crop_pct in (0, 1]")
    if interpolation not in {"bilinear", "bicubic"}:
        raise ValueError("timm MobileNetV4 supports only bilinear or bicubic interpolation")
    return {
        "image_size_h": image_h,
        "image_size_w": image_w,
        "num_classes": num_classes,
        "mean": mean,
        "std": std,
        "crop_pct": crop_pct,
        "interpolation": interpolation,
    }


def _block_kind(present: set[str]) -> str:
    for kind, marker in _KIND_MARKERS:
        if marker in present:
            return kind
    raise ValueError(f"Unsupported MobileNetV4 block keys: {sorted(present)}")


def _discover_layout(readers) -> dict:
    """Recover and validate the MobileNetV4 block layout from the checkpoint."""
    leaves: dict[tuple[int, int], set[str]] = {}
    pattern = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")
    for name in readers.keys():
        match = pattern.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".")[0])
    if not leaves:
        raise ValueError("Checkpoint has no blocks.<stage>.<index> entries")

    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(_STAGE_COUNT)):
        raise ValueError(
            f"MobileNetV4 requires exactly {_STAGE_COUNT} contiguous stages, found {stages}"
        )

    blocks = []
    for stage in stages:
        indices = sorted(index for owner, index in leaves if owner == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"MobileNetV4 stage {stage} has a gap in its block numbering")
        for index in indices:
            present = leaves[(stage, index)]
            kind = _block_kind(present)
            required = _REQUIRED_LEAVES[kind]
            allowed = required | _OPTIONAL_LEAVES[kind]
            if not required <= present:
                raise ValueError(
                    f"MobileNetV4 stage {stage} block {index} is missing "
                    f"{sorted(required - present)}"
                )
            if not present <= allowed:
                # Squeeze-excite and layer scale appear in other MobileNetV4
                # variants. Refusing here keeps a silently partial build from
                # looking like a working one.
                raise ValueError(
                    f"MobileNetV4 stage {stage} block {index} carries unsupported "
                    f"leaves {sorted(present - allowed)}"
                )
            blocks.append(
                {
                    "prefix": f"blocks.{stage}.{index}",
                    "kind": kind,
                    "stride": 2 if index == 0 and stage in _STRIDED_STAGES else 1,
                    "leaves": sorted(present),
                }
            )
    return {"blocks": blocks}


class _TimmMobilenetv4Model:
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
        raw["_timm_mobilenetv4_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def bn(prefix: str) -> None:
            # Kept fp32: the fold computes 1/sqrt(var + eps) on the host.
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                weights[f"{prefix}.{suffix}"] = _load_tensor(readers, f"{prefix}.{suffix}").astype(
                    np.float32
                )

        conv("conv_stem.weight")
        bn("bn1")

        for block in layout["blocks"]:
            prefix = block["prefix"]
            leaves = set(block["leaves"])
            if block["kind"] == "conv_bn_act":
                conv(f"{prefix}.conv.weight")
                bn(f"{prefix}.bn1")
            elif block["kind"] == "edge_residual":
                conv(f"{prefix}.conv_exp.weight")
                bn(f"{prefix}.bn1")
                conv(f"{prefix}.conv_pwl.weight")
                bn(f"{prefix}.bn2")
            else:
                for leaf in ("dw_start", "pw_exp", "dw_mid", "pw_proj", "dw_end"):
                    if leaf in leaves:
                        conv(f"{prefix}.{leaf}.conv.weight")
                        bn(f"{prefix}.{leaf}.bn")

        conv("conv_head.weight")
        bn("norm_head")
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

    def _conv_norm(
        self,
        network,
        hidden,
        weights,
        conv_key,
        norm_prefix,
        dtype,
        *,
        stride: int = 1,
        depthwise: bool = False,
        activated: bool,
    ):
        """One convolution, its norm, and the activation the reference uses."""
        weight = weights[conv_key]
        kernel = int(weight.shape[2])
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            weight,
            None,
            int(weight.shape[0]),
            (kernel, kernel),
            stride=(stride, stride),
            padding=(kernel // 2, kernel // 2),
            groups=int(weight.shape[0]) if depthwise else 1,
            dtype=dtype,
        )
        hidden = self._bn(network, hidden, weights, norm_prefix, dtype)
        if activated:
            hidden = graph_ops.add_relu(network, hidden)
        return hidden

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
            raise ValueError(f"Unsupported timm_mobilenetv4 precision: {precision}")

        cfg = config.raw.get("_timm_mobilenetv4_config")
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
                f"timm_mobilenetv4 input {image_h}x{image_w} must be divisible by {total_stride}"
            )

        if verbose:
            kinds = {
                kind: sum(1 for block in blocks if block["kind"] == kind)
                for kind in _REQUIRED_LEAVES
            }
            print(
                "[trtmc build] timm_mobilenetv4: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, {kinds}, "
                f"classes={num_classes}, precision={precision}",
                file=sys.stderr,
            )

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
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
            kind = block["kind"]
            stride = block["stride"]
            leaves = set(block["leaves"])
            identity = hidden
            in_channels = int(hidden.shape[1])

            if kind == "conv_bn_act":
                hidden = self._conv_norm(
                    network,
                    hidden,
                    weights,
                    f"{prefix}.conv.weight",
                    f"{prefix}.bn1",
                    work_np_dtype,
                    stride=stride,
                    activated=_ACTIVATED[(kind, "bn1")],
                )
            elif kind == "edge_residual":
                hidden = self._conv_norm(
                    network,
                    hidden,
                    weights,
                    f"{prefix}.conv_exp.weight",
                    f"{prefix}.bn1",
                    work_np_dtype,
                    stride=stride,
                    activated=_ACTIVATED[(kind, "bn1")],
                )
                hidden = self._conv_norm(
                    network,
                    hidden,
                    weights,
                    f"{prefix}.conv_pwl.weight",
                    f"{prefix}.bn2",
                    work_np_dtype,
                    activated=_ACTIVATED[(kind, "bn2")],
                )
            else:
                # The stride sits on the middle depthwise convolution, the only
                # one that can carry it; a strided block without one would be
                # built at the wrong resolution, so it is refused.
                if stride != 1 and "dw_mid" not in leaves:
                    raise ValueError(
                        f"MobileNetV4 {prefix} halves the resolution but has no dw_mid"
                    )
                for leaf in ("dw_start", "pw_exp", "dw_mid", "pw_proj", "dw_end"):
                    if leaf not in leaves:
                        continue
                    hidden = self._conv_norm(
                        network,
                        hidden,
                        weights,
                        f"{prefix}.{leaf}.conv.weight",
                        f"{prefix}.{leaf}.bn",
                        work_np_dtype,
                        stride=stride if leaf == "dw_mid" else 1,
                        depthwise=leaf.startswith("dw_"),
                        activated=_ACTIVATED[(kind, leaf)],
                    )

            cur_h, cur_w = cur_h // stride, cur_w // stride
            if _skips_input(kind, stride, in_channels, int(hidden.shape[1])):
                hidden = graph_ops.add_sum(network, hidden, identity)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))

        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network,
            hidden,
            head_w,
            None,
            int(head_w.shape[0]),
            (1, 1),
            dtype=work_np_dtype,
        )
        # MobileNetV4 normalises after the head convolution, which MobileNetV3
        # does not; the head convolution therefore carries no bias.
        hidden = self._bn(network, hidden, weights, "norm_head", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

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
            raise RuntimeError("TensorRT timm_mobilenetv4 engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_mobilenetv4_config")
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
    """Build one timm MobileNetV4 image-classification bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_mobilenetv4 does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_mobilenetv4 does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_mobilenetv4 does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_mobilenetv4 does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_mobilenetv4 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_mobilenetv4 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_mobilenetv4 does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_mobilenetv4 supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_mobilenetv4 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_mobilenetv4 does not support mixed-precision layers")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("timm_mobilenetv4 supports only max_sequence_length=1")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if config.architecture not in _ARCHITECTURES:
        raise ValueError(f"timm MobileNetV4 does not support architecture={config.architecture!r}")
    precision = str(request.precision).lower()
    model = _TimmMobilenetv4Model()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="timm_mobilenetv4", task=request.task, backend=request.backend)
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
