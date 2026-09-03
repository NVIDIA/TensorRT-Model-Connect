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

# Same for MobileNetV3-Small, which starts with a stride-2 depthwise stage.
_SMALL_SCHEDULE = (
    (2, "relu"),
    (2, "relu"),
    (2, "hard_swish"),
    (1, "hard_swish"),
    (2, "hard_swish"),
    (1, "hard_swish"),
)

_SCHEDULES = {6: _SMALL_SCHEDULE, 7: _LARGE_SCHEDULE}


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
    """Recover the block list from the checkpoint keys."""
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
    schedule = _SCHEDULES.get(len(stages))
    if schedule is None:
        raise ValueError(
            f"No MobileNetV3 stride/activation schedule for {len(stages)} stages")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        stride, activation = schedule[stage]
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
                    "stride": stride if index == 0 else 1,
                    "activation": activation,
                    "has_se": "se" in present,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class TimmMobilenetv3Plugin:
    name = "timm_mobilenetv3"
    runtime_strategy = "timm_mobilenetv3_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_mobilenetv3":
            return True
        return mt.startswith("mobilenetv3")

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
                        readers, f"{prefix}.{leaf}.bias").astype(target_dtype)

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

    def _act(self, network, x, kind, dtype):
        if kind == "relu":
            return graph_ops.add_relu(network, x)
        return graph_ops.add_hard_swish(network, x, dtype=dtype)

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
            raise NotImplementedError(
                "timm_mobilenetv3 does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError(
                "timm_mobilenetv3 does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_mobilenetv3 precision: {precision}")

        cfg = config.raw.get("_timm_mobilenetv3_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
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
                f"timm_mobilenetv3 input {image_h}x{image_w} must be divisible "
                f"by {total_stride}")

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
        # The stem activation is hard-swish in both MobileNetV3 variants.
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
                    network, hidden, w, None, int(w.shape[0]), (1, 1),
                    dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)
                continue

            if block["kind"] == "depthwise_separable":
                dw = weights[f"{prefix}.conv_dw.weight"]
                k = int(dw.shape[2])
                hidden = graph_ops.add_conv2d(
                    network, hidden, dw, None, int(dw.shape[0]), (k, k),
                    stride=(stride, stride), padding=(k // 2, k // 2),
                    groups=int(dw.shape[0]), dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)
                cur_h, cur_w = cur_h // stride, cur_w // stride
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1),
                    dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn2", work_np_dtype)
                # timm's depthwise-separable block has no activation after the
                # pointwise projection.
            else:
                pw = weights[f"{prefix}.conv_pw.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pw, None, int(pw.shape[0]), (1, 1),
                    dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn1", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)

                dw = weights[f"{prefix}.conv_dw.weight"]
                k = int(dw.shape[2])
                hidden = graph_ops.add_conv2d(
                    network, hidden, dw, None, int(dw.shape[0]), (k, k),
                    stride=(stride, stride), padding=(k // 2, k // 2),
                    groups=int(dw.shape[0]), dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn2", work_np_dtype)
                hidden = self._act(network, hidden, act, work_np_dtype)
                cur_h, cur_w = cur_h // stride, cur_w // stride

                if block["has_se"]:
                    hidden = graph_ops.add_squeeze_excite(
                        network, hidden, (cur_h, cur_w),
                        weights[f"{prefix}.se.conv_reduce.weight"],
                        weights[f"{prefix}.se.conv_reduce.bias"],
                        weights[f"{prefix}.se.conv_expand.weight"],
                        weights[f"{prefix}.se.conv_expand.bias"],
                        dtype=work_np_dtype)

                pwl = weights[f"{prefix}.conv_pwl.weight"]
                hidden = graph_ops.add_conv2d(
                    network, hidden, pwl, None, int(pwl.shape[0]), (1, 1),
                    dtype=work_np_dtype)
                hidden = self._bn(network, hidden, weights, f"{prefix}.bn3", work_np_dtype)

            # Residual only when the block keeps both shape and channel count.
            in_ch = int(identity.shape[1])
            out_ch = int(hidden.shape[1])
            if stride == 1 and in_ch == out_ch:
                hidden = graph_ops.add_sum(network, hidden, identity)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (cur_h, cur_w))

        head_w = weights["conv_head.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, head_w, weights["conv_head.bias"],
            int(head_w.shape[0]), (1, 1), dtype=work_np_dtype)
        hidden = graph_ops.add_hard_swish(network, hidden, dtype=work_np_dtype)

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
            raise RuntimeError("TensorRT timm_mobilenetv3 engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_mobilenetv3_config") or _resolve_config(config.raw)
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


plugin = TimmMobilenetv3Plugin()
