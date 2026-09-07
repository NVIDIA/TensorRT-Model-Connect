# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm RepVGG image-classification family plugin.

Supports timm RepVGG classifiers stored in HF Hub format. The initial target is:
  timm/repvgg_a2.rvgg_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

The published checkpoints are in RepVGG's *training* form: every block keeps a
3x3 branch, a 1x1 branch, and, when the shape is unchanged, a batch-norm
identity branch. This loader performs the structural reparameterisation on the
host, folding all three into a single 3x3 convolution with bias, which is how
RepVGG is meant to run. The engine is therefore a plain convolution stack.

The layout is recovered from the checkpoint, including the stride: a block
downsamples exactly when it has no identity branch, which is the only case where
its input and output shapes can differ.
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
        "num_features": int(raw.get("num_features", 1408)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bilinear")),
    }


def _fold_conv_bn(
    weight: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold an inference-time batch norm into its convolution."""
    scale = gamma / np.sqrt(var + _BN_EPS)
    folded = weight * scale.reshape(-1, 1, 1, 1)
    return folded.astype(np.float32), (beta - mean * scale).astype(np.float32)


def _identity_kernel(
    channels: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The identity branch as an equivalent 3x3 convolution.

    A batch norm applied straight to the input is a 1x1 identity convolution
    scaled per channel, which pads into the centre of a 3x3 kernel.
    """
    scale = gamma / np.sqrt(var + _BN_EPS)
    kernel = np.zeros((channels, channels, 3, 3), dtype=np.float32)
    for channel in range(channels):
        kernel[channel, channel, 1, 1] = scale[channel]
    return kernel, (beta - mean * scale).astype(np.float32)


def _discover_layout(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    leaves: dict[tuple[int, int], set[str]] = {}
    pattern = re.compile(r"^stages\.(\d+)\.(\d+)\.(.+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            leaves.setdefault(key, set()).add(match.group(3).split(".")[0])
    if not leaves:
        raise ValueError("Checkpoint has no stages.<stage>.<block> entries")

    stages = sorted({stage for stage, _ in leaves})
    if stages != list(range(len(stages))):
        raise ValueError("Stage indices are not contiguous")

    blocks = []
    for stage in stages:
        indices = sorted(index for s, index in leaves if s == stage)
        if indices != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        for index in indices:
            present = leaves[(stage, index)]
            if "conv_kxk" not in present:
                raise ValueError(f"stages.{stage}.{index} has no conv_kxk branch")
            has_identity = "identity" in present
            blocks.append(
                {
                    "prefix": f"stages.{stage}.{index}",
                    "has_identity": has_identity,
                    # Only a block without an identity branch may change shape.
                    "stride": 1 if has_identity else 2,
                }
            )
    return {"blocks": blocks, "num_stages": len(stages)}


class TimmRepvggPlugin:
    name = "timm_repvgg"
    runtime_strategy = "timm_repvgg_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_repvgg":
            return True
        return mt.startswith("repvgg")

    def _reparameterise(
        self, readers, prefix: str, has_identity: bool, target_dtype
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse the 3x3, 1x1 and identity branches into one 3x3 convolution."""

        def bn(leaf: str):
            return tuple(
                _load_tensor(readers, f"{leaf}.{suffix}").astype(np.float32)
                for suffix in ("weight", "bias", "running_mean", "running_var")
            )

        kxk = _load_tensor(readers, f"{prefix}.conv_kxk.conv.weight").astype(np.float32)
        out_ch, in_per_group, kh, kw = kxk.shape
        if (kh, kw) != (3, 3):
            raise ValueError(f"{prefix}: expected a 3x3 branch, found {kh}x{kw}")

        gamma, beta, mean, var = bn(f"{prefix}.conv_kxk.bn")
        fused_w, fused_b = _fold_conv_bn(kxk, gamma, beta, mean, var)

        one = _load_tensor(readers, f"{prefix}.conv_1x1.conv.weight").astype(np.float32)
        if one.shape[1] != in_per_group:
            raise ValueError(
                f"{prefix}: the 1x1 and 3x3 branches disagree on grouping")
        gamma, beta, mean, var = bn(f"{prefix}.conv_1x1.bn")
        one_w, one_b = _fold_conv_bn(one, gamma, beta, mean, var)
        # A 1x1 kernel is a 3x3 kernel with only its centre populated.
        padded = np.zeros_like(fused_w)
        padded[:, :, 1:2, 1:2] = one_w
        fused_w = fused_w + padded
        fused_b = fused_b + one_b

        if has_identity:
            if in_per_group != out_ch:
                raise ValueError(
                    f"{prefix}: an identity branch needs matching channel counts; "
                    f"grouped RepVGG variants are not supported")
            gamma, beta, mean, var = bn(f"{prefix}.identity")
            id_w, id_b = _identity_kernel(out_ch, gamma, beta, mean, var)
            fused_w = fused_w + id_w
            fused_b = fused_b + id_b

        return fused_w.astype(target_dtype), fused_b.astype(target_dtype)

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
        raw["_timm_repvgg_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # The stem is the same multi-branch block without an identity path.
        stem_w, stem_b = self._reparameterise(readers, "stem", False, target_dtype)
        weights["stem.weight"] = stem_w
        weights["stem.bias"] = stem_b

        for block in layout["blocks"]:
            prefix = block["prefix"]
            fused_w, fused_b = self._reparameterise(
                readers, prefix, block["has_identity"], target_dtype)
            weights[f"{prefix}.weight"] = fused_w
            weights[f"{prefix}.bias"] = fused_b

        for key in ("head.fc.weight", "head.fc.bias"):
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
            raise NotImplementedError("timm_repvgg does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_repvgg does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_repvgg precision: {precision}")

        cfg = config.raw.get("_timm_repvgg_config")
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
                f"timm_repvgg input {image_h}x{image_w} must be divisible by {total_stride}")
        feat_h, feat_w = image_h // total_stride, image_w // total_stride

        if verbose:
            print(
                "[trtmc build] timm_repvgg: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, "
                f"classes={num_classes}, precision={precision} "
                "(branches fused into single 3x3 convolutions)",
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

        stem_w = weights["stem.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, stem_w, weights["stem.bias"], int(stem_w.shape[0]), (3, 3),
            stride=(2, 2), padding=(1, 1), dtype=work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]
            w = weights[f"{prefix}.weight"]
            hidden = graph_ops.add_conv2d(
                network, hidden, w, weights[f"{prefix}.bias"], int(w.shape[0]), (3, 3),
                stride=(stride, stride), padding=(1, 1), dtype=work_np_dtype)
            hidden = graph_ops.add_relu(network, hidden)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (feat_h, feat_w))

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
            raise RuntimeError("TensorRT timm_repvgg engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_repvgg_config") or _resolve_config(config.raw)
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


plugin = TimmRepvggPlugin()
