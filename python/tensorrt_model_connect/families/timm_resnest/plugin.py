# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ResNeSt image-classification family plugin.

Supports timm ResNeSt classifiers stored in HF Hub format. The initial target
is:
  timm/resnest50d.in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

ResNeSt replaces the 3x3 convolution in a ResNet bottleneck with split
attention: the convolution produces `radix` copies of the output channels, a
gate is computed from their sum, and the copies are recombined with a softmax
over the radix axis rather than a sigmoid over channels. That softmax is what
distinguishes it from the SE families, where the gate is per-channel and
independent.
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

# Two things the checkpoint cannot record, both specific to the `d` variants
# this family targets. A stride never lands on the split-attention convolution;
# it is carried by an average pool placed after it, and the shortcut is
# downsampled by its own average pool before the 1x1 projection.
_AVD_POOL = (3, 2, 1)
_SHORTCUT_POOL = (2, 2, 0)


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
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bilinear")),
    }


def _discover_layout(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    depths = []
    for stage in range(1, 5):
        regex = re.compile(rf"^layer{stage}\.(\d+)\.")
        indices = {int(m.group(1)) for m in map(regex.match, names) if m}
        if not indices:
            raise ValueError(f"Checkpoint has no layer{stage} blocks")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"layer{stage} block indices are not contiguous")
        depths.append(len(indices))

    if not any(re.match(r"^conv1\.\d+\.weight$", n) for n in names):
        raise ValueError("Checkpoint has no deep stem")

    return {"depths": depths}


class TimmResnestPlugin:
    name = "timm_resnest"
    runtime_strategy = "timm_resnest_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_resnest" or mt.startswith("resnest")

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
        raw["_timm_resnest_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for name in sorted(names):
            if name.endswith(".num_batches_tracked"):
                continue
            tensor = _load_tensor(readers, name)
            if tensor.ndim == 1 and not name.startswith("fc."):
                # Norm statistics stay fp32: the fold divides by their variance.
                weights[name] = tensor.astype(np.float32)
            else:
                weights[name] = tensor.astype(target_dtype)

        for key in ("fc.weight", "fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")

        return weights

    def _conv(self, network, x, weights, conv_key, dtype,
              *, stride=1, groups=1, bias_key=None):
        w = weights[conv_key]
        kernel = (int(w.shape[2]), int(w.shape[3]))
        bias = weights[bias_key] if bias_key and bias_key in weights else None
        return graph_ops.add_conv2d(
            network, x, w, bias, int(w.shape[0]), kernel,
            stride=(stride, stride),
            padding=(kernel[0] // 2, kernel[1] // 2),
            groups=groups, dtype=dtype)

    def _bn(self, network, x, weights, prefix, dtype):
        return graph_ops.add_batch_norm(
            network, x,
            weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"], weights[f"{prefix}.running_var"],
            _BN_EPS, dtype=dtype)

    def _split_attention(self, network, x, weights, prefix, dtype, *, in_channels):
        """The split-attention convolution that replaces the bottleneck 3x3.

        The convolution emits `radix` copies of the output channels. The gate is
        computed from their sum and normalised with a softmax across the radix
        axis, so the copies compete; a per-channel sigmoid, as in the SE
        families, would let them all pass.
        """
        conv_w = weights[f"{prefix}.conv.weight"]
        mid_channels = int(conv_w.shape[0])
        out_channels = int(weights[f"{prefix}.fc1.weight"].shape[1])
        radix = mid_channels // out_channels
        if radix * out_channels != mid_channels:
            raise ValueError(
                f"{prefix}: {mid_channels} channels is not a multiple of "
                f"{out_channels}")
        groups = in_channels // int(conv_w.shape[1])
        cardinality = groups // radix
        if cardinality < 1 or cardinality * radix != groups:
            raise ValueError(f"{prefix}: {groups} groups is not a multiple of radix")

        hidden = self._conv(
            network, x, weights, f"{prefix}.conv.weight", dtype, groups=groups)
        hidden = self._bn(network, hidden, weights, f"{prefix}.bn0", dtype)
        hidden = graph_ops.add_relu(network, hidden)

        shape = [int(d) for d in hidden.shape]
        height, width = shape[2], shape[3]
        split = graph_ops.add_reshape(
            network, hidden, (1, radix, out_channels, height, width))
        gap = graph_ops.add_reduce_sum(network, split, 1)
        gap = graph_ops.add_mean_spatial(network, gap, (2, 3))

        gate = self._conv(
            network, gap, weights, f"{prefix}.fc1.weight", dtype,
            groups=cardinality, bias_key=f"{prefix}.fc1.bias")
        gate = self._bn(network, gate, weights, f"{prefix}.bn1", dtype)
        gate = graph_ops.add_relu(network, gate)
        gate = self._conv(
            network, gate, weights, f"{prefix}.fc2.weight", dtype,
            groups=cardinality, bias_key=f"{prefix}.fc2.bias")

        # [1, cardinality, radix, M] -> softmax over radix -> [1, radix, C, 1, 1]
        per_group = mid_channels // (cardinality * radix)
        gate = graph_ops.add_reshape(
            network, gate, (1, cardinality, radix, per_group))
        gate = graph_ops.add_permute(network, gate, (0, 2, 1, 3))
        gate = graph_ops.add_softmax(network, gate, 1)
        gate = graph_ops.add_reshape(
            network, gate, (1, radix, out_channels, 1, 1))

        weighted = graph_ops.add_product(network, split, gate)
        return graph_ops.add_reduce_sum(network, weighted, 1)

    def _block(self, network, x, weights, prefix, dtype, *, stride):
        out = self._conv(network, x, weights, f"{prefix}.conv1.weight", dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn1", dtype)
        out = graph_ops.add_relu(network, out)

        in_channels = int(out.shape[1])
        out = self._split_attention(
            network, out, weights, f"{prefix}.conv2", dtype, in_channels=in_channels)
        if stride != 1:
            out = graph_ops.add_avg_pool2d(network, out, *_AVD_POOL)

        out = self._conv(network, out, weights, f"{prefix}.conv3.weight", dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn3", dtype)

        shortcut = x
        if f"{prefix}.downsample.1.weight" in weights:
            if stride != 1:
                shortcut = graph_ops.add_avg_pool2d(
                    network, shortcut, *_SHORTCUT_POOL)
            shortcut = self._conv(
                network, shortcut, weights, f"{prefix}.downsample.1.weight", dtype)
            shortcut = self._bn(
                network, shortcut, weights, f"{prefix}.downsample.2", dtype)

        return graph_ops.add_relu(
            network, graph_ops.add_sum(network, out, shortcut))

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
            raise NotImplementedError("timm_resnest does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_resnest does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_resnest precision: {precision}")

        cfg = config.raw.get("_timm_resnest_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        depths = cfg["depths"]

        if verbose:
            print(
                "[trtmc build] timm_resnest: "
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

        # Deep stem: the first convolution strides, the rest keep the size, and
        # each is followed by a norm and an activation except the last, whose
        # norm is the top-level bn1.
        # A norm inside the stem also has a `.weight`, so the convolutions are
        # picked out by rank rather than by position.
        stem_convs = sorted(
            int(m.group(1))
            for m in (re.match(r"^conv1\.(\d+)\.weight$", n) for n in weights) if m
            and weights[m.group(0)].ndim == 4)
        if not stem_convs:
            raise ValueError("Checkpoint has no stem convolutions")
        for position, index in enumerate(stem_convs):
            first = position == 0
            hidden = self._conv(
                network, hidden, weights, f"conv1.{index}.weight", work_np_dtype,
                stride=2 if first else 1)
            if position == len(stem_convs) - 1:
                break
            hidden = self._bn(network, hidden, weights, f"conv1.{index + 1}", work_np_dtype)
            hidden = graph_ops.add_relu(network, hidden)
        hidden = self._bn(network, hidden, weights, "bn1", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = graph_ops.add_max_pool2d(network, hidden, 3, 2, 1)

        for stage, depth in enumerate(depths, start=1):
            for block in range(depth):
                hidden = self._block(
                    network, hidden, weights, f"layer{stage}.{block}", work_np_dtype,
                    # Only the first block of stages 2 and up reduces.
                    stride=2 if (block == 0 and stage > 1) else 1)

        shape = hidden.shape
        hidden = graph_ops.add_global_avg_pool(
            network, hidden, (int(shape[2]), int(shape[3])))

        fc_w = weights["fc.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["fc.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_resnest engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_resnest_config") or _resolve_config(config.raw)
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


plugin = TimmResnestPlugin()
