# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm SENet image-classification family plugin.

Supports timm SENet classifiers stored in HF Hub format. The initial target is:
  timm/senet154.gluon_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

SENet-154 is the original squeeze-and-excitation network rather than a ResNet
with SE bolted on, so three things differ from `timm_seresnet`: the stem is
three 3x3 convolutions instead of one 7x7, the bottleneck 3x3 is grouped, and
the projection shortcut is itself a 3x3 convolution in every stage that
reduces.
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


class TimmSenetPlugin:
    name = "timm_senet"
    runtime_strategy = "timm_senet_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_senet" or mt.startswith("senet")

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
        raw["_timm_senet_config"] = cfg
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

    def _conv(self, network, x, weights, key, dtype, *, stride=1, groups=1,
              bias_key=None):
        """A convolution whose kernel, padding, and grouping come from weights."""
        w = weights[key]
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

    def _squeeze_excite(self, network, x, weights, prefix, dtype):
        """Per-channel gate: pool, two 1x1 convolutions, sigmoid, multiply.

        The gate is a sigmoid over each channel independently. ResNeSt's
        split-attention looks similar but normalises across a radix axis
        instead, so the two are not interchangeable.
        """
        gate = graph_ops.add_mean_spatial(network, x)
        gate = self._conv(
            network, gate, weights, f"{prefix}.fc1.weight", dtype,
            bias_key=f"{prefix}.fc1.bias")
        gate = graph_ops.add_relu(network, gate)
        gate = self._conv(
            network, gate, weights, f"{prefix}.fc2.weight", dtype,
            bias_key=f"{prefix}.fc2.bias")
        return graph_ops.add_product(network, x, graph_ops.add_sigmoid(network, gate))

    def _block(self, network, x, weights, prefix, dtype, *, stride):
        out = self._conv(network, x, weights, f"{prefix}.conv1.weight", dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn1", dtype)
        out = graph_ops.add_relu(network, out)

        # The 3x3 is grouped; the group count follows from how many input
        # channels each filter actually sees.
        conv2 = weights[f"{prefix}.conv2.weight"]
        groups = int(out.shape[1]) // int(conv2.shape[1])
        out = self._conv(
            network, out, weights, f"{prefix}.conv2.weight", dtype,
            stride=stride, groups=groups)
        out = self._bn(network, out, weights, f"{prefix}.bn2", dtype)
        out = graph_ops.add_relu(network, out)

        out = self._conv(network, out, weights, f"{prefix}.conv3.weight", dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn3", dtype)

        if f"{prefix}.se.fc1.weight" in weights:
            out = self._squeeze_excite(network, out, weights, f"{prefix}.se", dtype)

        shortcut = x
        if f"{prefix}.downsample.0.weight" in weights:
            shortcut = self._conv(
                network, shortcut, weights, f"{prefix}.downsample.0.weight", dtype,
                stride=stride)
            shortcut = self._bn(
                network, shortcut, weights, f"{prefix}.downsample.1", dtype)

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
            raise NotImplementedError("timm_senet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_senet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_senet precision: {precision}")

        cfg = config.raw.get("_timm_senet_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        depths = cfg["depths"]

        if verbose:
            print(
                "[trtmc build] timm_senet: "
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

        # Deep stem: three 3x3 convolutions, only the first strided. A norm in
        # the same Sequential also has a `.weight`, so the convolutions are
        # picked out by rank rather than by position, and the last one's norm
        # is the top-level bn1.
        stem_convs = sorted(
            int(m.group(1))
            for m in (re.match(r"^conv1\.(\d+)\.weight$", n) for n in weights) if m
            and weights[m.group(0)].ndim == 4)
        if not stem_convs:
            raise ValueError("Checkpoint has no stem convolutions")
        for position, index in enumerate(stem_convs):
            hidden = self._conv(
                network, hidden, weights, f"conv1.{index}.weight", work_np_dtype,
                stride=2 if position == 0 else 1)
            if position == len(stem_convs) - 1:
                break
            hidden = self._bn(
                network, hidden, weights, f"conv1.{index + 1}", work_np_dtype)
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
            raise RuntimeError("TensorRT timm_senet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_senet_config") or _resolve_config(config.raw)
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


plugin = TimmSenetPlugin()
