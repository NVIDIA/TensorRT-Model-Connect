# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm Inception image-classification family plugin.

Supports timm Inception-v3 classifiers stored in HF Hub format. The initial
target is:
  timm/inception_v3.tv_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Unlike the repeating stacks in the other convolutional families, Inception has
five distinct block topologies. Each one is identified from the branch names
present in the checkpoint rather than from a positional table, so the block
order is read off the checkpoint and only the branch wiring is written out.
"""

from __future__ import annotations

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

# Inception uses the TensorFlow batch-norm epsilon, not the PyTorch default.
_BN_EPS = 1e-3

# The stem, in order: (name, kernel, stride, padding) or a max pool marker.
_STEM = (
    ("Conv2d_1a_3x3", 3, 2, 0),
    ("Conv2d_2a_3x3", 3, 1, 0),
    ("Conv2d_2b_3x3", 3, 1, 1),
    ("maxpool", 3, 2, 0),
    ("Conv2d_3b_1x1", 1, 1, 0),
    ("Conv2d_4a_3x3", 3, 1, 0),
    ("maxpool", 3, 2, 0),
)

# Which branch name identifies each block topology.
_BLOCK_MARKERS = (
    ("branch5x5_1", "inception_a"),
    ("branch7x7_1", "inception_c"),
    ("branch7x7x3_1", "inception_d"),
    ("branch3x3_2a", "inception_e"),
    ("branch3x3", "inception_b"),
)


def _pretrained_cfg(raw: dict) -> dict:
    nested = raw.get("pretrained_cfg")
    return nested if isinstance(nested, dict) else raw


def _resolve_config(raw: dict) -> dict:
    pcfg = _pretrained_cfg(raw)
    input_size = pcfg.get("input_size", [3, 299, 299])
    if isinstance(input_size, int):
        image_h = image_w = int(input_size)
    else:
        image_h, image_w = int(input_size[-2]), int(input_size[-1])
    return {
        "image_size_h": image_h,
        "image_size_w": image_w,
        "num_classes": int(raw.get("num_classes", pcfg.get("num_classes", 1000))),
        "num_features": int(raw.get("num_features", 2048)),
        "mean": [float(v) for v in pcfg.get("mean", [0.5, 0.5, 0.5])],
        "std": [float(v) for v in pcfg.get("std", [0.5, 0.5, 0.5])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _discover_layout(readers) -> dict:
    """Read the Mixed block order and classify each one by its branch names."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    tops: dict[str, set[str]] = {}
    for name in names:
        if not name.startswith("Mixed_"):
            continue
        parts = name.split(".")
        if len(parts) >= 2:
            tops.setdefault(parts[0], set()).add(parts[1])
    if not tops:
        raise ValueError("Checkpoint has no Mixed_* blocks")

    blocks = []
    # Mixed_5b .. Mixed_7c sort correctly as plain strings.
    for top in sorted(tops):
        branches = tops[top]
        kind = None
        for marker, candidate in _BLOCK_MARKERS:
            if marker in branches:
                kind = candidate
                break
        if kind is None:
            raise ValueError(f"{top}: unrecognised Inception block topology")
        blocks.append({"prefix": top, "kind": kind})
    return {"blocks": blocks, "num_blocks": len(blocks)}


class TimmInceptionPlugin:
    name = "timm_inception"
    runtime_strategy = "timm_inception_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_inception":
            return True
        return mt.startswith("inception_v3")

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
        raw["_timm_inception_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def conv_bn(prefix: str) -> None:
            """Every Inception convolution is a ConvNormAct: conv plus norm."""
            weights[f"{prefix}.conv.weight"] = _load_tensor(
                readers, f"{prefix}.conv.weight").astype(target_dtype)
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.bn.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        for name, _k, _s, _p in _STEM:
            if name != "maxpool":
                conv_bn(name)

        # Load whatever branches the checkpoint declares for each block.
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for block in layout["blocks"]:
            top = block["prefix"]
            branches = sorted({
                name.split(".")[1] for name in names
                if name.startswith(top + ".") and len(name.split(".")) >= 2
            })
            for branch in branches:
                conv_bn(f"{top}.{branch}")

        for key in ("fc.weight", "fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _conv(self, network, x, weights, prefix, dtype, *, stride=1, padding=(0, 0)):
        """One ConvNormAct: convolution, folded batch norm, ReLU."""
        w = weights[f"{prefix}.conv.weight"]
        out = graph_ops.add_conv2d(
            network, x, w, None, int(w.shape[0]),
            (int(w.shape[2]), int(w.shape[3])),
            stride=(stride, stride), padding=padding, dtype=dtype)
        out = graph_ops.add_batch_norm(
            network, out,
            weights[f"{prefix}.bn.weight"], weights[f"{prefix}.bn.bias"],
            weights[f"{prefix}.bn.running_mean"], weights[f"{prefix}.bn.running_var"],
            _BN_EPS, dtype=dtype)
        return graph_ops.add_relu(network, out)

    def _block(self, network, x, weights, top, kind, dtype):
        """Build one Inception block. Branch wiring differs per topology."""
        c = self._conv

        if kind == "inception_a":
            b1 = c(network, x, weights, f"{top}.branch1x1", dtype)
            b5 = c(network, x, weights, f"{top}.branch5x5_1", dtype)
            b5 = c(network, b5, weights, f"{top}.branch5x5_2", dtype, padding=(2, 2))
            b3 = c(network, x, weights, f"{top}.branch3x3dbl_1", dtype)
            b3 = c(network, b3, weights, f"{top}.branch3x3dbl_2", dtype, padding=(1, 1))
            b3 = c(network, b3, weights, f"{top}.branch3x3dbl_3", dtype, padding=(1, 1))
            bp = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
            bp = c(network, bp, weights, f"{top}.branch_pool", dtype)
            return graph_ops.add_concat(network, [b1, b5, b3, bp])

        if kind == "inception_b":
            b3 = c(network, x, weights, f"{top}.branch3x3", dtype, stride=2)
            bd = c(network, x, weights, f"{top}.branch3x3dbl_1", dtype)
            bd = c(network, bd, weights, f"{top}.branch3x3dbl_2", dtype, padding=(1, 1))
            bd = c(network, bd, weights, f"{top}.branch3x3dbl_3", dtype, stride=2)
            bp = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            return graph_ops.add_concat(network, [b3, bd, bp])

        if kind == "inception_c":
            b1 = c(network, x, weights, f"{top}.branch1x1", dtype)
            b7 = c(network, x, weights, f"{top}.branch7x7_1", dtype)
            b7 = c(network, b7, weights, f"{top}.branch7x7_2", dtype, padding=(0, 3))
            b7 = c(network, b7, weights, f"{top}.branch7x7_3", dtype, padding=(3, 0))
            bd = c(network, x, weights, f"{top}.branch7x7dbl_1", dtype)
            bd = c(network, bd, weights, f"{top}.branch7x7dbl_2", dtype, padding=(3, 0))
            bd = c(network, bd, weights, f"{top}.branch7x7dbl_3", dtype, padding=(0, 3))
            bd = c(network, bd, weights, f"{top}.branch7x7dbl_4", dtype, padding=(3, 0))
            bd = c(network, bd, weights, f"{top}.branch7x7dbl_5", dtype, padding=(0, 3))
            bp = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
            bp = c(network, bp, weights, f"{top}.branch_pool", dtype)
            return graph_ops.add_concat(network, [b1, b7, bd, bp])

        if kind == "inception_d":
            b3 = c(network, x, weights, f"{top}.branch3x3_1", dtype)
            b3 = c(network, b3, weights, f"{top}.branch3x3_2", dtype, stride=2)
            b7 = c(network, x, weights, f"{top}.branch7x7x3_1", dtype)
            b7 = c(network, b7, weights, f"{top}.branch7x7x3_2", dtype, padding=(0, 3))
            b7 = c(network, b7, weights, f"{top}.branch7x7x3_3", dtype, padding=(3, 0))
            b7 = c(network, b7, weights, f"{top}.branch7x7x3_4", dtype, stride=2)
            bp = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            return graph_ops.add_concat(network, [b3, b7, bp])

        # inception_e: two branches split into an asymmetric pair and rejoin.
        b1 = c(network, x, weights, f"{top}.branch1x1", dtype)
        b3 = c(network, x, weights, f"{top}.branch3x3_1", dtype)
        b3 = graph_ops.add_concat(network, [
            c(network, b3, weights, f"{top}.branch3x3_2a", dtype, padding=(0, 1)),
            c(network, b3, weights, f"{top}.branch3x3_2b", dtype, padding=(1, 0)),
        ])
        bd = c(network, x, weights, f"{top}.branch3x3dbl_1", dtype)
        bd = c(network, bd, weights, f"{top}.branch3x3dbl_2", dtype, padding=(1, 1))
        bd = graph_ops.add_concat(network, [
            c(network, bd, weights, f"{top}.branch3x3dbl_3a", dtype, padding=(0, 1)),
            c(network, bd, weights, f"{top}.branch3x3dbl_3b", dtype, padding=(1, 0)),
        ])
        bp = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
        bp = c(network, bp, weights, f"{top}.branch_pool", dtype)
        return graph_ops.add_concat(network, [b1, b3, bd, bp])

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
            raise NotImplementedError("timm_inception does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_inception does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_inception precision: {precision}")

        cfg = config.raw.get("_timm_inception_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        if verbose:
            print(
                "[trtmc build] timm_inception: "
                f"image={image_h}x{image_w}, blocks={len(blocks)}, "
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

        for name, kernel, stride, padding in _STEM:
            if name == "maxpool":
                hidden = graph_ops.add_max_pool2d(network, hidden, kernel, stride, padding)
                continue
            hidden = self._conv(
                network, hidden, weights, name, work_np_dtype,
                stride=stride, padding=(padding, padding))

        for block in blocks:
            hidden = self._block(
                network, hidden, weights, block["prefix"], block["kind"], work_np_dtype)

        # The pooled map is 8x8 for the standard 299x299 input.
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
            raise RuntimeError("TensorRT timm_inception engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_inception_config") or _resolve_config(config.raw)
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


plugin = TimmInceptionPlugin()
