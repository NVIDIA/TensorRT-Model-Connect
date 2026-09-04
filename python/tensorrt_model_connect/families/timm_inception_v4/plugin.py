# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm Inception-v4 image-classification family plugin.

Supports timm Inception-v4 classifiers stored in HF Hub format. The initial
target is:
  timm/inception_v4.tf_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Inception-v4 is a flat `features` sequence of 22 blocks in eight distinct
shapes. Each block is classified by the branch names present in the checkpoint,
so the block order is read off the checkpoint and only the branch wiring is
written out. Convolution kernels come from the weight shapes; the strides and
paddings that are not derivable were read out of timm rather than guessed.
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

# Inception uses the TensorFlow batch-norm epsilon, not the PyTorch default.
_BN_EPS = 1e-3

# A "same"-style branch pads according to its kernel; the factorised 1xN and
# Nx1 convolutions pad on one axis only. Strided reduction convolutions pad zero
# and pass their padding explicitly.
_SAME_PAD = {(1, 1): (0, 0), (3, 3): (1, 1), (1, 7): (0, 3), (7, 1): (3, 0),
             (1, 3): (0, 1), (3, 1): (1, 0)}

# The three stem convolutions, in order: (stride, padding).
_STEM = ((2, 0), (1, 0), (1, 1))


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
        "num_features": int(raw.get("num_features", 1536)),
        "mean": [float(v) for v in pcfg.get("mean", [0.5, 0.5, 0.5])],
        "std": [float(v) for v in pcfg.get("std", [0.5, 0.5, 0.5])],
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _classify(index: int, branches: set[str], keys: set[str]) -> str:
    """Name a block from the keys the checkpoint carries for it.

    The stem convolutions never reach here; they are handled by index. The
    pooling branches carry no weights, so several topologies present the same
    top-level branch names and have to be told apart one level deeper.
    """
    if branches == {"conv"}:
        # a single convolution beside a weightless pooling branch
        return "mixed3a" if index == 3 else "mixed5a"
    if "branch1_0" in branches:
        return "inception_c"
    if branches == {"branch0", "branch1", "branch2", "branch3"}:
        return "inception_ab"
    if branches == {"branch0", "branch1"}:
        # Reduction-A's first branch is a single convolution, so it is
        # distinguishable. Mixed4a and Reduction-B are not: they have identical
        # branch shapes and differ only in stride, which the checkpoint does not
        # record. Ordering is therefore the only honest discriminator, and the
        # caller supplies it.
        if any(k.startswith("branch0.conv") for k in keys):
            return "reduction_a"
        return "chained_pair"
    raise ValueError(f"features.{index}: unrecognised Inception-v4 block topology")


def _discover_layout(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    tops: dict[int, set[str]] = {}
    detail: dict[int, set[str]] = {}
    pattern = re.compile(r"^features\.(\d+)\.(.+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            block = int(match.group(1))
            tops.setdefault(block, set()).add(match.group(2).split(".")[0])
            detail.setdefault(block, set()).add(match.group(2))
    if not tops:
        raise ValueError("Checkpoint has no features.<index> blocks")
    indices = sorted(tops)
    if indices != list(range(len(indices))):
        raise ValueError("Block indices are not contiguous")

    blocks = []
    seen_chained_pair = False
    for index in indices:
        if index < len(_STEM):
            blocks.append({"index": index, "kind": "stem"})
            continue
        kind = _classify(index, tops[index], detail[index])
        if kind == "chained_pair":
            # The first such block is Mixed4a, which keeps its resolution; any
            # later one is Reduction-B, which halves it.
            kind = "reduction_b" if seen_chained_pair else "mixed4a"
            seen_chained_pair = True
        blocks.append({"index": index, "kind": kind})
    return {"blocks": blocks, "num_blocks": len(blocks)}


class TimmInceptionV4Plugin:
    name = "timm_inception_v4"
    runtime_strategy = "timm_inception_v4_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_inception_v4":
            return True
        return mt.startswith("inception_v4")

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
        raw["_timm_inception_v4_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        # Every convolution here is a ConvNormAct, so load them uniformly. The
        # norm statistics stay fp32 because the fold divides by their variance.
        for name in sorted(names):
            if name.endswith(".conv.weight"):
                weights[name] = _load_tensor(readers, name).astype(target_dtype)
            elif re.search(r"\.bn\.(weight|bias|running_mean|running_var)$", name):
                weights[name] = _load_tensor(readers, name).astype(np.float32)

        for key in ("last_linear.weight", "last_linear.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _conv(self, network, x, weights, prefix, dtype, *, stride=1, padding=None):
        """One ConvNormAct: convolution, folded batch norm, ReLU."""
        w = weights[f"{prefix}.conv.weight"]
        kernel = (int(w.shape[2]), int(w.shape[3]))
        pad = _SAME_PAD[kernel] if padding is None else padding
        out = graph_ops.add_conv2d(
            network, x, w, None, int(w.shape[0]), kernel,
            stride=(stride, stride), padding=pad, dtype=dtype)
        out = graph_ops.add_batch_norm(
            network, out,
            weights[f"{prefix}.bn.weight"], weights[f"{prefix}.bn.bias"],
            weights[f"{prefix}.bn.running_mean"], weights[f"{prefix}.bn.running_var"],
            _BN_EPS, dtype=dtype)
        return graph_ops.add_relu(network, out)

    def _chain(self, network, x, weights, prefix, dtype):
        """Apply prefix.0, prefix.1, ... for as many steps as the weights define."""
        out = x
        step = 0
        while f"{prefix}.{step}.conv.weight" in weights:
            out = self._conv(network, out, weights, f"{prefix}.{step}", dtype)
            step += 1
        if step == 0:
            raise ValueError(f"{prefix}: no convolutions found")
        return out

    def _block(self, network, x, weights, index, kind, dtype):
        c = self._conv
        f = f"features.{index}"

        if kind == "mixed3a":
            pool = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            conv = c(network, x, weights, f"{f}.conv", dtype, stride=2, padding=(0, 0))
            return graph_ops.add_concat(network, [pool, conv])

        if kind == "mixed5a":
            conv = c(network, x, weights, f"{f}.conv", dtype, stride=2, padding=(0, 0))
            pool = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            return graph_ops.add_concat(network, [conv, pool])

        if kind == "mixed4a":
            b0 = c(network, x, weights, f"{f}.branch0.0", dtype)
            b0 = c(network, b0, weights, f"{f}.branch0.1", dtype, padding=(0, 0))
            b1 = c(network, x, weights, f"{f}.branch1.0", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.1", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.2", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.3", dtype, padding=(0, 0))
            return graph_ops.add_concat(network, [b0, b1])

        if kind == "inception_ab":
            # InceptionA and InceptionB share a shape: a 1x1 branch, two chains
            # of differing depth, and a pooled branch. Walk whatever depth the
            # checkpoint declares rather than tabulating it.
            b0 = c(network, x, weights, f"{f}.branch0", dtype)
            b1 = self._chain(network, x, weights, f"{f}.branch1", dtype)
            b2 = self._chain(network, x, weights, f"{f}.branch2", dtype)
            b3 = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
            b3 = c(network, b3, weights, f"{f}.branch3.1", dtype)
            return graph_ops.add_concat(network, [b0, b1, b2, b3])

        if kind == "reduction_a":
            b0 = c(network, x, weights, f"{f}.branch0", dtype, stride=2, padding=(0, 0))
            b1 = c(network, x, weights, f"{f}.branch1.0", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.1", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.2", dtype, stride=2, padding=(0, 0))
            b2 = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            return graph_ops.add_concat(network, [b0, b1, b2])

        if kind == "reduction_b":
            b0 = c(network, x, weights, f"{f}.branch0.0", dtype)
            b0 = c(network, b0, weights, f"{f}.branch0.1", dtype, stride=2, padding=(0, 0))
            b1 = c(network, x, weights, f"{f}.branch1.0", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.1", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.2", dtype)
            b1 = c(network, b1, weights, f"{f}.branch1.3", dtype, stride=2, padding=(0, 0))
            b2 = graph_ops.add_max_pool2d(network, x, 3, 2, 0)
            return graph_ops.add_concat(network, [b0, b1, b2])

        # inception_c: two branches split into asymmetric pairs and rejoin.
        b0 = c(network, x, weights, f"{f}.branch0", dtype)
        b1 = c(network, x, weights, f"{f}.branch1_0", dtype)
        b1 = graph_ops.add_concat(network, [
            c(network, b1, weights, f"{f}.branch1_1a", dtype),
            c(network, b1, weights, f"{f}.branch1_1b", dtype),
        ])
        b2 = c(network, x, weights, f"{f}.branch2_0", dtype)
        b2 = c(network, b2, weights, f"{f}.branch2_1", dtype)
        b2 = c(network, b2, weights, f"{f}.branch2_2", dtype)
        b2 = graph_ops.add_concat(network, [
            c(network, b2, weights, f"{f}.branch2_3a", dtype),
            c(network, b2, weights, f"{f}.branch2_3b", dtype),
        ])
        b3 = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
        b3 = c(network, b3, weights, f"{f}.branch3.1", dtype)
        return graph_ops.add_concat(network, [b0, b1, b2, b3])

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
                "timm_inception_v4 does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError(
                "timm_inception_v4 does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_inception_v4 precision: {precision}")

        cfg = config.raw.get("_timm_inception_v4_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        if verbose:
            print(
                "[trtmc build] timm_inception_v4: "
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

        for block in blocks:
            index, kind = block["index"], block["kind"]
            if kind == "stem":
                stride, pad = _STEM[index]
                hidden = self._conv(
                    network, hidden, weights, f"features.{index}", work_np_dtype,
                    stride=stride, padding=(pad, pad))
                continue
            hidden = self._block(network, hidden, weights, index, kind, work_np_dtype)

        shape = hidden.shape
        hidden = graph_ops.add_global_avg_pool(
            network, hidden, (int(shape[2]), int(shape[3])))

        fc_w = weights["last_linear.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["last_linear.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_inception_v4 engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_inception_v4_config") or _resolve_config(config.raw)
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


plugin = TimmInceptionV4Plugin()
