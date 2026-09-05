# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm Inception-ResNet image-classification family plugin.

Supports timm Inception-ResNet-v2 classifiers stored in HF Hub format. The
initial target is:
  timm/inception_resnet_v2.tf_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

The repeat-group lengths come from the checkpoint, so the depth is not
tabulated. Two things are not recoverable from weights and are stated here
explicitly: the constant each group scales its residual by, and the fact that
the final block omits the activation after the add.
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

# The residual scale per repeat group. These are architecture constants; the
# checkpoint records nothing about them.
_GROUP_SCALE = {"repeat": 0.17, "repeat_1": 0.10, "repeat_2": 0.20}
# The trailing standalone block adds its branch unscaled and does not activate.
_FINAL_BLOCK_SCALE = 1.0

_SAME_PAD = {(1, 1): (0, 0), (3, 3): (1, 1), (5, 5): (2, 2), (1, 7): (0, 3), (7, 1): (3, 0),
             (1, 3): (0, 1), (3, 1): (1, 0)}

# Stem convolutions in order: (name, stride, padding), with pools between.
_STEM = (
    ("conv2d_1a", 2, 0),
    ("conv2d_2a", 1, 0),
    ("conv2d_2b", 1, 1),
    ("maxpool", 0, 0),
    ("conv2d_3b", 1, 0),
    ("conv2d_4a", 1, 0),
    ("maxpool", 0, 0),
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
        "num_features": int(raw.get("num_features", 1536)),
        "mean": [float(v) for v in pcfg.get("mean", [0.5, 0.5, 0.5])],
        "std": [float(v) for v in pcfg.get("std", [0.5, 0.5, 0.5])],
        "crop_pct": float(pcfg.get("crop_pct", 0.8975)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _discover_layout(readers) -> dict:
    """Read each repeat group's length from the checkpoint."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    groups: dict[str, int] = {}
    for group in _GROUP_SCALE:
        pattern = re.compile(rf"^{group}\.(\d+)\.")
        indices = {int(m.group(1)) for m in map(pattern.match, names) if m}
        if not indices:
            raise ValueError(f"Checkpoint has no {group} blocks")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"{group} block indices are not contiguous")
        groups[group] = len(indices)

    for required in ("mixed_5b", "mixed_6a", "mixed_7a", "block8", "conv2d_7b"):
        if not any(name.startswith(required + ".") for name in names):
            raise ValueError(f"Checkpoint is missing {required}")

    return {"groups": groups, "num_blocks": sum(groups.values())}


class TimmInceptionResnetPlugin:
    name = "timm_inception_resnet"
    runtime_strategy = "timm_inception_resnet_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_inception_resnet":
            return True
        return mt.startswith("inception_resnet")

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
        raw["_timm_inception_resnet_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for name in sorted(names):
            if name.endswith(".num_batches_tracked"):
                continue
            if re.search(r"\.bn\.(weight|bias|running_mean|running_var)$", name):
                # Norm statistics stay fp32: the fold divides by their variance.
                weights[name] = _load_tensor(readers, name).astype(np.float32)
            else:
                weights[name] = _load_tensor(readers, name).astype(target_dtype)

        for key in ("classif.weight", "classif.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")

        return weights

    def _conv(self, network, x, weights, prefix, dtype, *, stride=1, padding=None):
        """A ConvNormAct: convolution, folded batch norm, ReLU."""
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
        out = x
        step = 0
        while f"{prefix}.{step}.conv.weight" in weights:
            out = self._conv(network, out, weights, f"{prefix}.{step}", dtype)
            step += 1
        if step == 0:
            raise ValueError(f"{prefix}: no convolutions found")
        return out

    def _residual_block(self, network, x, weights, prefix, scale, dtype, *, activate):
        """Concatenated branches, a biased 1x1 projection, then a scaled add.

        The projection has no batch norm, unlike every other convolution here.
        """
        branches = [self._conv(network, x, weights, f"{prefix}.branch0", dtype)]
        for index in (1, 2):
            name = f"{prefix}.branch{index}"
            if f"{name}.0.conv.weight" in weights:
                branches.append(self._chain(network, x, weights, name, dtype))
        merged = graph_ops.add_concat(network, branches)

        proj_w = weights[f"{prefix}.conv2d.weight"]
        out = graph_ops.add_conv2d(
            network, merged, proj_w, weights[f"{prefix}.conv2d.bias"],
            int(proj_w.shape[0]), (1, 1), dtype=dtype)
        out = graph_ops.add_constant_scale(network, out, scale, dtype=dtype)
        out = graph_ops.add_sum(network, out, x)
        return graph_ops.add_relu(network, out) if activate else out

    def _mixed_5b(self, network, x, weights, dtype):
        b0 = self._conv(network, x, weights, "mixed_5b.branch0", dtype)
        b1 = self._chain(network, x, weights, "mixed_5b.branch1", dtype)
        b2 = self._chain(network, x, weights, "mixed_5b.branch2", dtype)
        b3 = graph_ops.add_avg_pool2d(network, x, 3, 1, 1)
        b3 = self._conv(network, b3, weights, "mixed_5b.branch3.1", dtype)
        return graph_ops.add_concat(network, [b0, b1, b2, b3])

    def _reduction(self, network, x, weights, prefix, dtype):
        """A reduction: strided branches beside a strided max pool."""
        branches = []
        if f"{prefix}.branch0.conv.weight" in weights:
            branches.append(self._conv(
                network, x, weights, f"{prefix}.branch0", dtype, stride=2, padding=(0, 0)))
        else:
            branches.append(self._strided_chain(network, x, weights, f"{prefix}.branch0", dtype))
        index = 1
        while f"{prefix}.branch{index}.0.conv.weight" in weights:
            branches.append(
                self._strided_chain(network, x, weights, f"{prefix}.branch{index}", dtype))
            index += 1
        branches.append(graph_ops.add_max_pool2d(network, x, 3, 2, 0))
        return graph_ops.add_concat(network, branches)

    def _strided_chain(self, network, x, weights, prefix, dtype):
        """A chain whose final convolution carries the stride."""
        steps = 0
        while f"{prefix}.{steps}.conv.weight" in weights:
            steps += 1
        out = x
        for step in range(steps):
            last = step == steps - 1
            out = self._conv(
                network, out, weights, f"{prefix}.{step}", dtype,
                stride=2 if last else 1, padding=(0, 0) if last else None)
        return out

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
                "timm_inception_resnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError(
                "timm_inception_resnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported timm_inception_resnet precision: {precision}")

        cfg = config.raw.get("_timm_inception_resnet_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        groups = cfg["groups"]

        if verbose:
            print(
                "[trtmc build] timm_inception_resnet: "
                f"image={image_h}x{image_w}, groups={groups}, "
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

        for name, stride, pad in _STEM:
            if name == "maxpool":
                hidden = graph_ops.add_max_pool2d(network, hidden, 3, 2, 0)
                continue
            hidden = self._conv(
                network, hidden, weights, name, work_np_dtype,
                stride=stride, padding=(pad, pad))

        hidden = self._mixed_5b(network, hidden, weights, work_np_dtype)
        for index in range(groups["repeat"]):
            hidden = self._residual_block(
                network, hidden, weights, f"repeat.{index}",
                _GROUP_SCALE["repeat"], work_np_dtype, activate=True)

        hidden = self._reduction(network, hidden, weights, "mixed_6a", work_np_dtype)
        for index in range(groups["repeat_1"]):
            hidden = self._residual_block(
                network, hidden, weights, f"repeat_1.{index}",
                _GROUP_SCALE["repeat_1"], work_np_dtype, activate=True)

        hidden = self._reduction(network, hidden, weights, "mixed_7a", work_np_dtype)
        for index in range(groups["repeat_2"]):
            hidden = self._residual_block(
                network, hidden, weights, f"repeat_2.{index}",
                _GROUP_SCALE["repeat_2"], work_np_dtype, activate=True)

        # The trailing block adds its branch unscaled and does not activate.
        hidden = self._residual_block(
            network, hidden, weights, "block8", _FINAL_BLOCK_SCALE, work_np_dtype,
            activate=False)

        hidden = self._conv(network, hidden, weights, "conv2d_7b", work_np_dtype)
        shape = hidden.shape
        hidden = graph_ops.add_global_avg_pool(
            network, hidden, (int(shape[2]), int(shape[3])))

        fc_w = weights["classif.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["classif.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_inception_resnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = (config.raw.get("_timm_inception_resnet_config")
               or _resolve_config(config.raw))
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


plugin = TimmInceptionResnetPlugin()
