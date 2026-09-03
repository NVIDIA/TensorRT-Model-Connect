# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm Xception image-classification family plugin.

Supports timm's aligned Xception classifiers stored in HF Hub format. The
initial target is:
  timm/xception41.tf_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Most of the layout is recovered from the checkpoint: the block count from the
`blocks.<n>` keys, and the stride from whether a block carries a projection
shortcut, since Xception downsamples exactly in the blocks that project.

One thing is not recoverable, because activations carry no weights: the final
block is built differently from the rest. Blocks before it apply a ReLU *before*
each separable convolution and none inside, add a residual, and the last block
inverts both - it has no residual and applies its activations inside the
separable convolutions instead. That is keyed on the block being last, and
checked against timm.
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

# Xception uses the TensorFlow batch-norm epsilon, not the PyTorch default.
_BN_EPS = 1e-3

_STACK_CONVS = ("conv1", "conv2", "conv3")


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
        "crop_pct": float(pcfg.get("crop_pct", 0.903)),
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

    leaves: dict[int, set[str]] = {}
    pattern = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for name in names:
        match = pattern.match(name)
        if match:
            leaves.setdefault(int(match.group(1)), set()).add(match.group(2).split(".")[0])
    if not leaves:
        raise ValueError("Checkpoint has no blocks.<index> entries")

    indices = sorted(leaves)
    if indices != list(range(len(indices))):
        raise ValueError("Block indices are not contiguous")

    blocks = []
    for index in indices:
        present = leaves[index]
        if "stack" not in present:
            raise ValueError(f"blocks.{index} has no separable convolution stack")
        has_shortcut = "shortcut" in present
        blocks.append(
            {
                "prefix": f"blocks.{index}",
                "has_shortcut": has_shortcut,
                # Xception downsamples exactly in the blocks that project.
                "stride": 2 if has_shortcut else 1,
                # The final block carries its activations inside the separable
                # convolutions and takes no residual.
                "is_exit": index == indices[-1],
            }
        )
    return {"blocks": blocks, "num_blocks": len(blocks)}


class TimmXceptionPlugin:
    name = "timm_xception"
    runtime_strategy = "timm_xception_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        if mt == "timm_xception":
            return True
        return mt.startswith("xception")

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
        raw["_timm_xception_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def bn(prefix: str) -> None:
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        def conv_bn(prefix: str) -> None:
            weights[f"{prefix}.conv.weight"] = _load_tensor(
                readers, f"{prefix}.conv.weight").astype(target_dtype)
            bn(f"{prefix}.bn")

        def separable(prefix: str) -> None:
            """Depthwise 3x3 then pointwise 1x1, each with its own norm."""
            for leaf in ("conv_dw", "conv_pw"):
                weights[f"{prefix}.{leaf}.weight"] = _load_tensor(
                    readers, f"{prefix}.{leaf}.weight").astype(target_dtype)
            bn(f"{prefix}.bn_dw")
            bn(f"{prefix}.bn_pw")

        conv_bn("stem.0")
        conv_bn("stem.1")

        for block in layout["blocks"]:
            prefix = block["prefix"]
            for leaf in _STACK_CONVS:
                separable(f"{prefix}.stack.{leaf}")
            if block["has_shortcut"]:
                conv_bn(f"{prefix}.shortcut")

        for key in ("head.fc.weight", "head.fc.bias"):
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

    def _conv_bn_act(self, network, x, weights, prefix, dtype, *, stride=1):
        w = weights[f"{prefix}.conv.weight"]
        out = graph_ops.add_conv2d(
            network, x, w, None, int(w.shape[0]),
            (int(w.shape[2]), int(w.shape[3])),
            stride=(stride, stride), padding=(1, 1) if w.shape[2] == 3 else (0, 0),
            dtype=dtype)
        return self._bn(network, out, weights, f"{prefix}.bn", dtype)

    def _separable(self, network, x, weights, prefix, dtype, *, stride=1, inner_act: bool):
        """conv_dw, bn_dw, [act], conv_pw, bn_pw, [act]."""
        dw = weights[f"{prefix}.conv_dw.weight"]
        out = graph_ops.add_conv2d(
            network, x, dw, None, int(dw.shape[0]), (3, 3),
            stride=(stride, stride), padding=(1, 1),
            groups=int(dw.shape[0]), dtype=dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn_dw", dtype)
        if inner_act:
            out = graph_ops.add_relu(network, out)

        pw = weights[f"{prefix}.conv_pw.weight"]
        out = graph_ops.add_conv2d(
            network, out, pw, None, int(pw.shape[0]), (1, 1), dtype=dtype)
        out = self._bn(network, out, weights, f"{prefix}.bn_pw", dtype)
        if inner_act:
            out = graph_ops.add_relu(network, out)
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
            raise NotImplementedError("timm_xception does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_xception does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_xception precision: {precision}")

        cfg = config.raw.get("_timm_xception_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]

        # No divisibility guard here: Xception's strided convolutions use
        # padding 1, so an odd input such as the standard 299x299 halves
        # cleanly to 150, 75, 38 and so on. The pooled size is read back from
        # the built network instead of being computed.

        if verbose:
            print(
                "[trtmc build] timm_xception: "
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

        hidden = self._conv_bn_act(network, hidden, weights, "stem.0", work_np_dtype, stride=2)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = self._conv_bn_act(network, hidden, weights, "stem.1", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)

        for block in blocks:
            prefix = block["prefix"]
            stride = block["stride"]
            is_exit = block["is_exit"]
            skip = hidden
            out = hidden

            for position, leaf in enumerate(_STACK_CONVS):
                # Only the third separable convolution carries the stride.
                conv_stride = stride if position == 2 else 1
                if not is_exit:
                    # A leading ReLU, and no activation inside the separable conv.
                    out = graph_ops.add_relu(network, out)
                out = self._separable(
                    network, out, weights, f"{prefix}.stack.{leaf}", work_np_dtype,
                    stride=conv_stride, inner_act=is_exit)

            if block["has_shortcut"]:
                skip = self._conv_bn_act(
                    network, skip, weights, f"{prefix}.shortcut", work_np_dtype,
                    stride=stride)
            if not is_exit:
                out = graph_ops.add_sum(network, out, skip)
            hidden = out

        shape = hidden.shape
        hidden = graph_ops.add_global_avg_pool(
            network, hidden, (int(shape[2]), int(shape[3])))

        fc_w = weights["head.fc.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["head.fc.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_xception engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_xception_config") or _resolve_config(config.raw)
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


plugin = TimmXceptionPlugin()
