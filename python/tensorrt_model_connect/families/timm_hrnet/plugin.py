# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm HRNet image-classification family plugin.

Supports timm HRNet classifiers stored in HF Hub format. The initial target is:
  timm/hrnet_w18.ms_aug_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

HRNet keeps several resolutions alive at once instead of reducing to one. Each
stage runs one branch per resolution and then fuses every branch into every
other, so the graph is a grid rather than a chain. The branch counts, module
counts, and block counts all come from the checkpoint keys.
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

# The stages, in order, with the transition that feeds each one.
_STAGES = (("transition1", "stage2"), ("transition2", "stage3"),
           ("transition3", "stage4"))


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
        "num_features": int(raw.get("num_features", 2048)),
        "mean": [float(v) for v in pcfg.get("mean", [0.485, 0.456, 0.406])],
        "std": [float(v) for v in pcfg.get("std", [0.229, 0.224, 0.225])],
        "crop_pct": float(pcfg.get("crop_pct", 0.95)),
        "interpolation": str(pcfg.get("interpolation", "bilinear")),
    }


def _count_indices(names, pattern: str) -> int:
    """Count contiguous numeric children matching a key pattern."""
    regex = re.compile(pattern)
    indices = {int(m.group(1)) for m in map(regex.match, names) if m}
    if not indices:
        return 0
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"Indices for {pattern} are not contiguous: {sorted(indices)}")
    return len(indices)


def _discover_layout(readers) -> dict:
    """Read the stage grid out of the checkpoint key names.

    Transitions are not counted directly: an identity transition carries no
    weights, so its index leaves no trace. The branch count of the stage the
    transition feeds is authoritative instead.
    """
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    layer1_blocks = _count_indices(names, r"^layer1\.(\d+)\.")
    if layer1_blocks == 0:
        raise ValueError("Checkpoint has no layer1 blocks")

    stages = []
    for _, stage in _STAGES:
        modules = _count_indices(names, rf"^{stage}\.(\d+)\.")
        if modules == 0:
            raise ValueError(f"Checkpoint has no {stage} modules")
        branches = _count_indices(names, rf"^{stage}\.0\.branches\.(\d+)\.")
        if branches == 0:
            raise ValueError(f"Checkpoint has no {stage} branches")
        blocks = [
            _count_indices(names, rf"^{stage}\.0\.branches\.{b}\.(\d+)\.")
            for b in range(branches)
        ]
        stages.append({"name": stage, "modules": modules,
                       "branches": branches, "blocks": blocks})

    incre = _count_indices(names, r"^incre_modules\.(\d+)\.")
    if incre != stages[-1]["branches"]:
        raise ValueError(
            f"incre_modules count {incre} does not match final branch count "
            f"{stages[-1]['branches']}")

    return {"layer1_blocks": layer1_blocks, "stages": stages, "head_inputs": incre}


class TimmHrnetPlugin:
    name = "timm_hrnet"
    runtime_strategy = "timm_hrnet_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_hrnet" or mt.startswith("hrnet")

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
        raw["_timm_hrnet_config"] = cfg
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
            if tensor.ndim == 1 and not name.startswith("classifier."):
                # Norm statistics stay fp32: the fold divides by their variance.
                weights[name] = tensor.astype(np.float32)
            else:
                weights[name] = tensor.astype(target_dtype)

        for key in ("classifier.weight", "classifier.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")

        return weights

    def _conv_bn(self, network, x, weights, conv_key, bn_prefix, dtype,
                 *, stride=1, padding=None, bias_key=None):
        w = weights[conv_key]
        kernel = (int(w.shape[2]), int(w.shape[3]))
        pad = ((kernel[0] // 2, kernel[1] // 2) if padding is None else padding)
        bias = weights[bias_key] if bias_key and bias_key in weights else None
        out = graph_ops.add_conv2d(
            network, x, w, bias, int(w.shape[0]), kernel,
            stride=(stride, stride), padding=pad, dtype=dtype)
        return graph_ops.add_batch_norm(
            network, out,
            weights[f"{bn_prefix}.weight"], weights[f"{bn_prefix}.bias"],
            weights[f"{bn_prefix}.running_mean"], weights[f"{bn_prefix}.running_var"],
            _BN_EPS, dtype=dtype)

    def _seq_conv_bn(self, network, x, weights, prefix, dtype, *, stride=1):
        """A bare nn.Sequential(Conv2d, BatchNorm2d[, ReLU]) block."""
        bias_key = f"{prefix}.0.bias"
        return self._conv_bn(
            network, x, weights, f"{prefix}.0.weight", f"{prefix}.1", dtype,
            stride=stride, bias_key=bias_key)

    def _residual(self, network, x, weights, prefix, dtype, *, bottleneck):
        """A BasicBlock or a Bottleneck, with an optional projection shortcut."""
        if bottleneck:
            out = self._conv_bn(
                network, x, weights, f"{prefix}.conv1.weight", f"{prefix}.bn1", dtype)
            out = graph_ops.add_relu(network, out)
            out = self._conv_bn(
                network, out, weights, f"{prefix}.conv2.weight", f"{prefix}.bn2", dtype)
            out = graph_ops.add_relu(network, out)
            out = self._conv_bn(
                network, out, weights, f"{prefix}.conv3.weight", f"{prefix}.bn3", dtype)
        else:
            out = self._conv_bn(
                network, x, weights, f"{prefix}.conv1.weight", f"{prefix}.bn1", dtype)
            out = graph_ops.add_relu(network, out)
            out = self._conv_bn(
                network, out, weights, f"{prefix}.conv2.weight", f"{prefix}.bn2", dtype)

        shortcut = x
        if f"{prefix}.downsample.0.weight" in weights:
            shortcut = self._seq_conv_bn(
                network, x, weights, f"{prefix}.downsample", dtype)
        return graph_ops.add_relu(network, graph_ops.add_sum(network, out, shortcut))

    def _transition(self, network, weights, prefix, inputs, branches, dtype):
        """Widen the branch list to `branches` entries.

        Existing branches either pass through untouched or go through one
        3x3 convolution; every new branch is built by halving the resolution of
        the last existing branch repeatedly.
        """
        outputs = []
        for index in range(branches):
            if index < len(inputs):
                if f"{prefix}.{index}.0.weight" in weights:
                    out = self._seq_conv_bn(
                        network, inputs[index], weights, f"{prefix}.{index}", dtype)
                    outputs.append(graph_ops.add_relu(network, out))
                else:
                    outputs.append(inputs[index])
                continue
            out = inputs[-1]
            step = 0
            while f"{prefix}.{index}.{step}.0.weight" in weights:
                out = self._seq_conv_bn(
                    network, out, weights, f"{prefix}.{index}.{step}", dtype, stride=2)
                out = graph_ops.add_relu(network, out)
                step += 1
            if step == 0:
                raise ValueError(f"{prefix}.{index}: no downsampling convolutions")
            outputs.append(out)
        return outputs

    def _fuse(self, network, weights, prefix, inputs, dtype):
        """Sum every branch into every branch, rescaling to match resolutions.

        A higher-index branch is at a lower resolution, so it is projected with
        a 1x1 convolution and upsampled; a lower-index branch is downsampled by
        a chain of strided 3x3 convolutions, and only the last of those omits
        the activation because its output feeds the sum.
        """
        outputs = []
        for i in range(len(inputs)):
            total = None
            for j, source in enumerate(inputs):
                if j == i:
                    term = source
                elif j > i:
                    term = self._seq_conv_bn(
                        network, source, weights, f"{prefix}.{i}.{j}", dtype)
                    term = graph_ops.add_nearest_upsample(network, term, 2 ** (j - i))
                else:
                    term = source
                    for step in range(i - j):
                        term = self._seq_conv_bn(
                            network, term, weights, f"{prefix}.{i}.{j}.{step}", dtype,
                            stride=2)
                        if step != i - j - 1:
                            term = graph_ops.add_relu(network, term)
                total = term if total is None else graph_ops.add_sum(network, total, term)
            outputs.append(graph_ops.add_relu(network, total))
        return outputs

    def _stage(self, network, weights, stage, inputs, dtype):
        current = inputs
        for module in range(stage["modules"]):
            prefix = f"{stage['name']}.{module}"
            branch_outputs = []
            for b, num_blocks in enumerate(stage["blocks"]):
                out = current[b]
                for block in range(num_blocks):
                    out = self._residual(
                        network, out, weights, f"{prefix}.branches.{b}.{block}",
                        dtype, bottleneck=False)
                branch_outputs.append(out)
            current = self._fuse(
                network, weights, f"{prefix}.fuse_layers", branch_outputs, dtype)
        return current

    def _head(self, network, weights, branches, dtype):
        """Collapse the branches into one tensor at the lowest resolution.

        Each branch gets its own Bottleneck, then the running sum is
        downsampled to meet the next branch. The downsampling convolutions and
        the final projection carry a bias, unlike the rest of the model.
        """
        total = None
        for index, source in enumerate(branches):
            widened = self._residual(
                network, source, weights, f"incre_modules.{index}.0", dtype,
                bottleneck=True)
            if total is None:
                total = widened
                continue
            down = self._seq_conv_bn(
                network, total, weights, f"downsamp_modules.{index - 1}", dtype,
                stride=2)
            down = graph_ops.add_relu(network, down)
            total = graph_ops.add_sum(network, widened, down)
        out = self._seq_conv_bn(network, total, weights, "final_layer", dtype)
        return graph_ops.add_relu(network, out)

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
            raise NotImplementedError("timm_hrnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError(
                "timm_hrnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_hrnet precision: {precision}")

        cfg = config.raw.get("_timm_hrnet_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        stages = cfg["stages"]

        if verbose:
            print(
                "[trtmc build] timm_hrnet: "
                f"image={image_h}x{image_w}, "
                f"branches={[s['branches'] for s in stages]}, "
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

        # Stem: two strided 3x3 convolutions take the input to one quarter size.
        hidden = self._conv_bn(
            network, hidden, weights, "conv1.weight", "bn1", work_np_dtype, stride=2)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = self._conv_bn(
            network, hidden, weights, "conv2.weight", "bn2", work_np_dtype, stride=2)
        hidden = graph_ops.add_relu(network, hidden)

        for block in range(cfg["layer1_blocks"]):
            hidden = self._residual(
                network, hidden, weights, f"layer1.{block}", work_np_dtype,
                bottleneck=True)

        branches = [hidden]
        for (transition, _), stage in zip(_STAGES, stages):
            branches = self._transition(
                network, weights, transition, branches, stage["branches"],
                work_np_dtype)
            branches = self._stage(network, weights, stage, branches, work_np_dtype)

        hidden = self._head(network, weights, branches, work_np_dtype)
        shape = hidden.shape
        hidden = graph_ops.add_global_avg_pool(
            network, hidden, (int(shape[2]), int(shape[3])))

        fc_w = weights["classifier.weight"]
        logits = graph_ops.add_fc(
            network, hidden, int(fc_w.shape[1]), num_classes,
            fc_w, weights["classifier.bias"], dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_hrnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_hrnet_config") or _resolve_config(config.raw)
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


plugin = TimmHrnetPlugin()
