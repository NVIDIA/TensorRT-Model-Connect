# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ResNet image-classification family plugin.

Supports timm ResNet/ResNeXt classifiers stored in HF Hub format. The initial
target is:
  timm/resnet50.a1_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the timm_vit family.

The stage/block layout is recovered from the checkpoint rather than from a
hardcoded depth table, so resnet18/34/50/101/152 and the ResNeXt grouped
variants all build from the same code path.
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
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
)
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

# timm's ResNet BatchNorm2d layers use the PyTorch default epsilon; it is not
# recorded in config.json.
_BN_EPS = 1e-5

_STAGES = ("layer1", "layer2", "layer3", "layer4")


def _pretrained_cfg(raw: dict) -> dict:
    """timm nests preprocessing under pretrained_cfg; older exports inline it."""
    nested = raw.get("pretrained_cfg")
    return nested if isinstance(nested, dict) else raw


def _resolve_resnet_config(raw: dict) -> dict:
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
        "crop_pct": float(pcfg.get("crop_pct", 0.875)),
        "interpolation": str(pcfg.get("interpolation", "bilinear")),
    }


def _discover_layout(readers) -> dict:
    """Recover block counts, block type, and group width from the checkpoint."""
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    blocks: list[int] = []
    for stage in _STAGES:
        indices = set()
        pattern = re.compile(rf"^{stage}\.(\d+)\.")
        for name in names:
            match = pattern.match(name)
            if match:
                indices.add(int(match.group(1)))
        if not indices:
            raise ValueError(f"Checkpoint has no blocks for stage {stage}")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"Stage {stage} block indices are not contiguous")
        blocks.append(len(indices))

    bottleneck = "layer1.0.conv3.weight" in names
    return {"blocks": blocks, "bottleneck": bottleneck}


class _TimmResnetModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        readers = _open_safetensors(Path(model_dir))
        raw = config.raw
        resnet_cfg = _resolve_resnet_config(raw)
        layout = _discover_layout(readers)
        resnet_cfg.update(layout)
        raw["_timm_resnet_config"] = resnet_cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        def take_conv(key: str) -> None:
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        def take_bn(prefix: str) -> None:
            # BN statistics stay fp32: they are folded into a scale/shift on the
            # host, and folding in fp16 loses precision in 1/sqrt(var + eps).
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                key = f"{prefix}.{suffix}"
                weights[key] = _load_tensor(readers, key).astype(np.float32)

        take_conv("conv1.weight")
        take_bn("bn1")

        convs = ("conv1", "conv2", "conv3") if layout["bottleneck"] else ("conv1", "conv2")
        bns = ("bn1", "bn2", "bn3") if layout["bottleneck"] else ("bn1", "bn2")
        for stage, count in zip(_STAGES, layout["blocks"]):
            for block in range(count):
                prefix = f"{stage}.{block}"
                for conv in convs:
                    take_conv(f"{prefix}.{conv}.weight")
                for bn in bns:
                    take_bn(f"{prefix}.{bn}")
                if _has_tensor(readers, f"{prefix}.downsample.0.weight"):
                    take_conv(f"{prefix}.downsample.0.weight")
                    take_bn(f"{prefix}.downsample.1")

        for key in ("fc.weight", "fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
            weights[key] = _load_tensor(readers, key).astype(target_dtype)

        return weights

    def _add_bn(self, network, x, weights, prefix, dtype):
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

    def _add_block(self, network, x, weights, prefix, stride, bottleneck, dtype):
        """One residual block. timm places the spatial stride on the 3x3 conv."""
        identity = x

        if bottleneck:
            w1 = weights[f"{prefix}.conv1.weight"]
            out = graph_ops.add_conv2d(
                network, x, w1, None, int(w1.shape[0]), (1, 1), dtype=dtype)
            out = self._add_bn(network, out, weights, f"{prefix}.bn1", dtype)
            out = graph_ops.add_relu(network, out)

            w2 = weights[f"{prefix}.conv2.weight"]
            # Grouped (ResNeXt) convs store (out, in/groups, kh, kw).
            groups = max(1, int(w1.shape[0]) // int(w2.shape[1]))
            out = graph_ops.add_conv2d(
                network, out, w2, None, int(w2.shape[0]), (3, 3),
                stride=(stride, stride), padding=(1, 1), groups=groups, dtype=dtype)
            out = self._add_bn(network, out, weights, f"{prefix}.bn2", dtype)
            out = graph_ops.add_relu(network, out)

            w3 = weights[f"{prefix}.conv3.weight"]
            out = graph_ops.add_conv2d(
                network, out, w3, None, int(w3.shape[0]), (1, 1), dtype=dtype)
            out = self._add_bn(network, out, weights, f"{prefix}.bn3", dtype)
        else:
            w1 = weights[f"{prefix}.conv1.weight"]
            out = graph_ops.add_conv2d(
                network, x, w1, None, int(w1.shape[0]), (3, 3),
                stride=(stride, stride), padding=(1, 1), dtype=dtype)
            out = self._add_bn(network, out, weights, f"{prefix}.bn1", dtype)
            out = graph_ops.add_relu(network, out)

            w2 = weights[f"{prefix}.conv2.weight"]
            out = graph_ops.add_conv2d(
                network, out, w2, None, int(w2.shape[0]), (3, 3),
                padding=(1, 1), dtype=dtype)
            out = self._add_bn(network, out, weights, f"{prefix}.bn2", dtype)

        down_key = f"{prefix}.downsample.0.weight"
        if down_key in weights:
            wd = weights[down_key]
            identity = graph_ops.add_conv2d(
                network, x, wd, None, int(wd.shape[0]), (1, 1),
                stride=(stride, stride), dtype=dtype)
            identity = self._add_bn(
                network, identity, weights, f"{prefix}.downsample.1", dtype)

        out = graph_ops.add_sum(network, out, identity)
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
            raise NotImplementedError(
                "timm_resnet does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError(
                "timm_resnet does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_resnet precision: {precision}")

        cfg = config.raw.get("_timm_resnet_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        blocks = cfg["blocks"]
        bottleneck = cfg["bottleneck"]

        # Stem halves twice (7x7 stride 2, then max pool stride 2) and stages
        # 2..4 halve once each, so the pooled map is input / 32.
        if image_h % 32 != 0 or image_w % 32 != 0:
            raise ValueError(
                f"timm_resnet input {image_h}x{image_w} must be divisible by 32")
        feat_h, feat_w = image_h // 32, image_w // 32

        if verbose:
            print(
                "[trtmc build] timm_resnet: "
                f"image={image_h}x{image_w}, blocks={blocks}, "
                f"bottleneck={bottleneck}, classes={num_classes}, "
                f"precision={precision}",
                file=sys.stderr,
            )

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
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

        stem_w = weights["conv1.weight"]
        hidden = graph_ops.add_conv2d(
            network, hidden, stem_w, None, int(stem_w.shape[0]), (7, 7),
            stride=(2, 2), padding=(3, 3), dtype=work_np_dtype)
        hidden = self._add_bn(network, hidden, weights, "bn1", work_np_dtype)
        hidden = graph_ops.add_relu(network, hidden)
        hidden = graph_ops.add_max_pool2d(network, hidden, 3, 2, 1)

        for stage_idx, (stage, count) in enumerate(zip(_STAGES, blocks)):
            for block in range(count):
                # First block of every stage after layer1 downsamples.
                stride = 2 if (stage_idx > 0 and block == 0) else 1
                hidden = self._add_block(
                    network, hidden, weights, f"{stage}.{block}",
                    stride, bottleneck, work_np_dtype)

        hidden = graph_ops.add_global_avg_pool(network, hidden, (feat_h, feat_w))

        fc_w = weights["fc.weight"]
        logits = graph_ops.add_fc(
            network,
            hidden,
            int(fc_w.shape[1]),
            num_classes,
            fc_w,
            weights["fc.bias"],
            dtype=work_np_dtype,
        )
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_resnet engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_resnet_config") or _resolve_resnet_config(config.raw)
        return {
            "model_type": config.model_type,
            "input_image_h": cfg["image_size_h"],
            "input_image_w": cfg["image_size_w"],
            "num_classes": cfg["num_classes"],
            "image_mean": cfg["mean"],
            "image_std": cfg["std"],
            "crop_pct": cfg["crop_pct"],
            "interpolation": cfg["interpolation"],
        }


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm ResNet image-classification bundle."""
    if request.image_height is not None:
        raise NotImplementedError("timm_resnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_resnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_resnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_resnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_resnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_resnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_resnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_resnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_resnet does not support mixed-precision layers")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    model_type = str(config.model_type).lower()
    if not model_type.startswith(("resnet", "resnext", "wide_resnet")):
        raise ValueError(f"timm ResNet does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    max_sequence_length = _positive_int(
        request.max_sequence_length or 1, "max_sequence_length"
    )
    model = _TimmResnetModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        max_sequence_length,
        precision=precision,
        quant_ctx=None,
        verbose=bool(request.verbose),
        parallel_config=None,
    )
    writer.set_header(family="timm_resnet", task=request.task, backend="trt")
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
