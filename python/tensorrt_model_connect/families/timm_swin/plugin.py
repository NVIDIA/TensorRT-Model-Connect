# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm Swin Transformer image-classification family plugin.

Supports timm Swin Transformer classifiers stored in HF Hub format. The initial
target is:
  timm/swin_tiny_patch4_window7_224.ms_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

Swin computes attention inside fixed windows rather than across the whole
feature map, and alternate blocks shift the window grid by half a window so the
windows overlap between blocks. The shift is done by cyclically rolling the
feature map, which wraps opposite edges into the same window; a per-window mask
then stops those unrelated positions from attending to each other.

Both the relative-position index and that mask are stored in the checkpoint, so
this plugin folds them into a single additive attention bias on the host rather
than rebuilding the geometry.
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

_LN_EPS = 1e-5
# Positions that the cyclic roll placed in the same window but that are not
# actually neighbours are pushed this far below the real scores.
_MASK_FILL = -100.0


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
        "crop_pct": float(pcfg.get("crop_pct", 0.9)),
        "interpolation": str(pcfg.get("interpolation", "bicubic")),
    }


def _count_indices(names, pattern: str) -> int:
    regex = re.compile(pattern)
    indices = {int(m.group(1)) for m in map(regex.match, names) if m}
    if not indices:
        return 0
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"Indices for {pattern} are not contiguous: {sorted(indices)}")
    return len(indices)


def _discover_layout(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    num_layers = _count_indices(names, r"^layers\.(\d+)\.")
    if num_layers == 0:
        raise ValueError("Checkpoint has no layers")
    depths = []
    for index in range(num_layers):
        depth = _count_indices(names, rf"^layers\.{index}\.blocks\.(\d+)\.")
        if depth == 0:
            raise ValueError(f"Checkpoint has no layers.{index} blocks")
        depths.append(depth)
    return {"num_layers": num_layers, "depths": depths}


def _attention_bias(index, table, mask, window_area, num_heads):
    """Fold the relative-position bias and the window mask into one tensor.

    Returns [num_windows, num_heads, window_area, window_area], the additive
    bias TensorRT's attention layer takes. Doing this on the host keeps the
    gather and the broadcast out of the engine, and the result is constant for
    a fixed input size.
    """
    bias = table[index.reshape(-1)].reshape(window_area, window_area, num_heads)
    bias = np.transpose(bias, (2, 0, 1))[None]
    if mask is None:
        return bias
    filled = np.where(mask != 0, _MASK_FILL, 0.0).astype(np.float32)
    return bias + filled[:, None]


class TimmSwinPlugin:
    name = "timm_swin"
    runtime_strategy = "timm_swin_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_swin" or mt.startswith("swin")

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
        raw["_timm_swin_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for name in sorted(names):
            tensor = _load_tensor(readers, name)
            if name.endswith("relative_position_index") or name.endswith("attn_mask"):
                # Geometry, not learned weights: it stays exact and is only
                # ever read on the host.
                weights[name] = tensor.astype(np.int64 if "index" in name else np.float32)
            else:
                weights[name] = tensor.astype(target_dtype)

        for key in ("head.fc.weight", "head.fc.bias"):
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")

        return weights

    def _layer_norm(self, network, x, weights, prefix, width, dtype):
        return graph_ops.add_layer_norm_native(
            network, x, width,
            np.asarray(weights[f"{prefix}.weight"], dtype=np.float32),
            np.asarray(weights[f"{prefix}.bias"], dtype=np.float32),
            _LN_EPS, dtype=dtype)

    def _linear(self, network, x, weights, prefix, dtype, *, bias=True):
        w = weights[f"{prefix}.weight"]
        out_width, in_width = int(w.shape[0]), int(w.shape[1])
        out = graph_ops.add_matmul_rhs_constant(
            network, x, in_width, out_width, np.asarray(w).T, dtype=dtype)
        if bias and f"{prefix}.bias" in weights:
            out = graph_ops.add_bias_sum(
                network, out, out_width, weights[f"{prefix}.bias"], dtype=dtype)
        return out

    def _window_attention(self, network, x, weights, prefix, dtype,
                          *, height, width, channels, window):
        """Window attention over an NHWC map whose sides divide the window."""
        windows_h, windows_w = height // window, width // window
        num_windows = windows_h * windows_w
        area = window * window

        table = np.asarray(weights[f"{prefix}.relative_position_bias_table"],
                           dtype=np.float32)
        num_heads = int(table.shape[1])
        head_dim = channels // num_heads
        index = np.asarray(weights[f"{prefix}.relative_position_index"], dtype=np.int64)
        mask_key = f"{prefix.rsplit('.', 1)[0]}.attn_mask"
        mask = weights.get(mask_key) if hasattr(weights, "get") else None
        if mask is not None:
            mask = np.asarray(mask, dtype=np.float32)
        bias = _attention_bias(index, table, mask, area, num_heads)
        if bias.shape[0] == 1 and num_windows != 1:
            bias = np.broadcast_to(bias, (num_windows, num_heads, area, area))

        # [1, H, W, C] -> [num_windows, area, C]
        tokens = graph_ops.add_permute_reshape(
            network, x,
            (windows_h, window, windows_w, window, channels),
            (0, 2, 1, 3, 4),
            (num_windows, area, channels))

        qkv = self._linear(network, tokens, weights, f"{prefix}.qkv", dtype)
        qkv = graph_ops.add_permute_reshape(
            network, qkv,
            (num_windows, area, 3, num_heads, head_dim),
            (2, 0, 3, 1, 4),
            None)
        parts = []
        for which in range(3):
            piece = network.add_slice(
                qkv,
                trt.Dims((which, 0, 0, 0, 0)),
                trt.Dims((1, num_windows, num_heads, area, head_dim)),
                trt.Dims((1, 1, 1, 1, 1))).get_output(0)
            parts.append(graph_ops.add_reshape(
                network, piece, (num_windows, num_heads, area, head_dim)))

        bias_t = graph_ops.add_constant(
            network, (num_windows, num_heads, area, area), bias, dtype=dtype)
        if bias_t.dtype != parts[0].dtype:
            bias_t = network.add_cast(bias_t, parts[0].dtype).get_output(0)
        context = graph_ops.add_attention_core(
            network, parts[0], parts[1], parts[2], mask=bias_t)

        # [num_windows, heads, area, head_dim] -> [1, H, W, C]
        merged = graph_ops.add_permute_reshape(
            network, context, None, (0, 2, 1, 3), (num_windows, area, channels))
        merged = self._linear(network, merged, weights, f"{prefix}.proj", dtype)
        return graph_ops.add_permute_reshape(
            network, merged,
            (windows_h, windows_w, window, window, channels),
            (0, 2, 1, 3, 4),
            (1, height, width, channels))

    def _block(self, network, x, weights, prefix, dtype,
               *, height, width, channels, window):
        shift = window // 2 if f"{prefix}.attn_mask" in weights else 0

        residual = x
        hidden = self._layer_norm(network, x, weights, f"{prefix}.norm1", channels, dtype)
        if shift:
            hidden = graph_ops.add_roll(network, hidden, (-shift, -shift), (1, 2))
        hidden = self._window_attention(
            network, hidden, weights, f"{prefix}.attn", dtype,
            height=height, width=width, channels=channels, window=window)
        if shift:
            hidden = graph_ops.add_roll(network, hidden, (shift, shift), (1, 2))
        x = network.add_elementwise(
            residual, hidden, trt.ElementWiseOperation.SUM).get_output(0)

        residual = x
        hidden = self._layer_norm(network, x, weights, f"{prefix}.norm2", channels, dtype)
        hidden = self._linear(network, hidden, weights, f"{prefix}.mlp.fc1", dtype)
        hidden = graph_ops.add_gelu_erf(network, hidden, dtype=dtype)
        hidden = self._linear(network, hidden, weights, f"{prefix}.mlp.fc2", dtype)
        return network.add_elementwise(
            residual, hidden, trt.ElementWiseOperation.SUM).get_output(0)

    def _patch_merging(self, network, x, weights, prefix, dtype,
                       *, height, width, channels):
        """Fold each 2x2 patch into the channel axis, then project it down.

        The four positions are interleaved column-inner-then-row-inner, which
        is the order the reduction weights were trained against; swapping them
        produces a plausible but wrong result.
        """
        folded = graph_ops.add_permute_reshape(
            network, x,
            (height // 2, 2, width // 2, 2, channels),
            (0, 2, 3, 1, 4),
            (1, height // 2, width // 2, 4 * channels))
        folded = self._layer_norm(
            network, folded, weights, f"{prefix}.norm", 4 * channels, dtype)
        return self._linear(
            network, folded, weights, f"{prefix}.reduction", dtype, bias=False)

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
            raise NotImplementedError("timm_swin does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_swin does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_swin precision: {precision}")

        cfg = config.raw.get("_timm_swin_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        depths = cfg["depths"]

        patch_w = weights["patch_embed.proj.weight"]
        patch = (int(patch_w.shape[2]), int(patch_w.shape[3]))
        embed_dim = int(patch_w.shape[0])
        height, width = image_h // patch[0], image_w // patch[1]
        # relative_position_index is [window_area, window_area], and a window is
        # square, so the side length is the fourth root of its element count.
        window_area = int(round(np.sqrt(
            np.asarray(weights["layers.0.blocks.0.attn.relative_position_index"]).size)))
        window = int(round(np.sqrt(window_area)))
        if window * window != window_area:
            raise ValueError(f"relative_position_index implies a non-square window: {window_area}")

        if verbose:
            print(
                "[trtmc build] timm_swin: "
                f"image={image_h}x{image_w}, patch={patch}, embed={embed_dim}, "
                f"depths={depths}, window={window}, classes={num_classes}, "
                f"precision={precision}",
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

        hidden = graph_ops.add_patch_conv(
            network, hidden, patch_w, weights["patch_embed.proj.bias"],
            embed_dim, patch, dtype=work_np_dtype)
        # NCHW -> NHWC: the rest of the model indexes tokens spatially.
        hidden = graph_ops.add_permute_reshape(
            network, hidden, None, (0, 2, 3, 1), None)
        hidden = self._layer_norm(
            network, hidden, weights, "patch_embed.norm", embed_dim, work_np_dtype)

        channels = embed_dim
        for layer, depth in enumerate(depths):
            if f"layers.{layer}.downsample.reduction.weight" in weights:
                hidden = self._patch_merging(
                    network, hidden, weights, f"layers.{layer}.downsample",
                    work_np_dtype, height=height, width=width, channels=channels)
                height, width = height // 2, width // 2
                channels = int(
                    weights[f"layers.{layer}.downsample.reduction.weight"].shape[0])
            for block in range(depth):
                hidden = self._block(
                    network, hidden, weights, f"layers.{layer}.blocks.{block}",
                    work_np_dtype, height=height, width=width, channels=channels,
                    # The last stage is one window wide, so it cannot shift.
                    window=min(window, height, width))

        hidden = self._layer_norm(network, hidden, weights, "norm", channels, work_np_dtype)
        hidden = graph_ops.add_global_avg_pool_nhwc(network, hidden)
        logits = self._linear(network, hidden, weights, "head.fc", work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits = graph_ops.add_reshape(network, logits, (1, num_classes))
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_swin engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_swin_config") or _resolve_config(config.raw)
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


plugin = TimmSwinPlugin()
