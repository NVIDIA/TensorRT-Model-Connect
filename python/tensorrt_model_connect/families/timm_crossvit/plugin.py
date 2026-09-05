# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm CrossViT image-classification family plugin.

Supports timm CrossViT classifiers stored in HF Hub format. The initial target
is:
  timm/crossvit_9_240.in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX, matching the other timm families.

CrossViT runs two vision transformers side by side on the same image at two
patch sizes, so they see coarse and fine detail. They exchange information only
through their class tokens: each branch projects its class token into the other
branch's width, attends over the other branch's patch tokens, and projects the
result back. The patch tokens themselves never cross.
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

_LN_EPS = 1e-6
# Head count is not recoverable from the weights: every projection is square,
# so the head split leaves no trace in any tensor shape.
_NUM_HEADS = 4


def _pretrained_cfg(raw: dict) -> dict:
    nested = raw.get("pretrained_cfg")
    return nested if isinstance(nested, dict) else raw


def _resolve_config(raw: dict) -> dict:
    pcfg = _pretrained_cfg(raw)
    input_size = pcfg.get("input_size", [3, 240, 240])
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


def _count(names, pattern: str) -> int:
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

    branches = _count(names, r"^patch_embed\.(\d+)\.")
    if branches < 2:
        raise ValueError("CrossViT needs at least two branches")
    stages = _count(names, r"^blocks\.(\d+)\.")
    if stages == 0:
        raise ValueError("Checkpoint has no multi-scale blocks")

    depths = []
    for stage in range(stages):
        per_branch = []
        for branch in range(branches):
            depth = _count(names, rf"^blocks\.{stage}\.blocks\.{branch}\.(\d+)\.")
            if depth == 0:
                raise ValueError(f"blocks.{stage}.blocks.{branch} is empty")
            per_branch.append(depth)
        depths.append(per_branch)

    return {"branches": branches, "stages": stages, "depths": depths}


class TimmCrossvitPlugin:
    name = "timm_crossvit"
    runtime_strategy = "timm_crossvit_image_classification"
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        mt = (model_type or "").lower()
        return mt == "timm_crossvit" or mt.startswith("crossvit")

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
        raw["_timm_crossvit_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()
        tensor_map = getattr(readers, "tensor_map", None)
        names = set(tensor_map) if tensor_map is not None else {
            key for reader in readers for key in reader.keys()
        }
        for name in sorted(names):
            weights[name] = _load_tensor(readers, name).astype(target_dtype)

        for branch in range(cfg["branches"]):
            for key in (f"head.{branch}.weight", f"head.{branch}.bias"):
                if not _has_tensor(readers, key):
                    raise KeyError(f"Tensor not found: {key}")

        return weights

    def _layer_norm(self, network, x, weights, prefix, width, dtype):
        return graph_ops.add_layer_norm_native(
            network, x, width,
            np.asarray(weights[f"{prefix}.weight"], dtype=np.float32),
            np.asarray(weights[f"{prefix}.bias"], dtype=np.float32),
            _LN_EPS, dtype=dtype)

    def _linear(self, network, x, weights, prefix, dtype):
        w = weights[f"{prefix}.weight"]
        out_width, in_width = int(w.shape[0]), int(w.shape[1])
        out = graph_ops.add_matmul_rhs_constant(
            network, x, in_width, out_width, np.asarray(w).T, dtype=dtype)
        if f"{prefix}.bias" in weights:
            out = graph_ops.add_bias_sum(
                network, out, out_width, weights[f"{prefix}.bias"], dtype=dtype)
        return out

    def _split_heads(self, network, x, tokens, width):
        head_dim = width // _NUM_HEADS
        return graph_ops.add_permute_reshape(
            network, x, (1, tokens, _NUM_HEADS, head_dim), (0, 2, 1, 3), None)

    def _merge_heads(self, network, x, tokens, width):
        return graph_ops.add_permute_reshape(
            network, x, None, (0, 2, 1, 3), (1, tokens, width))

    def _self_attention(self, network, x, weights, prefix, dtype, *, tokens, width):
        qkv = self._linear(network, x, weights, f"{prefix}.qkv", dtype)
        head_dim = width // _NUM_HEADS
        qkv = graph_ops.add_permute_reshape(
            network, qkv, (1, tokens, 3, _NUM_HEADS, head_dim), (2, 0, 3, 1, 4), None)
        parts = []
        for which in range(3):
            piece = network.add_slice(
                qkv, trt.Dims((which, 0, 0, 0, 0)),
                trt.Dims((1, 1, _NUM_HEADS, tokens, head_dim)),
                trt.Dims((1, 1, 1, 1, 1))).get_output(0)
            parts.append(graph_ops.add_reshape(
                network, piece, (1, _NUM_HEADS, tokens, head_dim)))
        context = graph_ops.add_attention_core(network, parts[0], parts[1], parts[2])
        context = self._merge_heads(network, context, tokens, width)
        return self._linear(network, context, weights, f"{prefix}.proj", dtype)

    def _block(self, network, x, weights, prefix, dtype, *, tokens, width):
        hidden = self._layer_norm(network, x, weights, f"{prefix}.norm1", width, dtype)
        hidden = self._self_attention(
            network, hidden, weights, f"{prefix}.attn", dtype, tokens=tokens, width=width)
        x = network.add_elementwise(
            x, hidden, trt.ElementWiseOperation.SUM).get_output(0)

        hidden = self._layer_norm(network, x, weights, f"{prefix}.norm2", width, dtype)
        hidden = self._linear(network, hidden, weights, f"{prefix}.mlp.fc1", dtype)
        hidden = graph_ops.add_gelu_erf(network, hidden, dtype=dtype)
        hidden = self._linear(network, hidden, weights, f"{prefix}.mlp.fc2", dtype)
        return network.add_elementwise(
            x, hidden, trt.ElementWiseOperation.SUM).get_output(0)

    def _projection(self, network, x, weights, prefix, dtype, *, in_width):
        """LayerNorm, GELU, Linear: the class-token bridge between branches."""
        out = self._layer_norm(network, x, weights, f"{prefix}.0", in_width, dtype)
        out = graph_ops.add_gelu_erf(network, out, dtype=dtype)
        return self._linear(network, out, weights, f"{prefix}.2", dtype)

    def _cross_attention(self, network, x, weights, prefix, dtype, *, tokens, width):
        """Attention with a single query: the class token attends, nothing else.

        Only the first token produces a query, so the output is one token wide.
        """
        head_dim = width // _NUM_HEADS
        query_token = graph_ops.add_token_slice(network, x, 0, 1)
        q = self._linear(network, query_token, weights, f"{prefix}.wq", dtype)
        q = self._split_heads(network, q, 1, width)
        k = self._split_heads(
            network, self._linear(network, x, weights, f"{prefix}.wk", dtype),
            tokens, width)
        v = self._split_heads(
            network, self._linear(network, x, weights, f"{prefix}.wv", dtype),
            tokens, width)
        context = graph_ops.add_attention_core(network, q, k, v)
        context = self._merge_heads(network, context, 1, width)
        return self._linear(network, context, weights, f"{prefix}.proj", dtype)

    def _fusion_block(self, network, x, weights, prefix, dtype, *, tokens, width):
        """Cross-attention block: the residual keeps only the class token."""
        hidden = self._layer_norm(network, x, weights, f"{prefix}.norm1", width, dtype)
        hidden = self._cross_attention(
            network, hidden, weights, f"{prefix}.attn", dtype, tokens=tokens, width=width)
        query_token = graph_ops.add_token_slice(network, x, 0, 1)
        return network.add_elementwise(
            query_token, hidden, trt.ElementWiseOperation.SUM).get_output(0)

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
            raise NotImplementedError("timm_crossvit does not support quantized builds yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise NotImplementedError("timm_crossvit does not support tensor-parallel builds")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported timm_crossvit precision: {precision}")

        cfg = config.raw.get("_timm_crossvit_config")
        if cfg is None:
            raise RuntimeError(
                "load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_classes = cfg["num_classes"]
        branches = cfg["branches"]
        depths = cfg["depths"]

        # Each branch's own input size follows from its patch size and its
        # positional table: the table has one entry per patch plus the class
        # token, and the patches tile a square.
        geometry = []
        for branch in range(branches):
            patch_w = weights[f"patch_embed.{branch}.proj.weight"]
            patch = int(patch_w.shape[2])
            width = int(patch_w.shape[0])
            tokens = int(np.asarray(weights[f"pos_embed_{branch}"]).shape[1])
            grid = int(round(np.sqrt(tokens - 1)))
            if grid * grid != tokens - 1:
                raise ValueError(
                    f"branch {branch}: {tokens - 1} patches do not tile a square")
            geometry.append({"patch": patch, "width": width, "tokens": tokens,
                             "size": grid * patch})

        if verbose:
            print(
                "[trtmc build] timm_crossvit: "
                f"image={image_h}x{image_w}, "
                f"branches={[(g['size'], g['patch'], g['width']) for g in geometry]}, "
                f"depths={depths}, classes={num_classes}, precision={precision}",
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
        image = pixel_values
        if image.dtype != work_trt_dtype:
            image = network.add_cast(image, work_trt_dtype).get_output(0)

        states = []
        for branch, geo in enumerate(geometry):
            branch_image = image
            if geo["size"] != image_h or geo["size"] != image_w:
                # timm rescales rather than crops for these checkpoints.
                branch_image = graph_ops.add_bicubic_resize(
                    network, image, (geo["size"], geo["size"]))
            patches = graph_ops.add_patch_conv(
                network, branch_image,
                weights[f"patch_embed.{branch}.proj.weight"],
                weights[f"patch_embed.{branch}.proj.bias"],
                geo["width"], (geo["patch"], geo["patch"]), dtype=work_np_dtype)
            grid = geo["size"] // geo["patch"]
            patches = graph_ops.add_permute_reshape(
                network, patches, None, (0, 2, 3, 1),
                (1, grid * grid, geo["width"]))
            cls_token = graph_ops.add_constant(
                network, (1, 1, geo["width"]),
                np.asarray(weights[f"cls_token_{branch}"]).reshape(1, 1, geo["width"]),
                dtype=work_np_dtype)
            if cls_token.dtype != patches.dtype:
                cls_token = network.add_cast(cls_token, patches.dtype).get_output(0)
            tokens = graph_ops.add_token_concat(network, [cls_token, patches])
            pos = graph_ops.add_constant(
                network, (1, geo["tokens"], geo["width"]),
                np.asarray(weights[f"pos_embed_{branch}"]), dtype=work_np_dtype)
            if pos.dtype != tokens.dtype:
                pos = network.add_cast(pos, tokens.dtype).get_output(0)
            states.append(network.add_elementwise(
                tokens, pos, trt.ElementWiseOperation.SUM).get_output(0))

        for stage in range(cfg["stages"]):
            encoded = []
            for branch in range(branches):
                hidden = states[branch]
                for block in range(depths[stage][branch]):
                    hidden = self._block(
                        network, hidden, weights,
                        f"blocks.{stage}.blocks.{branch}.{block}", work_np_dtype,
                        tokens=geometry[branch]["tokens"],
                        width=geometry[branch]["width"])
                encoded.append(hidden)

            fused = []
            for branch in range(branches):
                other = (branch + 1) % branches
                other_width = geometry[other]["width"]
                # This branch's class token, projected into the other branch's
                # width and placed at the front of the other branch's patches.
                own_cls = graph_ops.add_token_slice(network, encoded[branch], 0, 1)
                projected = self._projection(
                    network, own_cls, weights, f"blocks.{stage}.projs.{branch}",
                    work_np_dtype, in_width=geometry[branch]["width"])
                other_patches = graph_ops.add_token_slice(
                    network, encoded[other], 1, geometry[other]["tokens"] - 1)
                merged = graph_ops.add_token_concat(
                    network, [projected, other_patches])
                attended = self._fusion_block(
                    network, merged, weights, f"blocks.{stage}.fusion.{branch}",
                    work_np_dtype, tokens=geometry[other]["tokens"],
                    width=other_width)
                reverted = self._projection(
                    network, attended, weights,
                    f"blocks.{stage}.revert_projs.{branch}", work_np_dtype,
                    in_width=other_width)
                own_patches = graph_ops.add_token_slice(
                    network, encoded[branch], 1, geometry[branch]["tokens"] - 1)
                fused.append(graph_ops.add_token_concat(
                    network, [reverted, own_patches]))
            states = fused

        # Each branch classifies from its own class token; the logits are then
        # averaged, so neither branch alone decides the answer.
        branch_logits = []
        for branch in range(branches):
            hidden = self._layer_norm(
                network, states[branch], weights, f"norm.{branch}",
                geometry[branch]["width"], work_np_dtype)
            hidden = graph_ops.add_token_slice(network, hidden, 0, 1)
            hidden = graph_ops.add_reshape(
                network, hidden, (1, geometry[branch]["width"]))
            branch_logits.append(
                self._linear(network, hidden, weights, f"head.{branch}", work_np_dtype))

        logits = graph_ops.add_scaled_sum(
            network, branch_logits, 1.0 / len(branch_logits), dtype=work_np_dtype)
        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        logits = graph_ops.add_reshape(network, logits, (1, num_classes))
        logits.name = "logits"
        network.mark_output(logits)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT timm_crossvit engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_timm_crossvit_config") or _resolve_config(config.raw)
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


plugin = TimmCrossvitPlugin()
