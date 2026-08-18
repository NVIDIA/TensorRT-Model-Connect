# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ViT image-classification family model.

Supports fixed-size timm Vision Transformer classifiers stored in HF Hub
format. The initial target is:
  timm/vit_base_patch16_224.augreg_in21k_ft_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX.
"""

from __future__ import annotations

import json
import re
import time

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
    _transpose_2d,
)
from .config import ModelConfig
from ...parallel_config import (
    ParallelConfig,
    add_all_reduce_sum,
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


trt = trt_compat.get_trt()


graph_ops = sys.modules[__name__]


def _as_tuple2(value, default: int) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    return default, default


def _resolve_vit_config(raw: dict) -> dict:
    input_size = raw.get("input_size", [3, 224, 224])
    if isinstance(input_size, int):
        image_size_h, image_size_w = _as_tuple2(input_size, 224)
    else:
        image_size_h, image_size_w = _as_tuple2(input_size[-2:], 224)
    patch_h, patch_w = _as_tuple2(raw.get("patch_size", 16), 16)
    hidden = int(raw.get("embed_dim", raw.get("num_features", 768)))
    depth = int(raw.get("depth", raw.get("num_hidden_layers", 12)))
    heads = int(raw.get("num_heads", raw.get("num_attention_heads", 12)))
    mlp_hidden = int(float(raw.get("mlp_ratio", 4.0)) * hidden)
    return {
        "image_size_h": image_size_h,
        "image_size_w": image_size_w,
        "patch_h": patch_h,
        "patch_w": patch_w,
        "hidden": hidden,
        "depth": depth,
        "heads": heads,
        "mlp_hidden": int(raw.get("intermediate_size", mlp_hidden)),
        "num_classes": int(raw.get("num_classes", 1000)),
        "eps": float(raw.get("layer_norm_eps", raw.get("norm_eps", 1.0e-6))),
    }


name = "timm_vit"
runtime_strategy = "timm_vit_image_classification"
requires_tokenizer = False


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    mt = (model_type or "").lower()
    return mt.startswith("vit_") or mt in {"timm_vit", "vision_transformer"}


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    raw = config.raw
    vit_cfg = _resolve_vit_config(raw)
    raw["_timm_vit_config"] = vit_cfg
    target_dtype = _target_np_dtype(precision)

    weights = WeightDict()
    for key in (
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "cls_token",
        "pos_embed",
        "norm.weight",
        "norm.bias",
        "head.weight",
        "head.bias",
    ):
        if _has_tensor(readers, key):
            weights[key] = _load_tensor(readers, key).astype(
                np.float32
                if key.endswith(("bias", "weight")) and key.startswith("norm")
                else target_dtype
            )

    if "head.weight" not in weights:
        raise KeyError("Tensor not found: head.weight")
    if "head.bias" not in weights:
        weights["head.bias"] = np.zeros(int(weights["head.weight"].shape[0]), dtype=target_dtype)

    depth = vit_cfg["depth"]
    for layer_idx in range(depth):
        prefix = f"blocks.{layer_idx}"
        for key in (
            "norm1.weight",
            "norm1.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ):
            full_key = f"{prefix}.{key}"
            arr = _load_tensor(readers, full_key)
            if key.endswith("weight") and arr.ndim == 2:
                weights[full_key] = _transpose_2d(arr, full_key, precision=precision)
            else:
                weights[full_key] = arr.astype(
                    np.float32 if key.startswith("norm") else target_dtype
                )

    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="timm_vit tensor-parallel MLP builds"
        )
        if quant_ctx is not None:
            raise ValueError("timm_vit tensor-parallel builds do not support quantization")

        return build_timm_vit_tp_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            parallel_config=parallel,
        )

    del max_cache_length, quant_ctx
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported timm_vit precision: {precision}")

    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    image_h = vit_cfg["image_size_h"]
    image_w = vit_cfg["image_size_w"]
    patch_h = vit_cfg["patch_h"]
    patch_w = vit_cfg["patch_w"]
    hidden_size = vit_cfg["hidden"]
    depth = vit_cfg["depth"]
    num_heads = vit_cfg["heads"]
    mlp_hidden = vit_cfg["mlp_hidden"]
    num_classes = vit_cfg["num_classes"]
    eps_val = vit_cfg["eps"]

    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"image_size {image_h}x{image_w} must be divisible by patch {patch_h}x{patch_w}"
        )

    grid_h = image_h // patch_h
    grid_w = image_w // patch_w
    num_patches = grid_h * grid_w
    seq_len = num_patches + 1

    if verbose:
        print(
            "[trtmc build] timm_vit: "
            f"image={image_h}x{image_w}, patch={patch_h}x{patch_w}, "
            f"tokens={seq_len}, hidden={hidden_size}, layers={depth}, "
            f"heads={num_heads}, classes={num_classes}, "
            f"precision={precision}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            strongly_typed=True,
            explicit_batch=True,
        )
    )
    trt_config = builder.create_builder_config()
    trt_config.avg_timing_iterations = 8
    trt_config.max_aux_streams = 0
    trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_h, image_w))
    work_pixels = pixel_values
    if work_pixels.dtype != work_trt_dtype:
        work_pixels = network.add_cast(work_pixels, work_trt_dtype).get_output(0)

    patch = network.add_convolution_nd(
        work_pixels,
        num_output_maps=hidden_size,
        kernel_shape=(patch_h, patch_w),
        kernel=trt.Weights(
            np.ascontiguousarray(weights["patch_embed.proj.weight"], dtype=work_np_dtype)
        ),
        bias=trt.Weights(
            np.ascontiguousarray(weights["patch_embed.proj.bias"], dtype=work_np_dtype)
        ),
    )
    patch.stride_nd = (patch_h, patch_w)

    patches_nhwc = network.add_shuffle(patch.get_output(0))
    patches_nhwc.first_transpose = (0, 2, 3, 1)
    patches_nhwc.reshape_dims = (1, num_patches, hidden_size)
    hidden = patches_nhwc.get_output(0)

    cls_token = np.ascontiguousarray(
        weights["cls_token"].reshape(1, 1, hidden_size), dtype=work_np_dtype
    )
    cls_const = graph_ops.add_constant(network, (1, 1, hidden_size), cls_token, dtype=work_np_dtype)
    cat = network.add_concatenation([cls_const, hidden])
    cat.axis = 1
    hidden = cat.get_output(0)

    pos_embed = np.ascontiguousarray(
        weights["pos_embed"].reshape(1, seq_len, hidden_size), dtype=work_np_dtype
    )
    pos_const = graph_ops.add_constant(
        network, (1, seq_len, hidden_size), pos_embed, dtype=work_np_dtype
    )
    hidden = network.add_elementwise(hidden, pos_const, trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(depth):
        prefix = f"blocks.{layer_idx}"
        norm1 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm1.weight"],
            weights[f"{prefix}.norm1.bias"],
            eps_val,
            dtype=work_np_dtype,
        )

        qkv_w = weights[f"{prefix}.attn.qkv.weight"].astype(work_np_dtype)
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=1)
        qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
        q_b = k_b = v_b = None
        if qkv_b is not None:
            q_b, k_b, v_b = np.split(qkv_b.astype(work_np_dtype), 3)

        q = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            q_w,
            dtype=work_np_dtype,
        )
        k = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            k_w,
            dtype=work_np_dtype,
        )
        v = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            v_w,
            dtype=work_np_dtype,
        )
        if q_b is not None:
            q = graph_ops.add_bias_sum(network, q, hidden_size, q_b, dtype=work_np_dtype)
            k = graph_ops.add_bias_sum(network, k, hidden_size, k_b, dtype=work_np_dtype)
            v = graph_ops.add_bias_sum(network, v, hidden_size, v_b, dtype=work_np_dtype)

        head_dim = hidden_size // num_heads

        def to_heads(x: trt.ITensor) -> trt.ITensor:
            heads = network.add_shuffle(x)
            heads.reshape_dims = (1, seq_len, num_heads, head_dim)
            heads.second_transpose = trt.Permutation([0, 2, 1, 3])
            return heads.get_output(0)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)
        scale = graph_ops.add_constant(
            network,
            (1, 1, 1, 1),
            np.array([[[[1.0 / np.sqrt(head_dim)]]]], dtype=work_np_dtype),
            dtype=work_np_dtype,
        )
        q_scaled = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
        scores = network.add_matrix_multiply(
            q_scaled,
            trt.MatrixOperation.NONE,
            k,
            trt.MatrixOperation.TRANSPOSE,
        )
        probs = network.add_softmax(scores.get_output(0))
        probs.axes = 1 << 3
        context = network.add_matrix_multiply(
            probs.get_output(0),
            trt.MatrixOperation.NONE,
            v,
            trt.MatrixOperation.NONE,
        )
        attn_context = network.add_shuffle(context.get_output(0))
        attn_context.first_transpose = trt.Permutation([0, 2, 1, 3])
        attn_context.reshape_dims = (1, seq_len, hidden_size)
        attn = graph_ops.add_matmul_rhs_constant(
            network,
            attn_context.get_output(0),
            hidden_size,
            hidden_size,
            weights[f"{prefix}.attn.proj.weight"],
            dtype=work_np_dtype,
        )
        attn = graph_ops.add_bias_sum(
            network,
            attn,
            hidden_size,
            weights[f"{prefix}.attn.proj.bias"],
            dtype=work_np_dtype,
        )
        hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)

        norm2 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm2.weight"],
            weights[f"{prefix}.norm2.bias"],
            eps_val,
            dtype=work_np_dtype,
        )
        fc1 = graph_ops.add_matmul_rhs_constant(
            network,
            norm2,
            hidden_size,
            mlp_hidden,
            weights[f"{prefix}.mlp.fc1.weight"],
            dtype=work_np_dtype,
        )
        fc1 = graph_ops.add_bias_sum(
            network, fc1, mlp_hidden, weights[f"{prefix}.mlp.fc1.bias"], dtype=work_np_dtype
        )
        act = graph_ops.add_gelu_erf(network, fc1, dtype=work_np_dtype)
        fc2 = graph_ops.add_matmul_rhs_constant(
            network,
            act,
            mlp_hidden,
            hidden_size,
            weights[f"{prefix}.mlp.fc2.weight"],
            dtype=work_np_dtype,
        )
        fc2 = graph_ops.add_bias_sum(
            network, fc2, hidden_size, weights[f"{prefix}.mlp.fc2.bias"], dtype=work_np_dtype
        )
        hidden = network.add_elementwise(hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    hidden = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights["norm.weight"],
        weights["norm.bias"],
        eps_val,
        dtype=work_np_dtype,
    )
    cls = network.add_slice(
        hidden, start=(0, 0, 0), shape=(1, 1, hidden_size), stride=(1, 1, 1)
    ).get_output(0)

    logits = graph_ops.add_matmul_rhs_constant(
        network,
        cls,
        hidden_size,
        num_classes,
        _transpose_2d(
            weights["head.weight"],
            "head.weight",
            precision=precision,
        ),
        dtype=work_np_dtype,
    )
    logits = graph_ops.add_bias_sum(
        network, logits, num_classes, weights["head.bias"], dtype=work_np_dtype
    )
    flatten_logits = network.add_shuffle(logits)
    flatten_logits.reshape_dims = (1, num_classes)
    logits = flatten_logits.get_output(0)
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT timm_vit engine build failed")
    return bytes(plan)


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    mean = config.raw.get("mean", [0.5, 0.5, 0.5])
    std = config.raw.get("std", [0.5, 0.5, 0.5])
    return {
        "model_type": config.model_type,
        "runtime_strategy": runtime_strategy,
        "hidden_size": vit_cfg["hidden"],
        "num_hidden_layers": vit_cfg["depth"],
        "num_attention_heads": vit_cfg["heads"],
        "input_image_h": vit_cfg["image_size_h"],
        "input_image_w": vit_cfg["image_size_w"],
        "num_classes": vit_cfg["num_classes"],
        "image_mean": mean,
        "image_std": std,
        "crop_pct": float(config.raw.get("crop_pct", 0.9)),
        "interpolation": str(config.raw.get("interpolation", "bicubic")),
    }


def _validate_timm_vit_tp(config: "ModelConfig", parallel: "ParallelConfig") -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("timm_vit tensor-parallel build requires a concrete rank")
    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    mlp_hidden = int(vit_cfg["mlp_hidden"])
    if mlp_hidden % parallel.tp_size != 0:
        raise ValueError(
            "timm_vit tensor-parallel MLP requires mlp_hidden divisible by tp_size "
            f"({mlp_hidden} vs {parallel.tp_size})"
        )


def _slice_mlp_columns(
    arr: np.ndarray,
    mlp_hidden: int,
    parallel: "ParallelConfig",
) -> np.ndarray:
    local = mlp_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[..., start:end])


def _slice_mlp_rows(
    arr: np.ndarray,
    mlp_hidden: int,
    parallel: "ParallelConfig",
) -> np.ndarray:
    local = mlp_hidden // parallel.tp_size
    start = parallel.rank * local
    end = start + local
    return np.ascontiguousarray(arr[start:end, ...])


def build_timm_vit_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local timm ViT engine with tensor-parallel MLPs."""
    del max_cache_length
    if quant_ctx is not None:
        raise ValueError("timm_vit tensor-parallel builds do not support quantization")
    if precision not in ("fp32",):
        raise ValueError("timm_vit tensor-parallel path currently supports precision='fp32'")

    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("timm_vit tensor-parallel builder requires an enabled parallel config")
    _validate_timm_vit_tp(config, parallel)

    vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
    image_h = vit_cfg["image_size_h"]
    image_w = vit_cfg["image_size_w"]
    patch_h = vit_cfg["patch_h"]
    patch_w = vit_cfg["patch_w"]
    hidden_size = vit_cfg["hidden"]
    depth = vit_cfg["depth"]
    num_heads = vit_cfg["heads"]
    mlp_hidden = vit_cfg["mlp_hidden"]
    local_mlp_hidden = mlp_hidden // parallel.tp_size
    num_classes = vit_cfg["num_classes"]
    eps_val = vit_cfg["eps"]

    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"image_size {image_h}x{image_w} must be divisible by patch {patch_h}x{patch_w}"
        )

    grid_h = image_h // patch_h
    grid_w = image_w // patch_w
    num_patches = grid_h * grid_w
    seq_len = num_patches + 1

    if verbose:
        print(
            "[trtmc build] timm_vit TP: "
            f"rank={parallel.rank}/{parallel.tp_size}, image={image_h}x{image_w}, "
            f"patch={patch_h}x{patch_w}, tokens={seq_len}, hidden={hidden_size}, "
            f"layers={depth}, heads={num_heads}, mlp_local={local_mlp_hidden}, "
            f"classes={num_classes}",
            file=sys.stderr,
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            strongly_typed=True,
            explicit_batch=True,
        )
    )
    trt_config = builder.create_builder_config()
    trt_config.avg_timing_iterations = 8
    trt_config.max_aux_streams = 0
    trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_h, image_w))

    patch = network.add_convolution_nd(
        pixel_values,
        num_output_maps=hidden_size,
        kernel_shape=(patch_h, patch_w),
        kernel=trt.Weights(
            np.ascontiguousarray(weights["patch_embed.proj.weight"], dtype=np.float32)
        ),
        bias=trt.Weights(np.ascontiguousarray(weights["patch_embed.proj.bias"], dtype=np.float32)),
    )
    patch.stride_nd = (patch_h, patch_w)

    patches_nhwc = network.add_shuffle(patch.get_output(0))
    patches_nhwc.first_transpose = (0, 2, 3, 1)
    patches_nhwc.reshape_dims = (1, num_patches, hidden_size)
    hidden = patches_nhwc.get_output(0)

    cls_token = np.ascontiguousarray(
        weights["cls_token"].reshape(1, 1, hidden_size), dtype=np.float32
    )
    cls_const = graph_ops.add_constant(network, (1, 1, hidden_size), cls_token, dtype=np.float32)
    cat = network.add_concatenation([cls_const, hidden])
    cat.axis = 1
    hidden = cat.get_output(0)

    pos_embed = np.ascontiguousarray(
        weights["pos_embed"].reshape(1, seq_len, hidden_size), dtype=np.float32
    )
    pos_const = graph_ops.add_constant(
        network, (1, seq_len, hidden_size), pos_embed, dtype=np.float32
    )
    hidden = network.add_elementwise(hidden, pos_const, trt.ElementWiseOperation.SUM).get_output(0)

    for layer_idx in range(depth):
        prefix = f"blocks.{layer_idx}"
        norm1 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm1.weight"].astype(np.float32),
            weights[f"{prefix}.norm1.bias"].astype(np.float32),
            eps_val,
        )

        qkv_w = weights[f"{prefix}.attn.qkv.weight"].astype(np.float32)
        q_w, k_w, v_w = np.split(qkv_w, 3, axis=1)
        qkv_b = weights.get(f"{prefix}.attn.qkv.bias")
        q_b = k_b = v_b = None
        if qkv_b is not None:
            q_b, k_b, v_b = np.split(qkv_b.astype(np.float32), 3)

        q = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            q_w,
        )
        k = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            k_w,
        )
        v = graph_ops.add_matmul_rhs_constant(
            network,
            norm1,
            hidden_size,
            hidden_size,
            v_w,
        )
        if q_b is not None:
            q = graph_ops.add_bias_sum(network, q, hidden_size, q_b)
            k = graph_ops.add_bias_sum(network, k, hidden_size, k_b)
            v = graph_ops.add_bias_sum(network, v, hidden_size, v_b)

        head_dim = hidden_size // num_heads

        def to_heads(x: trt.ITensor) -> trt.ITensor:
            heads = network.add_shuffle(x)
            heads.reshape_dims = (1, seq_len, num_heads, head_dim)
            heads.second_transpose = trt.Permutation([0, 2, 1, 3])
            return heads.get_output(0)

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)
        scale = graph_ops.add_constant(
            network,
            (1, 1, 1, 1),
            np.array([[[[1.0 / np.sqrt(head_dim)]]]], dtype=np.float32),
            dtype=np.float32,
        )
        q_scaled = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
        scores = network.add_matrix_multiply(
            q_scaled,
            trt.MatrixOperation.NONE,
            k,
            trt.MatrixOperation.TRANSPOSE,
        )
        probs = network.add_softmax(scores.get_output(0))
        probs.axes = 1 << 3
        context = network.add_matrix_multiply(
            probs.get_output(0),
            trt.MatrixOperation.NONE,
            v,
            trt.MatrixOperation.NONE,
        )
        attn_context = network.add_shuffle(context.get_output(0))
        attn_context.first_transpose = trt.Permutation([0, 2, 1, 3])
        attn_context.reshape_dims = (1, seq_len, hidden_size)
        attn = graph_ops.add_matmul_rhs_constant(
            network,
            attn_context.get_output(0),
            hidden_size,
            hidden_size,
            weights[f"{prefix}.attn.proj.weight"].astype(np.float32),
        )
        attn = graph_ops.add_bias_sum(
            network,
            attn,
            hidden_size,
            weights[f"{prefix}.attn.proj.bias"].astype(np.float32),
        )
        hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)

        norm2 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm2.weight"].astype(np.float32),
            weights[f"{prefix}.norm2.bias"].astype(np.float32),
            eps_val,
        )
        fc1_w = _slice_mlp_columns(
            weights[f"{prefix}.mlp.fc1.weight"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc1_b = _slice_mlp_columns(
            weights[f"{prefix}.mlp.fc1.bias"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc1 = graph_ops.add_matmul_rhs_constant(
            network,
            norm2,
            hidden_size,
            local_mlp_hidden,
            fc1_w,
        )
        fc1 = graph_ops.add_bias_sum(network, fc1, local_mlp_hidden, fc1_b)
        act = graph_ops.add_gelu_erf(network, fc1)
        fc2_w = _slice_mlp_rows(
            weights[f"{prefix}.mlp.fc2.weight"].astype(np.float32),
            mlp_hidden,
            parallel,
        )
        fc2 = graph_ops.add_matmul_rhs_constant(
            network,
            act,
            local_mlp_hidden,
            hidden_size,
            fc2_w,
        )
        fc2 = add_all_reduce_sum(network, fc2, parallel.tp_size)
        fc2 = graph_ops.add_bias_sum(
            network,
            fc2,
            hidden_size,
            weights[f"{prefix}.mlp.fc2.bias"].astype(np.float32),
        )
        hidden = network.add_elementwise(hidden, fc2, trt.ElementWiseOperation.SUM).get_output(0)

    hidden = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights["norm.weight"].astype(np.float32),
        weights["norm.bias"].astype(np.float32),
        eps_val,
    )
    cls = network.add_slice(
        hidden, start=(0, 0, 0), shape=(1, 1, hidden_size), stride=(1, 1, 1)
    ).get_output(0)

    logits = graph_ops.add_matmul_rhs_constant(
        network,
        cls,
        hidden_size,
        num_classes,
        _transpose_2d(
            weights["head.weight"].astype(np.float32),
            "head.weight",
        ),
    )
    logits = graph_ops.add_bias_sum(
        network, logits, num_classes, weights["head.bias"].astype(np.float32)
    )
    flatten_logits = network.add_shuffle(logits)
    flatten_logits.reshape_dims = (1, num_classes)
    logits = flatten_logits.get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT timm_vit tensor-parallel engine build failed")
    return bytes(plan)


requires_tokenizer = False


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(
    config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete timm_vit bundle inside its owning family module."""
    from dataclasses import replace
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("timm_vit does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("timm_vit does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context

        family_graph_ops = sys.modules[__name__]

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=family_graph_ops,
        )

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="timm_vit tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("timm_vit tensor-parallel builds do not support quantization")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, quant_ctx, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    overrides = get_bundle_config_overrides(config)
    if overrides is not None:
        merged = dict(overrides)
        merged.update(runtime_config)
        merged.update(overrides)
        runtime_config = merged

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_gelu_erf(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (exact, erf-based): 0.5 * x * (1 + erf(x / sqrt(2))).

    Constants are cast to ``inp.dtype`` for the same STRONGLY_TYPED reason
    documented on ``add_gelu_new``.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    inv_sqrt2 = _const(1.0 / np.sqrt(2.0))
    x_scaled = network.add_elementwise(inp, inv_sqrt2, trt.ElementWiseOperation.PROD)
    erf_out = network.add_unary(x_scaled.get_output(0), trt.UnaryOperation.ERF)
    one = _const(1.0)
    one_plus_erf = network.add_elementwise(one, erf_out.get_output(0), trt.ElementWiseOperation.SUM)
    half = _const(0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_erf.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


def add_layer_norm_native(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm via TRT native INormalizationLayer (add_normalization_v2).

    Replaces the manual reduce/elementwise chain in add_layer_norm with a
    single fused layer that TRT can optimize end-to-end. In strongly typed
    networks, input/scale/bias must have identical tensor types; compute
    precision is set to FP32 for numerical stability when the TensorRT Python
    layer exposes that control.

    Note: INormalizationLayer computes (x - mean) / sqrt(var + eps) * gamma + beta.
    This is LayerNorm, NOT RMSNorm.  Use add_rms_norm for RMSNorm models.

    Args:
        inp:         Input tensor [*, hidden_size].
        hidden_size: Size of the normalized dimension (last axis).
        gamma:       Scale weights [hidden_size].
        beta:        Bias weights [hidden_size].
        eps:         Numerical stability epsilon (scalar, not a tensor).
        dtype:       Storage dtype for gamma/beta constants before TRT cast.
    """
    inp_shape = getattr(inp, "shape", None)
    rank = len(tuple(inp_shape)) if inp_shape is not None else 2
    param_shape = (hidden_size,) if rank <= 1 else (1,) * (rank - 1) + (hidden_size,)
    gamma_t = add_constant(
        network, param_shape, np.asarray(gamma).reshape(param_shape), dtype=dtype
    )
    beta_t = add_constant(network, param_shape, np.asarray(beta).reshape(param_shape), dtype=dtype)
    gamma_t = _cast_back_to_trt_dtype(network, gamma_t, inp.dtype)
    beta_t = _cast_back_to_trt_dtype(network, beta_t, inp.dtype)
    # axesMask bit i selects axis i as a reduction axis. The normalized
    # hidden dimension is always the last axis for [*, hidden_size] tensors.
    norm = network.add_normalization_v2(inp, gamma_t, beta_t, 1 << (rank - 1))
    norm.epsilon = eps
    # TensorRT 11 removed the Python INormalizationLayer.compute_precision
    # attribute. Keep the TRT 10 hint, and let TRT 11 infer the precision.
    if hasattr(norm, "compute_precision"):
        norm.compute_precision = trt.float32
    return norm.get_output(0)
