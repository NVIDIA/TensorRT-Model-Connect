# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""timm ViT image-classification family plugin.

Supports fixed-size timm Vision Transformer classifiers stored in HF Hub
format. The initial target is:
  timm/vit_base_patch16_224.augreg_in21k_ft_in1k

The builder constructs the classifier with TensorRT Network API calls rather
than routing through ONNX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .graph import model as graph_ops
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
    _transpose_2d,
)
from .config import ModelConfig
from .parallel import ParallelConfig
from .parallel import normalize_parallel_config


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


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _TimmVitModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
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
            weights["head.bias"] = np.zeros(
                int(weights["head.weight"].shape[0]), dtype=target_dtype
            )

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
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
            if quant_ctx is not None:
                raise ValueError("timm_vit tensor-parallel builds do not support quantization")
            from .model.parallel import build_timm_vit_tp_engine

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
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
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
        cls_const = graph_ops.add_constant(
            network, (1, 1, hidden_size), cls_token, dtype=work_np_dtype
        )
        cat = network.add_concatenation([cls_const, hidden])
        cat.axis = 1
        hidden = cat.get_output(0)

        pos_embed = np.ascontiguousarray(
            weights["pos_embed"].reshape(1, seq_len, hidden_size), dtype=work_np_dtype
        )
        pos_const = graph_ops.add_constant(
            network, (1, seq_len, hidden_size), pos_embed, dtype=work_np_dtype
        )
        hidden = network.add_elementwise(
            hidden, pos_const, trt.ElementWiseOperation.SUM
        ).get_output(0)

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
            q_scaled = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(
                0
            )
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
            hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(
                0
            )

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
            hidden = network.add_elementwise(hidden, fc2, trt.ElementWiseOperation.SUM).get_output(
                0
            )

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

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        vit_cfg = config.raw.get("_timm_vit_config") or _resolve_vit_config(config.raw)
        mean = config.raw.get("mean", [0.5, 0.5, 0.5])
        std = config.raw.get("std", [0.5, 0.5, 0.5])
        return {
            "model_type": config.model_type,
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


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one timm ViT bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_vit does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("timm_vit does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("timm_vit does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("timm_vit does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("timm_vit does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "classification":
        raise ValueError("timm_vit supports only task=classification")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if not (
        str(config.model_type).lower().startswith("vit_")
        or str(config.model_type).lower() in {"timm_vit", "vision_transformer"}
    ):
        raise ValueError(f"timm ViT does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    max_sequence_length = _positive_int(request.max_sequence_length or 1, "max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm ViT does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm ViT does not support mixed-precision layers")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    model = _TimmVitModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    writer.set_header(family="timm_vit", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
    runtime_source = model.get_bundle_config_overrides(config)
    runtime = {
        key: runtime_source[key]
        for key in (
            "input_image_h",
            "input_image_w",
            "crop_pct",
            "interpolation",
            "image_mean",
            "image_std",
        )
    }
    runtime["tensor_parallel_size"] = request.tensor_parallel_size
    writer.add_json("runtime.json", runtime)
