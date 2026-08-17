# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT DINOv3 image-feature-extraction family.

This owner implements both architectures exposed by the Transformers DINOv3
model page.  Network construction uses TensorRT's Python Network API directly;
there is no ONNX export, parser, or fallback path.
"""

from __future__ import annotations

import sys
from typing import Protocol

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import (
    WeightDict,
    as_weight,
    has_tensor,
    layer_key,
    load_tensor,
    open_checkpoint,
    target_dtype,
    transpose_linear,
)


trt = trt_compat.get_trt()


class ModelConfig(Protocol):
    """Structural subset supplied by the repository's model-config loader."""

    model_type: str
    raw: dict
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int


def _pair(value, *, field: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"DINOv3 {field} must be an int or pair, got {value!r}")


def resolve_vit_config(raw: dict) -> dict:
    image_h, image_w = _pair(raw.get("image_size", 224), field="image_size")
    patch_h, patch_w = _pair(raw.get("patch_size", 16), field="patch_size")
    hidden_size = int(raw.get("hidden_size", 384))
    num_heads = int(raw.get("num_attention_heads", 6))
    if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError(
            "DINOv3 hidden_size must be positive and divisible by num_attention_heads"
        )
    head_dim = hidden_size // num_heads
    if head_dim % 4 != 0:
        raise ValueError("DINOv3 2D RoPE requires head_dim divisible by 4")
    if image_h % patch_h or image_w % patch_w:
        raise ValueError("DINOv3 image dimensions must be divisible by patch dimensions")
    if patch_h != patch_w:
        raise ValueError("DINOv3 ViT currently requires square patches")
    return {
        "image_h": image_h,
        "image_w": image_w,
        "patch_size": patch_h,
        "hidden_size": hidden_size,
        "intermediate_size": int(raw.get("intermediate_size", 4 * hidden_size)),
        "num_hidden_layers": int(raw.get("num_hidden_layers", 12)),
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "hidden_act": str(raw.get("hidden_act", "gelu")),
        "layer_norm_eps": float(raw.get("layer_norm_eps", 1.0e-5)),
        "rope_theta": float(raw.get("rope_theta", 100.0)),
        "num_register_tokens": int(raw.get("num_register_tokens", 0)),
        "query_bias": bool(raw.get("query_bias", True)),
        "key_bias": bool(raw.get("key_bias", False)),
        "value_bias": bool(raw.get("value_bias", True)),
        "proj_bias": bool(raw.get("proj_bias", True)),
        "mlp_bias": bool(raw.get("mlp_bias", True)),
        "use_gated_mlp": bool(raw.get("use_gated_mlp", False)),
    }


def _required_layer(readers, layer: int, suffix: str, precision: str) -> np.ndarray:
    name = layer_key(readers, layer, suffix)
    return as_weight(load_tensor(readers, name), precision)


def _linear_layer(readers, layer: int, suffix: str, precision: str) -> np.ndarray:
    name = layer_key(readers, layer, suffix)
    return transpose_linear(load_tensor(readers, name), name, precision)


def _optional_layer_bias(
    readers, layer: int, suffix: str, *, enabled: bool, precision: str
) -> np.ndarray | None:
    candidates = (f"model.layer.{layer}.{suffix}", f"layer.{layer}.{suffix}")
    for name in candidates:
        if has_tensor(readers, name):
            return as_weight(load_tensor(readers, name), precision)
    if enabled:
        raise KeyError("Tensor not found; tried: " + ", ".join(candidates))
    return None


def load_vit_weights(
    model_dir: str, config: ModelConfig, *, precision: str
) -> WeightDict:
    readers = open_checkpoint(model_dir)
    cfg = resolve_vit_config(config.raw)
    config.raw["_dinov3_config"] = cfg
    weights = WeightDict()

    for logical, checkpoint_name in (
        ("cls_token", "embeddings.cls_token"),
        ("register_tokens", "embeddings.register_tokens"),
        ("patch.weight", "embeddings.patch_embeddings.weight"),
        ("patch.bias", "embeddings.patch_embeddings.bias"),
        ("norm.weight", "norm.weight"),
        ("norm.bias", "norm.bias"),
    ):
        weights[logical] = as_weight(load_tensor(readers, checkpoint_name), precision)

    register_shape = tuple(weights["register_tokens"].shape)
    expected_register_shape = (1, cfg["num_register_tokens"], cfg["hidden_size"])
    if register_shape != expected_register_shape:
        raise ValueError(
            "DINOv3 register token shape mismatch: "
            f"checkpoint={register_shape}, config={expected_register_shape}"
        )

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"layer.{layer}"
        for suffix in (
            "norm1.weight",
            "norm1.bias",
            "layer_scale1.lambda1",
            "norm2.weight",
            "norm2.bias",
            "layer_scale2.lambda1",
        ):
            weights[f"{prefix}.{suffix}"] = _required_layer(
                readers, layer, suffix, precision
            )

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            suffix = f"attention.{projection}.weight"
            weights[f"{prefix}.{suffix}"] = _linear_layer(
                readers, layer, suffix, precision
            )
        for projection, enabled in (
            ("q_proj", cfg["query_bias"]),
            ("k_proj", cfg["key_bias"]),
            ("v_proj", cfg["value_bias"]),
            ("o_proj", cfg["proj_bias"]),
        ):
            suffix = f"attention.{projection}.bias"
            bias = _optional_layer_bias(
                readers, layer, suffix, enabled=enabled, precision=precision
            )
            if bias is not None:
                weights[f"{prefix}.{suffix}"] = bias

        mlp_projections = ["up_proj", "down_proj"]
        if cfg["use_gated_mlp"]:
            mlp_projections.insert(0, "gate_proj")
        for projection in mlp_projections:
            suffix = f"mlp.{projection}.weight"
            weights[f"{prefix}.{suffix}"] = _linear_layer(
                readers, layer, suffix, precision
            )
            bias_suffix = f"mlp.{projection}.bias"
            bias = _optional_layer_bias(
                readers,
                layer,
                bias_suffix,
                enabled=cfg["mlp_bias"],
                precision=precision,
            )
            if bias is not None:
                weights[f"{prefix}.{bias_suffix}"] = bias
    return weights


def _work_types(precision: str) -> tuple[np.dtype, trt.DataType]:
    if precision == "fp32":
        return np.dtype(np.float32), trt.float32
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    raise ValueError(f"Unsupported DINOv3 precision: {precision}")


def _new_network(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(strongly_typed=True, explicit_batch=True)
    )
    builder_config = builder.create_builder_config()
    builder_config.avg_timing_iterations = 8
    builder_config.max_aux_streams = 0
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    return builder, network, builder_config


def _with_bias(network, tensor, weights, key: str, dtype: np.dtype):
    return graph_ops.add_bias(network, tensor, weights.get(key), dtype)


def build_vit_engine(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    cfg = config.raw.get("_dinov3_config") or resolve_vit_config(config.raw)
    work_dtype, work_trt_dtype = _work_types(precision)
    builder, network, builder_config = _new_network(verbose)

    image_h = cfg["image_h"]
    image_w = cfg["image_w"]
    patch_size = cfg["patch_size"]
    hidden_size = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg["head_dim"]
    num_registers = cfg["num_register_tokens"]
    num_prefix = 1 + num_registers
    grid_h = image_h // patch_size
    grid_w = image_w // patch_size
    num_patches = grid_h * grid_w
    sequence_length = num_prefix + num_patches

    if verbose:
        print(
            "[trtmc build] dinov3_vit: "
            f"image={image_h}x{image_w}, patch={patch_size}, tokens={sequence_length}, "
            f"hidden={hidden_size}, layers={num_layers}, heads={num_heads}, "
            f"registers={num_registers}, precision={precision}",
            file=sys.stderr,
        )

    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, image_h, image_w)
    )
    pixels = graph_ops.cast(network, pixel_values, work_trt_dtype)
    patch = network.add_convolution_nd(
        pixels,
        hidden_size,
        (patch_size, patch_size),
        trt.Weights(np.ascontiguousarray(weights["patch.weight"], dtype=work_dtype)),
        trt.Weights(np.ascontiguousarray(weights["patch.bias"], dtype=work_dtype)),
    )
    patch.stride_nd = (patch_size, patch_size)
    patch_tokens = network.add_shuffle(patch.get_output(0))
    patch_tokens.first_transpose = (0, 2, 3, 1)
    patch_tokens.reshape_dims = (1, num_patches, hidden_size)

    cls = graph_ops.constant(
        network, weights["cls_token"], (1, 1, hidden_size), work_dtype
    )
    cls = graph_ops.cast(network, cls, work_trt_dtype)
    pieces = [cls]
    if num_registers:
        registers = graph_ops.constant(
            network,
            weights["register_tokens"],
            (1, num_registers, hidden_size),
            work_dtype,
        )
        pieces.append(graph_ops.cast(network, registers, work_trt_dtype))
    pieces.append(patch_tokens.get_output(0))
    concat = network.add_concatenation(pieces)
    concat.axis = 1
    hidden = concat.get_output(0)

    def to_heads(tensor):
        shuffle = network.add_shuffle(tensor)
        shuffle.reshape_dims = (1, sequence_length, num_heads, head_dim)
        shuffle.second_transpose = (0, 2, 1, 3)
        return shuffle.get_output(0)

    for layer in range(num_layers):
        prefix = f"layer.{layer}"
        residual = hidden
        normalized = graph_ops.layer_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm1.weight"],
            weights[f"{prefix}.norm1.bias"],
            cfg["layer_norm_eps"],
            work_dtype,
        )
        projections = {}
        for short, projection in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            value = graph_ops.linear(
                network,
                normalized,
                weights[f"{prefix}.attention.{projection}.weight"],
                work_dtype,
            )
            value = _with_bias(
                network,
                value,
                weights,
                f"{prefix}.attention.{projection}.bias",
                work_dtype,
            )
            projections[short] = to_heads(value)
        q = graph_ops.apply_patch_rope(
            network,
            projections["q"],
            num_heads=num_heads,
            num_prefix_tokens=num_prefix,
            grid_h=grid_h,
            grid_w=grid_w,
            head_dim=head_dim,
            theta=cfg["rope_theta"],
            dtype=work_dtype,
        )
        k = graph_ops.apply_patch_rope(
            network,
            projections["k"],
            num_heads=num_heads,
            num_prefix_tokens=num_prefix,
            grid_h=grid_h,
            grid_w=grid_w,
            head_dim=head_dim,
            theta=cfg["rope_theta"],
            dtype=work_dtype,
        )
        context = graph_ops.attention(
            network, q, k, projections["v"], head_dim, work_dtype
        )
        merged = network.add_shuffle(context)
        merged.first_transpose = (0, 2, 1, 3)
        merged.reshape_dims = (1, sequence_length, hidden_size)
        attention_out = graph_ops.linear(
            network,
            merged.get_output(0),
            weights[f"{prefix}.attention.o_proj.weight"],
            work_dtype,
        )
        attention_out = _with_bias(
            network,
            attention_out,
            weights,
            f"{prefix}.attention.o_proj.bias",
            work_dtype,
        )
        attention_out = graph_ops.multiply_last_dim(
            network,
            attention_out,
            weights[f"{prefix}.layer_scale1.lambda1"],
            work_dtype,
        )
        hidden = network.add_elementwise(
            residual, attention_out, trt.ElementWiseOperation.SUM
        ).get_output(0)

        residual = hidden
        normalized = graph_ops.layer_norm(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.norm2.weight"],
            weights[f"{prefix}.norm2.bias"],
            cfg["layer_norm_eps"],
            work_dtype,
        )
        if cfg["use_gated_mlp"]:
            gate = graph_ops.linear(
                network,
                normalized,
                weights[f"{prefix}.mlp.gate_proj.weight"],
                work_dtype,
            )
            gate = _with_bias(
                network,
                gate,
                weights,
                f"{prefix}.mlp.gate_proj.bias",
                work_dtype,
            )
            gate = graph_ops.activation(
                network, gate, cfg["hidden_act"], work_dtype
            )
            up = graph_ops.linear(
                network,
                normalized,
                weights[f"{prefix}.mlp.up_proj.weight"],
                work_dtype,
            )
            up = _with_bias(
                network, up, weights, f"{prefix}.mlp.up_proj.bias", work_dtype
            )
            activated = network.add_elementwise(
                gate, up, trt.ElementWiseOperation.PROD
            ).get_output(0)
        else:
            activated = graph_ops.linear(
                network,
                normalized,
                weights[f"{prefix}.mlp.up_proj.weight"],
                work_dtype,
            )
            activated = _with_bias(
                network,
                activated,
                weights,
                f"{prefix}.mlp.up_proj.bias",
                work_dtype,
            )
            activated = graph_ops.activation(
                network, activated, cfg["hidden_act"], work_dtype
            )
        mlp = graph_ops.linear(
            network,
            activated,
            weights[f"{prefix}.mlp.down_proj.weight"],
            work_dtype,
        )
        mlp = _with_bias(
            network, mlp, weights, f"{prefix}.mlp.down_proj.bias", work_dtype
        )
        mlp = graph_ops.multiply_last_dim(
            network,
            mlp,
            weights[f"{prefix}.layer_scale2.lambda1"],
            work_dtype,
        )
        hidden = network.add_elementwise(
            residual, mlp, trt.ElementWiseOperation.SUM
        ).get_output(0)

    hidden = graph_ops.layer_norm(
        network,
        hidden,
        hidden_size,
        weights["norm.weight"],
        weights["norm.bias"],
        cfg["layer_norm_eps"],
        work_dtype,
    )
    last_hidden_state = graph_ops.cast(network, hidden, trt.float32)
    last_hidden_state.name = "last_hidden_state"
    network.mark_output(last_hidden_state)

    pooled = network.add_slice(
        last_hidden_state,
        (0, 0, 0),
        (1, 1, hidden_size),
        (1, 1, 1),
    ).get_output(0)
    pooled_shuffle = network.add_shuffle(pooled)
    pooled_shuffle.reshape_dims = (1, hidden_size)
    pooled = pooled_shuffle.get_output(0)
    pooled.name = "pooler_output"
    network.mark_output(pooled)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT DINOv3 ViT engine build failed")
    return bytes(plan)


class Dinov3Plugin:
    name = "dinov3"
    runtime_strategy = "dinov3_image_feature_extraction"
    requires_tokenizer = False

    def default_max_cache_length(self, config: ModelConfig) -> int:
        del config
        return 1

    def matches(self, model_type: str) -> bool:
        return (model_type or "").lower() in {"dinov3_vit", "dinov3_convnext"}

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        target_dtype(precision)  # validate before opening a large checkpoint
        if config.model_type == "dinov3_vit":
            return load_vit_weights(model_dir, config, precision=precision)
        if config.model_type == "dinov3_convnext":
            from .convnext_builder import load_convnext_weights, resolve_convnext_config

            resolved = resolve_convnext_config(config)
            # ConvNeXt expresses these values as hidden_sizes/depths rather
            # than the scalar fields understood by the generic BundleInfo
            # writer. Repair the already-parsed object before bundle metadata
            # is serialized so `trtmc inspect` reports the real architecture.
            config.hidden_size = int(resolved["output_dim"])
            config.num_hidden_layers = int(resolved["num_layers"])
            config.num_attention_heads = 0
            config.num_key_value_heads = 0
            return load_convnext_weights(model_dir, config, precision=precision)
        raise ValueError(f"Unsupported DINOv3 model_type: {config.model_type!r}")

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
            raise ValueError("DINOv3 native image encoders do not support quantization yet")
        if parallel_config is not None and getattr(parallel_config, "enabled", False):
            raise ValueError("DINOv3 native image encoders do not support tensor parallelism")
        if config.model_type == "dinov3_vit":
            return build_vit_engine(
                config, weights, precision=precision, verbose=verbose
            )
        if config.model_type == "dinov3_convnext":
            from .convnext_builder import build_convnext_engine

            return build_convnext_engine(
                config, weights, precision=precision, verbose=verbose
            )
        raise ValueError(f"Unsupported DINOv3 model_type: {config.model_type!r}")

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        if config.model_type == "dinov3_vit":
            cfg = config.raw.get("_dinov3_config") or resolve_vit_config(config.raw)
            hidden_size = cfg["hidden_size"]
            sequence_length = (
                1
                + cfg["num_register_tokens"]
                + (cfg["image_h"] // cfg["patch_size"])
                * (cfg["image_w"] // cfg["patch_size"])
            )
            image_h, image_w = cfg["image_h"], cfg["image_w"]
            architecture = "vit"
        elif config.model_type == "dinov3_convnext":
            from .convnext_builder import convnext_bundle_metadata, resolve_convnext_config

            cfg = resolve_convnext_config(config)
            metadata = convnext_bundle_metadata(config)
            hidden_size = cfg["hidden_sizes"][-1]
            image_h, image_w = cfg["image_h"], cfg["image_w"]
            sequence_length = metadata["num_feature_tokens"]
            architecture = "convnext"
        else:
            raise ValueError(f"Unsupported DINOv3 model_type: {config.model_type!r}")
        overrides = {
            "model_type": config.model_type,
            "runtime_strategy": self.runtime_strategy,
            "dinov3_architecture": architecture,
            "input_image_h": image_h,
            "input_image_w": image_w,
            "hidden_size": hidden_size,
            "sequence_length": sequence_length,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "interpolation": "bilinear",
            "do_center_crop": False,
        }
        if config.model_type == "dinov3_convnext":
            overrides.update(metadata)
            overrides["runtime_strategy"] = self.runtime_strategy
            overrides["dinov3_architecture"] = architecture
            overrides["sequence_length"] = sequence_length
        return overrides


plugin = Dinov3Plugin()
