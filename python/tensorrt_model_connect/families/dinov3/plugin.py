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


_TIMM_DINOV3_VIT_ARCHITECTURE = "vit_small_patch16_dinov3_qkvb"
_TIMM_DINOV3_VIT_CONFIG = {
    "model_type": "dinov3_vit",
    "architectures": ["DINOv3ViTModel"],
    "image_size": 224,
    "patch_size": 16,
    "num_channels": 3,
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "num_key_value_heads": 6,
    "hidden_act": "gelu",
    "layer_norm_eps": 1.0e-5,
    "rope_theta": 100.0,
    "num_register_tokens": 4,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "use_gated_mlp": False,
}


class ModelConfig(Protocol):
    """Structural subset supplied by the repository's model-config loader."""

    model_type: str
    architectures: list[str]
    raw: dict
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str


def _is_timm_dinov3_vit_config(config: ModelConfig) -> bool:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    architecture = str(config.raw.get("architecture", "") or "").lower()
    return _TIMM_DINOV3_VIT_ARCHITECTURE in {model_type, architecture}


def _normalize_timm_dinov3_vit_config(config: ModelConfig) -> None:
    """Expand the exact public timm mirror config into the HF ViT contract."""
    if not _is_timm_dinov3_vit_config(config):
        return

    config.model_type = "dinov3_vit"
    config.architectures = ["DINOv3ViTModel"]
    config.hidden_size = 384
    config.intermediate_size = 1536
    config.num_hidden_layers = 12
    config.num_attention_heads = 6
    config.num_key_value_heads = 6
    config.rms_norm_eps = 1.0e-5
    config.rope_theta = 100.0
    config.hidden_act = "gelu"
    config.raw.update(
        {
            **_TIMM_DINOV3_VIT_CONFIG,
            "architectures": list(_TIMM_DINOV3_VIT_CONFIG["architectures"]),
        }
    )
    config.raw["_dinov3_checkpoint_layout"] = "timm"


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


def _layer_weight(
    readers, layer: int, suffix: str, precision: str, *, transpose: bool = False
) -> np.ndarray:
    name = layer_key(readers, layer, suffix)
    value = load_tensor(readers, name)
    return transpose_linear(value, name, precision) if transpose else as_weight(value, precision)


def _optional_layer_bias(
    readers, layer: int, suffix: str, *, enabled: bool, precision: str
) -> np.ndarray | None:
    try:
        name = layer_key(readers, layer, suffix)
    except KeyError:
        if enabled:
            raise
        return None
    return as_weight(load_tensor(readers, name), precision)


def _is_timm_vit_checkpoint(readers) -> bool:
    return all(
        has_tensor(readers, name)
        for name in (
            "cls_token",
            "reg_token",
            "patch_embed.proj.weight",
            "blocks.0.attn.qkv.weight",
        )
    )


def _load_timm_vit_weights(readers, cfg: dict, precision: str) -> WeightDict:
    """Map the public timm DINOv3 ViT layout to the existing HF graph ABI."""
    weights = WeightDict()
    for logical, checkpoint_name in (
        ("cls_token", "cls_token"),
        ("register_tokens", "reg_token"),
        ("patch.weight", "patch_embed.proj.weight"),
        ("patch.bias", "patch_embed.proj.bias"),
        ("norm.weight", "norm.weight"),
        ("norm.bias", "norm.bias"),
    ):
        weights[logical] = as_weight(load_tensor(readers, checkpoint_name), precision)

    hidden_size = cfg["hidden_size"]
    for layer in range(cfg["num_hidden_layers"]):
        source_prefix = f"blocks.{layer}"
        target_prefix = f"layer.{layer}"

        def store(logical: str, source: str, *, transpose=False):
            name = f"{source_prefix}.{source}"
            value = load_tensor(readers, name)
            weights[f"{target_prefix}.{logical}"] = (
                transpose_linear(value, name, precision)
                if transpose
                else as_weight(value, precision)
            )

        for target_suffix, source_suffix in (
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("layer_scale1.lambda1", "gamma_1"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
            ("layer_scale2.lambda1", "gamma_2"),
        ):
            store(target_suffix, source_suffix)

        qkv_name = f"{source_prefix}.attn.qkv.weight"
        qkv = load_tensor(readers, qkv_name)
        expected_shape = (3 * hidden_size, hidden_size)
        if tuple(qkv.shape) != expected_shape:
            raise ValueError(f"Expected {qkv_name} shape {expected_shape}, got {qkv.shape}")
        for projection, value in zip(
            ("q_proj", "k_proj", "v_proj"), np.split(qkv, 3, axis=0), strict=True
        ):
            weights[f"{target_prefix}.attention.{projection}.weight"] = transpose_linear(
                value, qkv_name, precision
            )

        store("attention.q_proj.bias", "attn.q_bias")
        store("attention.v_proj.bias", "attn.v_bias")
        store("attention.o_proj.weight", "attn.proj.weight", transpose=True)
        store("attention.o_proj.bias", "attn.proj.bias")

        for projection, source_projection in (
            ("up_proj", "fc1"),
            ("down_proj", "fc2"),
        ):
            store(f"mlp.{projection}.weight", f"mlp.{source_projection}.weight", transpose=True)
            store(f"mlp.{projection}.bias", f"mlp.{source_projection}.bias")
    return weights


def _validate_register_tokens(weights: WeightDict, cfg: dict) -> None:
    checkpoint = tuple(weights["register_tokens"].shape)
    expected = (1, cfg["num_register_tokens"], cfg["hidden_size"])
    if checkpoint != expected:
        raise ValueError(
            f"DINOv3 register token shape mismatch: checkpoint={checkpoint}, config={expected}"
        )


def load_vit_weights(
    model_dir: str, config: ModelConfig, *, precision: str
) -> WeightDict:
    readers = open_checkpoint(model_dir)
    cfg = resolve_vit_config(config.raw)
    config.raw["_dinov3_config"] = cfg

    if _is_timm_vit_checkpoint(readers):
        weights = _load_timm_vit_weights(readers, cfg, precision)
        _validate_register_tokens(weights, cfg)
        return weights

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

    _validate_register_tokens(weights, cfg)

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
            weights[f"{prefix}.{suffix}"] = _layer_weight(
                readers, layer, suffix, precision
            )

        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            suffix = f"attention.{projection}.weight"
            weights[f"{prefix}.{suffix}"] = _layer_weight(
                readers, layer, suffix, precision, transpose=True
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
            weights[f"{prefix}.{suffix}"] = _layer_weight(
                readers, layer, suffix, precision, transpose=True
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
    dtype = target_dtype(precision)
    return dtype, trt.float16 if dtype == np.dtype(np.float16) else trt.float32


def build_vit_engine(
    config: ModelConfig,
    weights: WeightDict,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    cfg = config.raw.get("_dinov3_config") or resolve_vit_config(config.raw)
    work_dtype, work_trt_dtype = _work_types(precision)
    builder, network, builder_config = graph_ops.new_network(verbose)

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
    patch_tokens = graph_ops.shuffle(
        network,
        patch.get_output(0),
        first_transpose=(0, 2, 3, 1),
        reshape_dims=(1, num_patches, hidden_size),
    )

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
    pieces.append(patch_tokens)
    concat = network.add_concatenation(pieces)
    concat.axis = 1
    hidden = concat.get_output(0)

    def to_heads(tensor):
        return graph_ops.shuffle(
            network,
            tensor,
            reshape_dims=(1, sequence_length, num_heads, head_dim),
            second_transpose=(0, 2, 1, 3),
        )

    for layer in range(num_layers):
        prefix = f"layer.{layer}"

        def project(group: str, name: str, tensor):
            return graph_ops.linear_with_bias(
                network, tensor, weights, f"{prefix}.{group}.{name}", work_dtype
            )

        def normalize(name: str, tensor):
            key = f"{prefix}.{name}"
            return graph_ops.layer_norm(
                network,
                tensor,
                hidden_size,
                weights[f"{key}.weight"],
                weights[f"{key}.bias"],
                cfg["layer_norm_eps"],
                work_dtype,
            )

        def add_residual(residual, tensor, scale: str):
            return graph_ops.add_scaled_residual(
                network, residual, tensor, weights[f"{prefix}.{scale}"], work_dtype
            )

        residual = hidden
        normalized = normalize("norm1", hidden)
        # A single wide GEMM matches the source checkpoint and avoids three
        # small projection launches on fixed-shape image tokens.
        qkv_weight = np.concatenate(
            [
                weights[f"{prefix}.attention.{name}_proj.weight"]
                for name in ("q", "k", "v")
            ],
            axis=1,
        )
        qkv = graph_ops.linear(network, normalized, qkv_weight, work_dtype)
        qkv_biases = [
            weights.get(f"{prefix}.attention.{name}_proj.bias")
            for name in ("q", "k", "v")
        ]
        if any(bias is not None for bias in qkv_biases):
            zero_bias = np.zeros(hidden_size, dtype=work_dtype)
            qkv = graph_ops.add_bias(
                network,
                qkv,
                np.concatenate(
                    [bias if bias is not None else zero_bias for bias in qkv_biases]
                ),
                work_dtype,
            )
        projections = {}
        for index, name in enumerate(("q", "k", "v")):
            projection = graph_ops.slice_tensor(
                network,
                qkv,
                (0, 0, index * hidden_size),
                (1, sequence_length, hidden_size),
            )
            projections[name] = to_heads(projection)
        # Q and K use identical RoPE coordinates, so transform them as one batch.
        qk = network.add_concatenation([projections["q"], projections["k"]])
        qk.axis = 0
        qk = graph_ops.apply_patch_rope(
            network,
            qk.get_output(0),
            num_heads=num_heads,
            num_prefix_tokens=num_prefix,
            grid_h=grid_h,
            grid_w=grid_w,
            head_dim=head_dim,
            theta=cfg["rope_theta"],
            dtype=work_dtype,
        )
        for index, name in enumerate(("q", "k")):
            projections[name] = graph_ops.slice_tensor(
                network,
                qk,
                (index, 0, 0, 0),
                (1, num_heads, sequence_length, head_dim),
            )
        context = graph_ops.attention(
            network,
            projections["q"],
            projections["k"],
            projections["v"],
            head_dim,
            work_dtype,
        )
        merged = graph_ops.shuffle(
            network,
            context,
            first_transpose=(0, 2, 1, 3),
            reshape_dims=(1, sequence_length, hidden_size),
        )
        attention_out = project("attention", "o_proj", merged)
        hidden = add_residual(residual, attention_out, "layer_scale1.lambda1")

        residual = hidden
        normalized = normalize("norm2", hidden)
        if cfg["use_gated_mlp"]:
            gate = project("mlp", "gate_proj", normalized)
            gate = graph_ops.activation(
                network, gate, cfg["hidden_act"], work_dtype
            )
            up = project("mlp", "up_proj", normalized)
            activated = network.add_elementwise(
                gate, up, trt.ElementWiseOperation.PROD
            ).get_output(0)
        else:
            activated = project("mlp", "up_proj", normalized)
            activated = graph_ops.activation(
                network, activated, cfg["hidden_act"], work_dtype
            )
        mlp = project("mlp", "down_proj", activated)
        hidden = add_residual(residual, mlp, "layer_scale2.lambda1")

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

    pooled = graph_ops.slice_tensor(
        network, last_hidden_state, (0, 0, 0), (1, 1, hidden_size)
    )
    pooled = graph_ops.shuffle(network, pooled, reshape_dims=(1, hidden_size))
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
        return (model_type or "").lower() in {
            "dinov3_vit",
            "dinov3_convnext",
            _TIMM_DINOV3_VIT_ARCHITECTURE,
        }

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        target_dtype(precision)  # validate before opening a large checkpoint
        _normalize_timm_dinov3_vit_config(config)
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
        _normalize_timm_dinov3_vit_config(config)
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
        _normalize_timm_dinov3_vit_config(config)
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
            specific = {
                "image_size": image_h if image_h == image_w else [image_h, image_w],
                "patch_size": cfg["patch_size"],
                "intermediate_size": cfg["intermediate_size"],
                "num_hidden_layers": cfg["num_hidden_layers"],
                "num_attention_heads": cfg["num_attention_heads"],
                "num_key_value_heads": cfg["num_attention_heads"],
                "hidden_act": cfg["hidden_act"],
                "layer_norm_eps": cfg["layer_norm_eps"],
                "rope_theta": cfg["rope_theta"],
                "num_register_tokens": cfg["num_register_tokens"],
                "query_bias": cfg["query_bias"],
                "key_bias": cfg["key_bias"],
                "value_bias": cfg["value_bias"],
                "proj_bias": cfg["proj_bias"],
                "mlp_bias": cfg["mlp_bias"],
                "use_gated_mlp": cfg["use_gated_mlp"],
            }
        elif config.model_type == "dinov3_convnext":
            from .convnext_builder import convnext_bundle_metadata, resolve_convnext_config

            cfg = resolve_convnext_config(config)
            specific = convnext_bundle_metadata(config)
            hidden_size = cfg["hidden_sizes"][-1]
            image_h, image_w = cfg["image_h"], cfg["image_w"]
            sequence_length = specific["num_feature_tokens"]
            architecture = "convnext"
        else:
            raise ValueError(f"Unsupported DINOv3 model_type: {config.model_type!r}")
        return {
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
            **specific,
        }


plugin = Dinov3Plugin()
