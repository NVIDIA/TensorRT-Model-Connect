# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT DINOv3 image-feature-extraction family.

This owner implements both architectures exposed by the Transformers DINOv3
model page.  Network construction uses TensorRT's Python Network API directly;
there is no ONNX export, parser, or fallback path.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import sys

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
from .config import ModelConfig


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


def _is_timm_dinov3_vit_config(config: ModelConfig) -> bool:
    model_type = str(getattr(config, "model_type", config) or "").lower()
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
        raise ValueError("DINOv3 hidden_size must be positive and divisible by num_attention_heads")
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


def load_vit_weights(model_dir: str, config: ModelConfig, *, precision: str) -> WeightDict:
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
            weights[f"{prefix}.{suffix}"] = _layer_weight(readers, layer, suffix, precision)

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

    pixel_values = network.add_input("pixel_values", trt.float32, (1, 3, image_h, image_w))
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

    cls = graph_ops.constant(network, weights["cls_token"], (1, 1, hidden_size), work_dtype)
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
            [weights[f"{prefix}.attention.{name}_proj.weight"] for name in ("q", "k", "v")],
            axis=1,
        )
        qkv = graph_ops.linear(network, normalized, qkv_weight, work_dtype)
        qkv_biases = [
            weights.get(f"{prefix}.attention.{name}_proj.bias") for name in ("q", "k", "v")
        ]
        if any(bias is not None for bias in qkv_biases):
            zero_bias = np.zeros(hidden_size, dtype=work_dtype)
            qkv = graph_ops.add_bias(
                network,
                qkv,
                np.concatenate([bias if bias is not None else zero_bias for bias in qkv_biases]),
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
            gate = graph_ops.activation(network, gate, cfg["hidden_act"], work_dtype)
            up = project("mlp", "up_proj", normalized)
            activated = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(
                0
            )
        else:
            activated = project("mlp", "up_proj", normalized)
            activated = graph_ops.activation(network, activated, cfg["hidden_act"], work_dtype)
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

    pooled = graph_ops.slice_tensor(network, last_hidden_state, (0, 0, 0), (1, 1, hidden_size))
    pooled = graph_ops.shuffle(network, pooled, reshape_dims=(1, hidden_size))
    pooled.name = "pooler_output"
    network.mark_output(pooled)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT DINOv3 ViT engine build failed")
    return bytes(plan)


name = "dinov3"
runtime_strategy = "dinov3_image_feature_extraction"
requires_tokenizer = False


def default_max_cache_length(config: ModelConfig) -> int:
    del config
    return 1


def matches(config) -> bool:
    """Return whether this module owns the resolved model config."""
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return (model_type or "").lower() in {
        "dinov3_vit",
        "dinov3_convnext",
        _TIMM_DINOV3_VIT_ARCHITECTURE,
    }


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
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
        return build_vit_engine(config, weights, precision=precision, verbose=verbose)
    if config.model_type == "dinov3_convnext":
        from .convnext_builder import build_convnext_engine

        return build_convnext_engine(config, weights, precision=precision, verbose=verbose)
    raise ValueError(f"Unsupported DINOv3 model_type: {config.model_type!r}")


def get_bundle_config_overrides(config: ModelConfig) -> dict:
    _normalize_timm_dinov3_vit_config(config)
    if config.model_type == "dinov3_vit":
        cfg = config.raw.get("_dinov3_config") or resolve_vit_config(config.raw)
        hidden_size = cfg["hidden_size"]
        sequence_length = (
            1
            + cfg["num_register_tokens"]
            + (cfg["image_h"] // cfg["patch_size"]) * (cfg["image_w"] // cfg["patch_size"])
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
        "runtime_strategy": runtime_strategy,
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
    """Build the complete dinov3 bundle inside its owning family module."""
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
        raise NotImplementedError("dinov3 does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("dinov3 does not use a decoder KV-cache runtime")

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
    max_cache_length = int(
        default_max_cache_length(config)
        if requested_cache_length is None
        else requested_cache_length
    )
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
        from . import graph_ops as family_graph_ops

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
        require_tensorrt_11_for_tensor_parallel(parallel, feature="dinov3 tensor-parallel builds")
        if quant_ctx is not None:
            raise ValueError("dinov3 tensor-parallel builds do not support quantization")

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
