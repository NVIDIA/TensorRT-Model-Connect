# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT builder for OpenPI vision and PaliGemma prefix prefill.

The engine is deliberately static and batch-one for the first qualification
target.  Three camera slots are evaluated as the SigLIP batch, flattened in
camera order, concatenated with 200 prompt embeddings, and passed through the
18-layer PaliGemma prefix expert.  Per-layer K/V outputs retain the upstream
``[batch, sequence, kv_heads, head_dim]`` layout and one-head MQA width.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .model_config import OpenPIProfile, get_profile
from .trt_plugin_loader import require_openpi_plugin_creator


# TensorRT tactic identifiers are deliberately treated as a versioned build
# contract.  On GB300 with TensorRT 11.2.0.113, the default FP32 patch-conv
# tactic differs slightly from the pinned JAX/XLA cuDNN stem.  The selected
# tactic below is bit-exact for the complete three-camera production shape and
# is the fastest of the four exact candidates measured by the qualification
# sweep.  Unknown TensorRT versions fail closed instead of silently emitting a
# numerically unqualified OpenPI engine.
_SIGLIP_STEM_TACTIC_CONTRACT = {
    "11.2.0.113": {
        "cache_key": "0x8e261b9b7b1cea14ade5994178f9a38a",
        "tactic_hash": 0x3C2307797D70E,
    }
}


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str


def prefix_cache_output_name(kind: str, layer: int) -> str:
    if kind not in {"k", "v"}:
        raise ValueError(f"cache kind must be 'k' or 'v', got {kind!r}")
    if layer < 0:
        raise ValueError("cache layer must be non-negative")
    return f"prefix_{kind}_{layer}"


def prefill_input_contract(profile: OpenPIProfile | str) -> tuple[TensorContract, ...]:
    cfg = get_profile(profile) if isinstance(profile, str) else profile
    image = cfg.vision.image_size
    return (
        TensorContract("pixel_values", (cfg.vision.num_image_slots, 3, image, image), "float32"),
        TensorContract("token_ids", (1, cfg.max_token_length), "int32"),
        TensorContract("prefix_mask", (1, cfg.prefix_length), "bool"),
        TensorContract("prefix_position_ids", (1, cfg.prefix_length), "int32"),
    )


def required_prefill_weight_shapes(profile: OpenPIProfile | str) -> dict[str, tuple[int, ...]]:
    """Return every canonical mapped weight consumed by the prefill engine."""

    cfg = get_profile(profile) if isinstance(profile, str) else profile
    vision = cfg.vision
    vision_head_dim = vision.width // vision.num_heads
    shapes: dict[str, tuple[int, ...]] = {
        "vision.patch_embedding.weight": (vision.width, 3, vision.patch_size, vision.patch_size),
        "vision.patch_embedding.bias": (vision.width,),
        "vision.position_embedding": (vision.tokens_per_image, vision.width),
        "vision.post_norm.weight": (vision.width,),
        "vision.post_norm.bias": (vision.width,),
        "vision.projector.weight": (vision.width, vision.output_width),
        "vision.projector.bias": (vision.output_width,),
        "prefix.embedding": (cfg.vocab_size, cfg.prefix.width),
        "prefix.final_norm.scale": (cfg.prefix.width,),
    }
    for layer in range(vision.depth):
        root = f"vision.layer.{layer}"
        for norm in ("norm1", "norm2"):
            shapes[f"{root}.{norm}.weight"] = (vision.width,)
            shapes[f"{root}.{norm}.bias"] = (vision.width,)
        for projection in ("q", "k", "v", "o"):
            shapes[f"{root}.attention.{projection}.weight"] = (
                vision.width,
                vision.num_heads * vision_head_dim,
            )
            shapes[f"{root}.attention.{projection}.bias"] = (vision.width,)
        shapes[f"{root}.mlp.fc1.weight"] = (vision.width, vision.mlp_dim)
        shapes[f"{root}.mlp.fc1.bias"] = (vision.mlp_dim,)
        shapes[f"{root}.mlp.fc2.weight"] = (vision.mlp_dim, vision.width)
        shapes[f"{root}.mlp.fc2.bias"] = (vision.width,)

    prefix = cfg.prefix
    for layer in range(prefix.depth):
        root = f"prefix.layer.{layer}"
        shapes[f"{root}.pre_attention_norm.scale"] = (prefix.width,)
        shapes[f"{root}.attention.q.weight"] = (prefix.width, prefix.attention_width)
        shapes[f"{root}.attention.k.weight"] = (prefix.width, prefix.kv_width)
        shapes[f"{root}.attention.v.weight"] = (prefix.width, prefix.kv_width)
        shapes[f"{root}.attention.o.weight"] = (prefix.attention_width, prefix.width)
        shapes[f"{root}.pre_ffw_norm.scale"] = (prefix.width,)
        shapes[f"{root}.mlp.gate.weight"] = (prefix.width, prefix.mlp_dim)
        shapes[f"{root}.mlp.up.weight"] = (prefix.width, prefix.mlp_dim)
        shapes[f"{root}.mlp.down.weight"] = (prefix.mlp_dim, prefix.width)
    return shapes


def validate_prefill_weights(
    weights: Mapping[str, np.ndarray], profile: OpenPIProfile | str
) -> None:
    expected = required_prefill_weight_shapes(profile)
    missing = sorted(set(expected) - set(weights))
    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    for name, shape in expected.items():
        if name not in weights:
            continue
        value = np.asarray(weights[name])
        if tuple(value.shape) != shape:
            shape_errors.append(f"{name}: expected {shape}, got {tuple(value.shape)}")
        if value.dtype.kind != "f" and value.dtype.name != "bfloat16":
            dtype_errors.append(f"{name}: expected floating point, got {value.dtype.name}")
    if missing or shape_errors or dtype_errors:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if shape_errors:
            details.append("shape=" + "; ".join(shape_errors))
        if dtype_errors:
            details.append("dtype=" + "; ".join(dtype_errors))
        raise ValueError("invalid OpenPI prefill weights: " + " | ".join(details))


def _shuffle(network, tensor, *, reshape: tuple[int, ...], transpose=None):
    layer = network.add_shuffle(tensor)
    if transpose is not None:
        layer.first_transpose = transpose
    layer.reshape_dims = reshape
    return layer.get_output(0)


def _rows_to_heads(network, rows, *, heads: int, head_dim: int):
    batch, sequence, _ = tuple(rows.shape)
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (batch, sequence, heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def _qkv_slice_to_heads(network, tensor, *, batch: int, sequence: int, heads: int, head_dim: int):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = (batch, sequence, heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def _heads_to_rows(network, heads_tensor):
    batch, heads, sequence, head_dim = tuple(heads_tensor.shape)
    layer = network.add_shuffle(heads_tensor)
    layer.first_transpose = (0, 2, 1, 3)
    layer.reshape_dims = (batch, sequence, heads * head_dim)
    return layer.get_output(0)


def _heads_to_cache(network, heads_tensor):
    batch, heads, sequence, head_dim = tuple(heads_tensor.shape)
    return _shuffle(
        network,
        heads_tensor,
        reshape=(batch, sequence, heads, head_dim),
        transpose=(0, 2, 1, 3),
    )


def _ranked_constant(network, tensor, value: float, *, dtype, ops):
    rank = len(tuple(tensor.shape))
    return ops.constant(
        network,
        np.array(value, dtype=np.float32),
        dtype=dtype,
        shape=(1,) * rank,
    )


def _install_siglip_stem_timing_cache(
    builder_config,
    weights,
    profile,
    *,
    workspace_bytes: int,
    verbose: bool,
    ops,
    trt,
) -> None:
    """Seed the production builder with the parity-qualified stem tactic."""

    vision = profile.vision
    production_shape = (
        vision.num_image_slots == 3
        and vision.image_size == 224
        and vision.patch_size == 14
        and vision.width == 1152
        and vision.tokens_per_image == 256
    )
    if not production_shape:
        # Tiny structural tests and future non-production profiles do not use
        # the GB300 tactic key, whose operation shape is intentionally exact.
        return
    version = str(getattr(trt, "__version__", ""))
    contract = _SIGLIP_STEM_TACTIC_CONTRACT.get(version)
    if contract is None:
        raise RuntimeError(
            "OpenPI SigLIP parity has no validated stem tactic for TensorRT "
            f"{version or '<unknown>'}; expected one of "
            f"{sorted(_SIGLIP_STEM_TACTIC_CONTRACT)}"
        )

    calibration_builder, calibration_network, calibration_config = ops.create_builder_context(
        verbose=verbose, workspace_bytes=workspace_bytes
    )
    calibration_config.clear_flag(trt.BuilderFlag.TF32)
    calibration_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    calibration_cache = calibration_config.create_timing_cache(b"")
    if not calibration_config.set_timing_cache(calibration_cache, False):
        raise RuntimeError("failed to attach OpenPI stem calibration timing cache")
    image = calibration_network.add_input(
        "pixel_values",
        trt.float32,
        (vision.num_image_slots, 3, vision.image_size, vision.image_size),
    )
    convolution = calibration_network.add_convolution_nd(
        image,
        vision.width,
        (vision.patch_size, vision.patch_size),
        trt.Weights(
            np.ascontiguousarray(weights["vision.patch_embedding.weight"], dtype=np.float32)
        ),
        trt.Weights(np.ascontiguousarray(weights["vision.patch_embedding.bias"], dtype=np.float32)),
    )
    convolution.stride_nd = (vision.patch_size, vision.patch_size)
    output = _shuffle(
        calibration_network,
        convolution.get_output(0),
        reshape=(vision.num_image_slots, vision.tokens_per_image, vision.width),
        transpose=(0, 2, 3, 1),
    )
    output.name = "stem"
    calibration_network.mark_output(output)
    calibration_plan = calibration_builder.build_serialized_network(
        calibration_network, calibration_config
    )
    if calibration_plan is None:
        raise RuntimeError("TensorRT OpenPI stem tactic calibration build failed")
    populated_cache = calibration_config.get_timing_cache()
    keys = {str(key) for key in populated_cache.queryKeys()}
    key_text = str(contract["cache_key"])
    if key_text not in keys:
        raise RuntimeError(
            "TensorRT OpenPI stem timing-cache key drifted: "
            f"expected {key_text}, observed {sorted(keys)}"
        )

    production_cache = builder_config.create_timing_cache(bytes(populated_cache.serialize()))
    key = trt.TimingCacheKey.parse(key_text)
    tactic_hash = int(contract["tactic_hash"])
    if not production_cache.update(key, trt.TimingCacheValue(tactic_hash, 0.0)):
        raise RuntimeError(f"failed to select OpenPI stem tactic {tactic_hash:#x} for {key_text}")
    builder_config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    if not builder_config.set_timing_cache(production_cache, False):
        raise RuntimeError("failed to attach parity-qualified OpenPI timing cache")


def _openpi_plugin_creator(name: str, *, trt):
    """Find an OpenPI creator through the deterministic build-time loader."""

    return require_openpi_plugin_creator(name, trt=trt)


def _create_openpi_rms_norm_plugin(*, epsilon: float, trt):
    """Create the family-owned TensorRT plugin used by production Gemma."""

    creator = _openpi_plugin_creator("OpenPIRmsNorm", trt=trt)
    epsilon_value = np.asarray([epsilon], dtype=np.float32)
    fields = trt.PluginFieldCollection(
        [trt.PluginField("epsilon", epsilon_value, trt.PluginFieldType.FLOAT32)]
    )
    plugin = creator.create_plugin("openpi_gemma_rms_norm", fields, trt.TensorRTPhase.BUILD)
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI RMSNorm TensorRT plugin")
    return plugin


def _create_openpi_siglip_layer_norm_plugin(*, epsilon: float, trt):
    """Create the fixed-shape SigLIP LayerNorm production plugin."""

    if epsilon != 1.0e-6:
        raise ValueError("OpenPI SigLIP LayerNorm plugin requires epsilon=1e-6")
    creator = _openpi_plugin_creator("OpenPISiglipLayerNorm", trt=trt)
    epsilon_value = np.asarray([epsilon], dtype=np.float32)
    fields = trt.PluginFieldCollection(
        [trt.PluginField("epsilon", epsilon_value, trt.PluginFieldType.FLOAT32)]
    )
    plugin = creator.create_plugin(
        "openpi_siglip_layer_norm",
        fields,
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI SigLIP LayerNorm TensorRT plugin")
    return plugin


def _apply_siglip_layer_norm_plugin(
    network,
    tensor,
    weight,
    bias,
    *,
    epsilon: float,
    name: str,
    ops,
    trt,
):
    """Apply the exact fixed-shape BF16 SigLIP LayerNorm production seam."""

    if epsilon != 1.0e-6:
        raise ValueError("OpenPI SigLIP LayerNorm plugin requires epsilon=1e-6")
    if tensor.dtype != trt.bfloat16 or tuple(tensor.shape) != (1, 256, 1152):
        raise ValueError("OpenPI SigLIP LayerNorm plugin requires BF16 activation [1,256,1152]")
    gamma_value = np.asarray(weight, dtype=np.float32)
    beta_value = np.asarray(bias, dtype=np.float32)
    if gamma_value.shape != (1152,) or beta_value.shape != (1152,):
        raise ValueError("OpenPI SigLIP LayerNorm plugin requires gamma/beta [1152]")
    if not name:
        raise ValueError("OpenPI SigLIP LayerNorm plugin requires a deterministic layer name")
    gamma = ops.constant(network, gamma_value, dtype=trt.bfloat16)
    beta = ops.constant(network, beta_value, dtype=trt.bfloat16)
    layer = network.add_plugin_v3(
        [tensor, gamma, beta],
        [],
        _create_openpi_siglip_layer_norm_plugin(epsilon=epsilon, trt=trt),
    )
    layer.name = name
    return layer.get_output(0)


def _create_openpi_siglip_attention_residual_plugin(*, trt):
    """Create the fixed-shape exact SigLIP attention-residual plugin."""

    creator = _openpi_plugin_creator("OpenPISiglipAttentionResidual", trt=trt)
    plugin = creator.create_plugin(
        "openpi_siglip_attention_residual",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI SigLIP attention-residual TensorRT plugin")
    return plugin


def _apply_siglip_attention_residual_plugin(
    network,
    hidden,
    norm_weight,
    norm_bias,
    qkv_weight,
    qkv_bias,
    output_weight,
    output_bias,
    *,
    name: str,
    ops,
    trt,
):
    """Apply the pinned BF16 SigLIP attention and its first residual add."""

    if hidden.dtype != trt.bfloat16 or tuple(hidden.shape) != (1, 256, 1152):
        raise ValueError(
            "OpenPI SigLIP attention-residual plugin requires BF16 hidden [1,256,1152]"
        )
    arrays = (
        np.asarray(norm_weight, dtype=np.float32),
        np.asarray(norm_bias, dtype=np.float32),
        np.asarray(qkv_weight, dtype=np.float32),
        np.asarray(qkv_bias, dtype=np.float32),
        np.asarray(output_weight, dtype=np.float32),
        np.asarray(output_bias, dtype=np.float32),
    )
    expected_shapes = (
        (1152,),
        (1152,),
        (1152, 3456),
        (3456,),
        (1152, 1152),
        (1152,),
    )
    if tuple(value.shape for value in arrays) != expected_shapes:
        raise ValueError(
            "OpenPI SigLIP attention-residual plugin requires norm [1152], "
            "QKV weight/bias [1152,3456]/[3456], and output weight/bias "
            "[1152,1152]/[1152]"
        )
    if not name:
        raise ValueError(
            "OpenPI SigLIP attention-residual plugin requires a deterministic layer name"
        )
    constants = [ops.constant(network, value, dtype=trt.bfloat16) for value in arrays]
    layer = network.add_plugin_v3(
        [hidden, *constants],
        [],
        _create_openpi_siglip_attention_residual_plugin(trt=trt),
    )
    layer.name = name
    return layer.get_output(0)


def _uses_siglip_attention_residual_plugin(hidden, cfg, *, trt) -> bool:
    """Gate the production plugin to its audited fixed geometry."""

    return (
        hidden.dtype == trt.bfloat16
        and tuple(hidden.shape) == (1, 256, 1152)
        and cfg.tokens_per_image == 256
        and cfg.width == 1152
        and cfg.num_heads == 16
        and cfg.width // cfg.num_heads == 72
    )


def _create_openpi_rope_plugin(*, trt):
    creator = _openpi_plugin_creator("OpenPIRopeQK", trt=trt)
    plugin = creator.create_plugin(
        "openpi_gemma_rope_qk",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI RoPE TensorRT plugin")
    return plugin


def _create_openpi_prefix_qk_plugin(*, trt):
    creator = _openpi_plugin_creator("OpenPIPrefixQK", trt=trt)
    plugin = creator.create_plugin(
        "openpi_gemma_prefix_qk",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI prefix-QK TensorRT plugin")
    return plugin


def _create_openpi_prefix_softmax_plugin(*, trt):
    creator = _openpi_plugin_creator("OpenPIPrefixSoftmax", trt=trt)
    plugin = creator.create_plugin(
        "openpi_gemma_prefix_softmax",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI prefix-softmax TensorRT plugin")
    return plugin


def _apply_prefix_qk(network, query, key, *, trt):
    """Issue XLA's flattened BF16-to-FP32 cuBLAS prefix contraction."""

    if (
        query.dtype == trt.bfloat16
        and key.dtype == trt.bfloat16
        and tuple(query.shape) == (1, 8, 968, 256)
        and tuple(key.shape) == (1, 1, 968, 256)
    ):
        plugin = _create_openpi_prefix_qk_plugin(trt=trt)
        raw_logits = network.add_plugin_v3([query, key], [], plugin).get_output(0)
        # The plugin exposes cuBLAS column-major [H*S,S] bytes as row-major
        # [S,S,H]. This shuffle is XLA's post-GEMM logical transpose.
        return _shuffle(
            network,
            raw_logits,
            reshape=(1, 8, 968, 968),
            transpose=(0, 3, 2, 1),
        )
    return None


def _apply_prefix_softmax(network, logits, attention_mask, *, trt):
    """Issue XLA's exact family-owned prefix softmax."""

    if (
        logits.dtype == trt.float32
        and attention_mask is not None
        and attention_mask.dtype == trt.bool
        and tuple(logits.shape) == (1, 8, 968, 968)
        and tuple(attention_mask.shape) == (1, 1, 968, 968)
    ):
        plugin = _create_openpi_prefix_softmax_plugin(trt=trt)
        return network.add_plugin_v3([logits, attention_mask], [], plugin).get_output(0)
    return None


def _apply_prefix_rope_qk(network, query, key, position_ids, *, profile, ops, trt):
    if (
        query.dtype == trt.bfloat16
        and key.dtype == trt.bfloat16
        and tuple(query.shape) == (1, 8, 968, 256)
        and tuple(key.shape) == (1, 1, 968, 256)
        and tuple(position_ids.shape) == (1, 968)
    ):
        plugin = _create_openpi_rope_plugin(trt=trt)
        layer = network.add_plugin_v3([query, key, position_ids], [], plugin)
        return layer.get_output(0), layer.get_output(1)

    max_positions = profile.prefix_length + profile.action_horizon
    query = ops.apply_rope(
        network,
        query,
        position_ids,
        max_positions=max_positions,
        head_dim=profile.prefix.head_dim,
    )
    key = ops.apply_rope(
        network,
        key,
        position_ids,
        max_positions=max_positions,
        head_dim=profile.prefix.head_dim,
    )
    return query, key


def _rms_norm_fp32(network, tensor, scale, *, epsilon: float, ops, trt):
    scale_value = np.asarray(scale, dtype=np.float32)
    if (
        tensor.dtype == trt.bfloat16
        and len(tuple(tensor.shape)) >= 2
        and tuple(tensor.shape)[-1] == 2048
        and scale_value.shape == (2048,)
    ):
        scale_tensor = ops.constant(
            network,
            scale_value,
            dtype=trt.bfloat16,
            shape=(2048,),
        )
        plugin = _create_openpi_rms_norm_plugin(epsilon=epsilon, trt=trt)
        return network.add_plugin_v3([tensor, scale_tensor], [], plugin).get_output(0)

    output_dtype = tensor.dtype
    x = ops.cast(network, tensor, trt.float32)
    axes = 1 << (len(tuple(x.shape)) - 1)
    square = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, axes, True).get_output(0)
    epsilon_tensor = _ranked_constant(network, variance, epsilon, dtype=trt.float32, ops=ops)
    denominator = network.add_elementwise(
        variance, epsilon_tensor, trt.ElementWiseOperation.SUM
    ).get_output(0)
    denominator = network.add_unary(denominator, trt.UnaryOperation.SQRT).get_output(0)
    denominator = network.add_unary(denominator, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(x, denominator, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    gamma_shape = (1,) * (len(tuple(x.shape)) - 1) + (scale_value.size,)
    gamma = ops.constant(network, scale_value, dtype=output_dtype, shape=gamma_shape)
    one = _ranked_constant(network, gamma, 1.0, dtype=output_dtype, ops=ops)
    gamma = network.add_elementwise(gamma, one, trt.ElementWiseOperation.SUM).get_output(0)
    gamma = ops.cast(network, gamma, trt.float32)
    normalized = network.add_elementwise(
        normalized, gamma, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return ops.cast(network, normalized, output_dtype)


def _gelu_tanh(network, tensor, *, ops, trt):
    """JAX's tanh GELU evaluated in the input activation dtype."""

    if tensor.dtype == trt.bfloat16:
        # TensorRT fuses the nominally BF16 pointwise chain and retains FP32
        # intermediates across operations. Pinned XLA rounds after every BF16
        # operation. Express those boundaries as observable FP32 -> BF16 ->
        # FP32 casts so TensorRT cannot legally reassociate them. This makes
        # the complete SigLIP FC1 -> GELU -> FC2 subgraph bit-exact while
        # retaining direct TensorRT layers and a BF16 public activation type.
        def rounded(value):
            return ops.cast(
                network,
                ops.cast(network, value, trt.bfloat16),
                trt.float32,
            )

        def constant_like(value, scalar: float):
            bf16_value = _ranked_constant(
                network,
                value,
                scalar,
                dtype=trt.bfloat16,
                ops=ops,
            )
            return ops.cast(network, bf16_value, trt.float32)

        x = ops.cast(network, tensor, trt.float32)
        x2 = rounded(network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0))
        x3 = rounded(network.add_elementwise(x2, x, trt.ElementWiseOperation.PROD).get_output(0))
        cubic = rounded(
            network.add_elementwise(
                x3,
                constant_like(x3, 0.044715),
                trt.ElementWiseOperation.PROD,
            ).get_output(0)
        )
        inner = rounded(
            network.add_elementwise(x, cubic, trt.ElementWiseOperation.SUM).get_output(0)
        )
        inner = rounded(
            network.add_elementwise(
                inner,
                constant_like(inner, math.sqrt(2.0 / math.pi)),
                trt.ElementWiseOperation.PROD,
            ).get_output(0)
        )
        tanh = rounded(network.add_activation(inner, trt.ActivationType.TANH).get_output(0))
        factor = rounded(
            network.add_elementwise(
                tanh,
                constant_like(tanh, 1.0),
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
        )
        factor = rounded(
            network.add_elementwise(
                factor,
                constant_like(factor, 0.5),
                trt.ElementWiseOperation.PROD,
            ).get_output(0)
        )
        output = rounded(
            network.add_elementwise(x, factor, trt.ElementWiseOperation.PROD).get_output(0)
        )
        return ops.cast(network, output, trt.bfloat16)

    x = tensor
    dtype = tensor.dtype
    x2 = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
    x3 = network.add_elementwise(x2, x, trt.ElementWiseOperation.PROD).get_output(0)
    cubic_scale = _ranked_constant(network, x3, 0.044715, dtype=dtype, ops=ops)
    cubic = network.add_elementwise(x3, cubic_scale, trt.ElementWiseOperation.PROD).get_output(0)
    inner = network.add_elementwise(x, cubic, trt.ElementWiseOperation.SUM).get_output(0)
    root_scale = _ranked_constant(network, inner, math.sqrt(2.0 / math.pi), dtype=dtype, ops=ops)
    inner = network.add_elementwise(inner, root_scale, trt.ElementWiseOperation.PROD).get_output(0)
    tanh = network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
    one = _ranked_constant(network, tanh, 1.0, dtype=dtype, ops=ops)
    factor = network.add_elementwise(tanh, one, trt.ElementWiseOperation.SUM).get_output(0)
    half = _ranked_constant(network, factor, 0.5, dtype=dtype, ops=ops)
    factor = network.add_elementwise(factor, half, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(x, factor, trt.ElementWiseOperation.PROD).get_output(0)


def _softmax_attention(
    network,
    q,
    k,
    v,
    mask,
    *,
    head_dim: int,
    fp32_logits: bool,
    scale_with_division: bool,
    use_einsum: bool = False,
    ops,
    trt,
):
    """Attention with the exact dtype and query scaling used upstream."""

    q_dtype = q.dtype
    use_siglip_bf16_scaling = use_einsum and q_dtype == trt.bfloat16 and not fp32_logits
    if scale_with_division and use_siglip_bf16_scaling:
        # Pinned XLA strength-reduces BF16 division by sqrt(head_dim) to a
        # rounded reciprocal multiply. A TensorRT elementwise layer is folded
        # into QK and retains extra precision. A scalar native einsum is the
        # smallest direct-TRT contraction that preserves the BF16 boundary.
        scale = ops.constant(
            network,
            np.asarray([1.0 / math.sqrt(head_dim)], dtype=np.float32),
            dtype=q_dtype,
        )
        q = network.add_einsum([q, scale], "bhqd,z->bhqd").get_output(0)
    elif scale_with_division:
        scale = _ranked_constant(network, q, math.sqrt(head_dim), dtype=q_dtype, ops=ops)
        q = network.add_elementwise(q, scale, trt.ElementWiseOperation.DIV).get_output(0)
    else:
        scale = _ranked_constant(network, q, 1.0 / math.sqrt(head_dim), dtype=q_dtype, ops=ops)
        q = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)
    logits = _apply_prefix_qk(network, q, k, trt=trt) if fp32_logits else None
    if logits is None:
        query = ops.cast(network, q, trt.float32) if fp32_logits else q
        key = ops.cast(network, k, trt.float32) if fp32_logits else k
        if use_einsum:
            logits = network.add_einsum([query, key], "bhqd,bhkd->bhqk").get_output(0)
        else:
            logits = network.add_matrix_multiply(
                query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
            ).get_output(0)
    prefix_probabilities = (
        _apply_prefix_softmax(network, logits, mask, trt=trt) if fp32_logits else None
    )
    if mask is not None and prefix_probabilities is None:
        negative = _ranked_constant(network, logits, -2.3819763e38, dtype=logits.dtype, ops=ops)
        logits = network.add_select(mask, logits, negative).get_output(0)
    softmax_axes = 1 << (len(tuple(logits.shape)) - 1)
    if prefix_probabilities is not None:
        probabilities = prefix_probabilities
    else:
        # TensorRT's native BF16 softmax is bit-exact with the pinned XLA
        # result in the complete SigLIP attention graph. The Gemma prefix
        # shape above instead needs its family-owned exact reduction plugin.
        softmax = network.add_softmax(logits)
        softmax.axes = softmax_axes
        probabilities = ops.cast(network, softmax.get_output(0), q_dtype)
    if use_einsum:
        context = network.add_einsum([probabilities, v], "bhqk,bhkd->bhqd").get_output(0)
    else:
        context = network.add_matrix_multiply(
            probabilities,
            trt.MatrixOperation.NONE,
            v,
            trt.MatrixOperation.NONE,
        ).get_output(0)
    return _heads_to_rows(network, context)


def _prefix_attention_mask(network, prefix_mask, *, sequence: int, trt):
    query_mask = _shuffle(network, prefix_mask, reshape=(1, 1, sequence, 1))
    key_mask = _shuffle(network, prefix_mask, reshape=(1, 1, 1, sequence))
    return network.add_elementwise(query_mask, key_mask, trt.ElementWiseOperation.AND).get_output(0)


def _einsum_linear_3d(network, tensor, weight, bias, *, ops):
    """Apply Flax/JAX ``bsi,io->bso`` with TensorRT's native einsum layer."""
    weight_value = np.asarray(weight)
    if len(tuple(tensor.shape)) != 3 or weight_value.ndim != 2:
        raise ValueError("OpenPI SigLIP einsum linear requires rank-3 input and rank-2 weight")
    rhs = ops.constant(network, weight_value, dtype=tensor.dtype)
    output = network.add_einsum([tensor, rhs], "bsi,io->bso").get_output(0)
    if bias is not None:
        bias_value = ops.constant(
            network,
            np.asarray(bias).reshape(1, 1, weight_value.shape[1]),
            dtype=tensor.dtype,
        )
        output = network.add_elementwise(
            output, bias_value, ops.trt.ElementWiseOperation.SUM
        ).get_output(0)
    return output


def _vision_transformer_camera(
    network,
    hidden,
    weights,
    cfg,
    *,
    camera_index,
    activation_dtype,
    ops,
    trt,
):
    """Run one SigLIP camera exactly as the pinned OpenPI image loop does."""

    if not isinstance(camera_index, int) or not 0 <= camera_index < cfg.num_image_slots:
        raise ValueError(f"invalid OpenPI SigLIP camera index {camera_index!r}")
    head_dim = cfg.width // cfg.num_heads
    for layer in range(cfg.depth):
        root = f"vision.layer.{layer}"
        # XLA folds the three Flax DenseGeneral projections into one
        # [tokens,width] @ [width,3*width] BF16 GEMM. Keep each camera at
        # M=tokens: upstream calls PaliGemma.img independently for every image,
        # and flattening the three cameras to M=3*tokens selects a different
        # reduction tactic with measurable BF16 drift.
        qkv_weight = np.concatenate(
            [np.asarray(weights[f"{root}.attention.{name}.weight"]) for name in ("q", "k", "v")],
            axis=1,
        )
        qkv_bias = np.concatenate(
            [np.asarray(weights[f"{root}.attention.{name}.bias"]) for name in ("q", "k", "v")],
            axis=0,
        )
        if _uses_siglip_attention_residual_plugin(hidden, cfg, trt=trt):
            hidden = _apply_siglip_attention_residual_plugin(
                network,
                hidden,
                weights[f"{root}.norm1.weight"],
                weights[f"{root}.norm1.bias"],
                qkv_weight,
                qkv_bias,
                weights[f"{root}.attention.o.weight"],
                weights[f"{root}.attention.o.bias"],
                name=(f"openpi/siglip/camera_{camera_index}/layer_{layer:02d}/attention_residual"),
                ops=ops,
                trt=trt,
            )
        else:
            normed = _apply_siglip_layer_norm_plugin(
                network,
                hidden,
                weights[f"{root}.norm1.weight"],
                weights[f"{root}.norm1.bias"],
                epsilon=1e-6,
                name=f"openpi/siglip/camera_{camera_index}/layer_{layer:02d}/norm1",
                ops=ops,
                trt=trt,
            )
            qkv_rhs = ops.constant(
                network,
                qkv_weight.reshape(cfg.width, 3, cfg.num_heads, head_dim),
                dtype=activation_dtype,
            )
            qkv = network.add_einsum([normed, qkv_rhs], "bsi,iqhd->bsqhd").get_output(0)
            qkv_bias_value = ops.constant(
                network,
                qkv_bias.reshape(1, 1, 3, cfg.num_heads, head_dim),
                dtype=activation_dtype,
            )
            qkv = network.add_elementwise(
                qkv, qkv_bias_value, trt.ElementWiseOperation.SUM
            ).get_output(0)
            projections = []
            for projection in range(3):
                value = network.add_slice(
                    qkv,
                    (0, 0, projection, 0, 0),
                    (1, cfg.tokens_per_image, 1, cfg.num_heads, head_dim),
                    (1, 1, 1, 1, 1),
                ).get_output(0)
                projections.append(
                    _qkv_slice_to_heads(
                        network,
                        value,
                        batch=1,
                        sequence=cfg.tokens_per_image,
                        heads=cfg.num_heads,
                        head_dim=head_dim,
                    )
                )
            attended = _softmax_attention(
                network,
                *projections,
                None,
                head_dim=head_dim,
                fp32_logits=False,
                scale_with_division=True,
                use_einsum=True,
                ops=ops,
                trt=trt,
            )
            attended = _einsum_linear_3d(
                network,
                attended,
                weights[f"{root}.attention.o.weight"],
                weights[f"{root}.attention.o.bias"],
                ops=ops,
            )
            hidden = ops.gated_residual(network, hidden, attended)

        normed = _apply_siglip_layer_norm_plugin(
            network,
            hidden,
            weights[f"{root}.norm2.weight"],
            weights[f"{root}.norm2.bias"],
            epsilon=1e-6,
            name=f"openpi/siglip/camera_{camera_index}/layer_{layer:02d}/norm2",
            ops=ops,
            trt=trt,
        )
        mlp = _einsum_linear_3d(
            network,
            normed,
            weights[f"{root}.mlp.fc1.weight"],
            weights[f"{root}.mlp.fc1.bias"],
            ops=ops,
        )
        mlp = _gelu_tanh(network, mlp, ops=ops, trt=trt)
        mlp = _einsum_linear_3d(
            network,
            mlp,
            weights[f"{root}.mlp.fc2.weight"],
            weights[f"{root}.mlp.fc2.bias"],
            ops=ops,
        )
        hidden = ops.gated_residual(network, hidden, mlp)

    hidden = _apply_siglip_layer_norm_plugin(
        network,
        hidden,
        weights["vision.post_norm.weight"],
        weights["vision.post_norm.bias"],
        epsilon=1e-6,
        name=f"openpi/siglip/camera_{camera_index}/post_norm",
        ops=ops,
        trt=trt,
    )
    return _einsum_linear_3d(
        network,
        hidden,
        weights["vision.projector.weight"],
        weights["vision.projector.bias"],
        ops=ops,
    )


def _vision_encoder(
    network,
    pixel_values,
    weights,
    profile,
    *,
    activation_dtype,
    ops,
    trt,
):
    cfg = profile.vision
    patch_weight = np.ascontiguousarray(weights["vision.patch_embedding.weight"], dtype=np.float32)
    patch_bias = np.ascontiguousarray(weights["vision.patch_embedding.bias"], dtype=np.float32)
    conv = network.add_convolution_nd(
        pixel_values,
        cfg.width,
        (cfg.patch_size, cfg.patch_size),
        trt.Weights(patch_weight),
        trt.Weights(patch_bias),
    )
    conv.stride_nd = (cfg.patch_size, cfg.patch_size)
    hidden = _shuffle(
        network,
        conv.get_output(0),
        reshape=(cfg.num_image_slots, cfg.tokens_per_image, cfg.width),
        transpose=(0, 2, 3, 1),
    )
    position = ops.constant(
        network,
        np.asarray(weights["vision.position_embedding"], dtype=np.float32).reshape(
            1, cfg.tokens_per_image, cfg.width
        ),
        dtype=trt.float32,
    )
    hidden = network.add_elementwise(hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
    hidden = ops.cast(network, hidden, activation_dtype)

    camera_outputs = []
    for camera in range(cfg.num_image_slots):
        camera_hidden = network.add_slice(
            hidden,
            (camera, 0, 0),
            (1, cfg.tokens_per_image, cfg.width),
            (1, 1, 1),
        ).get_output(0)
        camera_outputs.append(
            _vision_transformer_camera(
                network,
                camera_hidden,
                weights,
                cfg,
                camera_index=camera,
                activation_dtype=activation_dtype,
                ops=ops,
                trt=trt,
            )
        )

    concatenation = network.add_concatenation(camera_outputs)
    concatenation.axis = 0
    return _shuffle(
        network,
        concatenation.get_output(0),
        reshape=(1, cfg.num_image_slots * cfg.tokens_per_image, cfg.output_width),
    )


def _prompt_embeddings(network, token_ids, weights, profile, *, activation_dtype, ops, trt):
    cfg = profile.prefix
    embedding = ops.constant(
        network,
        np.asarray(weights["prefix.embedding"], dtype=np.float32),
        dtype=activation_dtype,
    )
    hidden = network.add_gather(embedding, token_ids, 0).get_output(0)
    scale = _ranked_constant(network, hidden, math.sqrt(cfg.width), dtype=activation_dtype, ops=ops)
    hidden = network.add_elementwise(hidden, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return hidden


def _prefix_expert(
    network,
    hidden,
    prefix_mask,
    position_ids,
    weights,
    profile,
    *,
    ops,
    trt,
):
    cfg = profile.prefix
    attention_mask = _prefix_attention_mask(
        network, prefix_mask, sequence=profile.prefix_length, trt=trt
    )
    caches = []
    for layer in range(cfg.depth):
        root = f"prefix.layer.{layer}"
        normed = _rms_norm_fp32(
            network,
            hidden,
            weights[f"{root}.pre_attention_norm.scale"],
            epsilon=profile.rms_norm_epsilon,
            ops=ops,
            trt=trt,
        )
        q = ops.linear(network, normed, weights[f"{root}.attention.q.weight"])
        k = ops.linear(network, normed, weights[f"{root}.attention.k.weight"])
        v = ops.linear(network, normed, weights[f"{root}.attention.v.weight"])
        q = _rows_to_heads(network, q, heads=cfg.num_heads, head_dim=cfg.head_dim)
        k = _rows_to_heads(network, k, heads=cfg.num_kv_heads, head_dim=cfg.head_dim)
        v = _rows_to_heads(network, v, heads=cfg.num_kv_heads, head_dim=cfg.head_dim)
        q, k = _apply_prefix_rope_qk(
            network,
            q,
            k,
            position_ids,
            profile=profile,
            ops=ops,
            trt=trt,
        )
        cache_k = _heads_to_cache(network, k)
        cache_v = _heads_to_cache(network, v)
        caches.append((cache_k, cache_v))

        attended = _softmax_attention(
            network,
            q,
            k,
            v,
            attention_mask,
            head_dim=cfg.head_dim,
            fp32_logits=True,
            scale_with_division=False,
            ops=ops,
            trt=trt,
        )
        attended = ops.linear(network, attended, weights[f"{root}.attention.o.weight"])
        hidden = ops.gated_residual(network, hidden, attended)

        normed = _rms_norm_fp32(
            network,
            hidden,
            weights[f"{root}.pre_ffw_norm.scale"],
            epsilon=profile.rms_norm_epsilon,
            ops=ops,
            trt=trt,
        )
        gate = ops.linear(network, normed, weights[f"{root}.mlp.gate.weight"])
        gate = _gelu_tanh(network, gate, ops=ops, trt=trt)
        up = ops.linear(network, normed, weights[f"{root}.mlp.up.weight"])
        mlp = network.add_elementwise(gate, up, trt.ElementWiseOperation.PROD).get_output(0)
        mlp = ops.linear(network, mlp, weights[f"{root}.mlp.down.weight"])
        hidden = ops.gated_residual(network, hidden, mlp)

    _ = _rms_norm_fp32(
        network,
        hidden,
        weights["prefix.final_norm.scale"],
        epsilon=profile.rms_norm_epsilon,
        ops=ops,
        trt=trt,
    )
    return caches


def build_prefill_engine(
    weights: Mapping[str, np.ndarray],
    profile: OpenPIProfile | str,
    *,
    precision: str = "bf16",
    workspace_bytes: int = 1 << 32,
    verbose: bool = False,
) -> bytes:
    """Build the fixed-shape OpenPI prefill plan with direct TensorRT APIs."""

    cfg = get_profile(profile) if isinstance(profile, str) else profile
    validate_prefill_weights(weights, cfg)

    # Kept lazy so config inspection and reference tests do not require a
    # TensorRT installation on the host.
    from . import graph_ops as ops

    trt = ops.trt
    _constant_dtype, activation_dtype = ops.precision_types(precision)
    builder, network, builder_config = ops.create_builder_context(
        verbose=verbose, workspace_bytes=workspace_bytes
    )
    # Flax promotes the BF16 SigLIP patch kernel to FP32, and pinned XLA/cuDNN
    # evaluates that convolution with IEEE FP32 products. TensorRT's default
    # TF32 mode changes enough stem values to cross BF16 rounding boundaries,
    # so disable TF32 for this parity-qualified prefill plan.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    _install_siglip_stem_timing_cache(
        builder_config,
        weights,
        cfg,
        workspace_bytes=workspace_bytes,
        verbose=verbose,
        ops=ops,
        trt=trt,
    )
    contracts = prefill_input_contract(cfg)
    pixel_values = network.add_input(contracts[0].name, trt.float32, contracts[0].shape)
    token_ids = network.add_input(contracts[1].name, trt.int32, contracts[1].shape)
    prefix_mask = network.add_input(contracts[2].name, trt.bool, contracts[2].shape)
    position_ids = network.add_input(contracts[3].name, trt.int32, contracts[3].shape)

    vision_tokens = _vision_encoder(
        network,
        pixel_values,
        weights,
        cfg,
        activation_dtype=activation_dtype,
        ops=ops,
        trt=trt,
    )
    # This output is always present so a runtime can bind a persistent device
    # buffer without rebuilding or switching plans for qualification. Normal
    # inference never copies it to the host. Adding it changes the serialized
    # engine contract, so existing production prefill plans must be rebuilt.
    vision_output = _shuffle(
        network,
        vision_tokens,
        reshape=(
            1,
            cfg.vision.num_image_slots,
            cfg.vision.tokens_per_image,
            cfg.vision.output_width,
        ),
    )
    vision_output.name = "vision_tokens"
    network.mark_output(vision_output)
    prompt_tokens = _prompt_embeddings(
        network,
        token_ids,
        weights,
        cfg,
        activation_dtype=activation_dtype,
        ops=ops,
        trt=trt,
    )
    concat = network.add_concatenation([vision_tokens, prompt_tokens])
    concat.axis = 1
    prefix_hidden = concat.get_output(0)
    caches = _prefix_expert(
        network,
        prefix_hidden,
        prefix_mask,
        position_ids,
        weights,
        cfg,
        ops=ops,
        trt=trt,
    )
    for layer, (cache_k, cache_v) in enumerate(caches):
        for kind, cache in (("k", cache_k), ("v", cache_v)):
            cache.name = prefix_cache_output_name(kind, layer)
            network.mark_output(cache)

    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT OpenPI prefill engine build failed")
    return bytes(plan)
