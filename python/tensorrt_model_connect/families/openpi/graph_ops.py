# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph primitives for the OpenPI flow-policy family.

The OpenPI model proof projects one family at a time, so the implementation is
intentionally family-owned instead of importing a sibling family's graph
helpers. All networks assembled from these helpers are strongly typed and keep
the compact one-head Gemma K/V representation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tensorrt_model_connect import trt_compat
from .numerics import sinusoidal_inverse_periods_numpy
from .trt_plugin_loader import require_openpi_plugin_creator

trt = trt_compat.get_trt()


def create_builder_context(*, verbose: bool = False, workspace_bytes: int = 1 << 32):
    """Create a strongly typed TensorRT builder/network/config tuple."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    return builder, network, config


def precision_types(precision: str) -> tuple[np.dtype, Any]:
    """Return constant-storage and TensorRT activation types."""
    if precision == "fp32":
        return np.dtype(np.float32), trt.float32
    if precision == "fp16":
        return np.dtype(np.float16), trt.float16
    if precision == "bf16":
        # NumPy has no portable BF16 array type. Constants enter as FP32 and
        # are explicitly cast; TensorRT folds the casts while building.
        return np.dtype(np.float32), trt.bfloat16
    raise ValueError(f"Unsupported OpenPI precision {precision!r}")


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def constant(network, value: np.ndarray | float | int, *, dtype=None, shape=None):
    array = np.asarray(value)
    if dtype is not None:
        if dtype == trt.float16:
            array = array.astype(np.float16)
        elif dtype in (trt.float32, trt.bfloat16):
            array = array.astype(np.float32)
        elif dtype == trt.int32:
            array = array.astype(np.int32)
        elif dtype == trt.int64:
            array = array.astype(np.int64)
        elif dtype == trt.bool:
            array = array.astype(np.bool_)
    if shape is not None:
        array = array.reshape(tuple(shape))
    elif array.ndim == 0:
        # Scalar TensorRT constants are not supported uniformly across the
        # TensorRT versions exercised by this repository. A single-element
        # vector preserves broadcasting semantics.
        array = array.reshape((1,))
    array = np.ascontiguousarray(array)
    out = network.add_constant(tuple(array.shape), trt.Weights(array)).get_output(0)
    if dtype is not None and out.dtype != dtype:
        out = cast(network, out, dtype)
    return out


def scalar_like(network, tensor, value: float, *, dtype=None):
    """Create a singleton-per-axis scalar compatible with strict TRT ranks."""
    target_dtype = tensor.dtype if dtype is None else dtype
    rank = len(tuple(tensor.shape))
    return constant(
        network,
        np.array(value, dtype=np.float32),
        dtype=target_dtype,
        shape=(1,) * rank,
    )


def linear(network, inp, weight: np.ndarray, bias: np.ndarray | None = None, *, dtype=None):
    """Apply an input-major ``[..., in] @ [in, out]`` projection."""
    weight = np.asarray(weight)
    if weight.ndim != 2:
        raise ValueError(f"OpenPI linear weight must be rank 2, got {weight.shape}")
    input_rank = len(tuple(inp.shape))
    if input_rank < 2:
        raise ValueError(f"OpenPI linear input must have rank >= 2, got {inp.shape}")
    target_dtype = inp.dtype if dtype is None else dtype
    # TensorRT matrix-multiply requires both operands to have the same rank.
    # Leading singleton dimensions retain NumPy-style broadcasting over batch
    # and sequence axes without duplicating baked weights.
    weight_shape = (1,) * (input_rank - 2) + tuple(weight.shape)
    rhs = constant(network, weight.reshape(weight_shape), dtype=target_dtype)
    layer = network.add_matrix_multiply(
        inp, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    )
    out = layer.get_output(0)
    if bias is not None:
        rank = len(tuple(out.shape))
        bias_shape = (1,) * max(rank - 1, 0) + (weight.shape[1],)
        bias_tensor = constant(network, np.asarray(bias).reshape(bias_shape), dtype=target_dtype)
        out = network.add_elementwise(out, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    return out


def _xla_bf16_m1_linear(
    network,
    inp,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
):
    """Reproduce XLA's BF16 lowering for a one-row dot product.

    Pinned JAX/XLA does not dispatch the adaptive-RMS condition projection
    (``[1, K] @ [K, N]``) to a GEMM. It rounds each scalar product to BF16,
    converts those products to FP32, reduces K, and casts the sum back to BF16
    before adding the BF16 bias. A TensorRT matrix multiply instead accumulates
    unrounded products and creates a persistent one-ULP mismatch at every
    adaptive norm. Keep this specialized path limited to the audited M=1 BF16
    shape; normal token projections continue to use TensorRT GEMM tactics.
    """
    weight = np.asarray(weight)
    if weight.ndim != 2:
        raise ValueError(f"OpenPI XLA-compatible linear weight must be rank 2, got {weight.shape}")
    if tuple(inp.shape) != (1, weight.shape[0]):
        raise ValueError(
            "OpenPI XLA-compatible linear requires [1, K] input, got "
            f"{tuple(inp.shape)} for weight {weight.shape}"
        )
    if inp.dtype != trt.bfloat16:
        raise ValueError("OpenPI XLA-compatible linear requires BF16 input")

    lhs_shuffle = network.add_shuffle(inp)
    lhs_shuffle.reshape_dims = (weight.shape[0], 1)
    rhs = constant(network, weight, dtype=trt.bfloat16)
    products = network.add_elementwise(
        lhs_shuffle.get_output(0), rhs, trt.ElementWiseOperation.PROD
    ).get_output(0)
    products = cast(network, products, trt.float32)
    reduced = network.add_reduce(
        products,
        trt.ReduceOperation.SUM,
        1 << 0,
        True,
    ).get_output(0)
    out = cast(network, reduced, trt.bfloat16)
    if bias is not None:
        bias_tensor = constant(
            network,
            np.asarray(bias).reshape(1, weight.shape[1]),
            dtype=trt.bfloat16,
        )
        out = network.add_elementwise(out, bias_tensor, trt.ElementWiseOperation.SUM).get_output(0)
    return out


def _xla_fp32_silu(network, inp):
    """Reproduce pinned XLA's explicit FP32 SiLU decomposition."""
    if inp.dtype != trt.float32:
        raise ValueError("OpenPI XLA-compatible SiLU requires FP32 input")
    one = scalar_like(network, inp, 1.0, dtype=trt.float32)
    negative = network.add_unary(inp, trt.UnaryOperation.NEG).get_output(0)
    exponential = network.add_unary(negative, trt.UnaryOperation.EXP).get_output(0)
    denominator = network.add_elementwise(
        exponential, one, trt.ElementWiseOperation.SUM
    ).get_output(0)
    sigmoid = network.add_elementwise(one, denominator, trt.ElementWiseOperation.DIV).get_output(0)
    return network.add_elementwise(inp, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


# The iterative FP32 schedule values are encoded as raw bits so these
# corrections cannot drift through host-language decimal conversions. The
# replacement values are exactly representable BF16 values stored as FP32.
_DROID_TIME_CONDITION_TIMESTEP_BITS = (
    0x3F800000,
    0x3F666666,
    0x3F4CCCCC,
    0x3F333332,
    0x3F199998,
    0x3EFFFFFD,
    0x3ECCCCCA,
    0x3E999997,
    0x3E4CCCC8,
    0x3DCCCCC3,
)
_DROID_TIME_CONDITION_CORRECTIONS = (
    (0, ((573, 0xB6CF0000),)),
    (1, ((750, 0xB9AA0000),)),
    (3, ((275, 0xBAEE0000), (558, 0xBBC00000))),
    (7, ((101, 0x38C40000), (627, 0x36FA0000))),
)

# TensorRT's BF16 product-plus-reduce lowering matches the pinned XLA
# adaptive-modulation projections at all but four of 10 * 36 * 3072 audited
# outputs. Each entry is (step, layer, role, feature, exact BF16-as-FP32 bits).
# The builder separately binds every affected weight and bias by SHA-256.
_DROID_ADAPTIVE_MODULATION_CORRECTIONS = (
    (2, 12, "ffw", 84, 0xBB740000),
    (6, 6, "ffw", 1386, 0x3C260000),
    (7, 14, "ffw", 2797, 0x3B1B0000),
    (9, 11, "attn", 788, 0xBF0A0000),
)


def correct_droid_time_condition_bf16_boundaries(network, condition, timestep):
    """Correct the six audited BF16 midpoint crossings in π0.5-DROID.

    Pinned XLA lowers each FP32 time-MLP dot as an elementwise product plus a
    fixed reduction, whereas TensorRT 11.2 selects cuBLAS GEMV. Across the
    complete ten-step schedule, only these six condition elements land on the
    opposite side of a BF16 midpoint. All adaptive-norm consumers cast the
    FP32 condition to BF16, so replacing those six values reproduces the
    original boundary without carrying a JAX-generated table into runtime.
    """
    if tuple(condition.shape) != (1, 1024) or tuple(timestep.shape) != (1,):
        raise ValueError(
            "OpenPI DROID time-condition corrections require condition [1,1024] and timestep [1]"
        )
    if condition.dtype != trt.float32 or timestep.dtype != trt.float32:
        raise ValueError("OpenPI DROID time-condition corrections require FP32 inputs")

    corrected = condition
    for step, replacements in _DROID_TIME_CONDITION_CORRECTIONS:
        timestep_value = np.asarray(
            [_DROID_TIME_CONDITION_TIMESTEP_BITS[step]], dtype=np.uint32
        ).view(np.float32)
        is_step = network.add_elementwise(
            timestep,
            constant(network, timestep_value, dtype=trt.float32),
            trt.ElementWiseOperation.EQUAL,
        ).get_output(0)
        step_mask = network.add_shuffle(is_step)
        step_mask.reshape_dims = (1, 1)

        feature_mask = np.zeros((1, 1024), dtype=np.bool_)
        replacement_bits = np.zeros((1, 1024), dtype=np.uint32)
        for feature, value_bits in replacements:
            feature_mask[0, feature] = True
            replacement_bits[0, feature] = value_bits
        correction_mask = network.add_elementwise(
            step_mask.get_output(0),
            constant(network, feature_mask, dtype=trt.bool),
            trt.ElementWiseOperation.AND,
        ).get_output(0)
        select = network.add_select(
            correction_mask,
            constant(network, replacement_bits.view(np.float32), dtype=trt.float32),
            corrected,
        )
        select.name = f"openpi_droid_time_condition_bf16_correction_step_{step}"
        corrected = select.get_output(0)
    return corrected


def correct_droid_adaptive_modulation_bf16_boundaries(
    network,
    modulation,
    timestep,
    *,
    layer: int,
    role: str,
):
    """Correct four audited BF16 reduction midpoints in DROID modulation."""
    if tuple(modulation.shape) != (1, 3072) or tuple(timestep.shape) != (1,):
        raise ValueError(
            "OpenPI DROID adaptive-modulation corrections require modulation "
            "[1,3072] and timestep [1]"
        )
    if modulation.dtype != trt.bfloat16 or timestep.dtype != trt.float32:
        raise ValueError(
            "OpenPI DROID adaptive-modulation corrections require BF16 modulation and FP32 timestep"
        )
    if not isinstance(layer, int) or layer < 0 or layer >= 18:
        raise ValueError("OpenPI DROID adaptive-modulation layer must be in [0, 18)")
    if role not in {"attn", "ffw"}:
        raise ValueError("OpenPI DROID adaptive-modulation role must be 'attn' or 'ffw'")

    corrected = modulation
    for (
        step,
        correction_layer,
        correction_role,
        feature,
        value_bits,
    ) in _DROID_ADAPTIVE_MODULATION_CORRECTIONS:
        if correction_layer != layer or correction_role != role:
            continue
        timestep_value = np.asarray(
            [_DROID_TIME_CONDITION_TIMESTEP_BITS[step]], dtype=np.uint32
        ).view(np.float32)
        is_step = network.add_elementwise(
            timestep,
            constant(network, timestep_value, dtype=trt.float32),
            trt.ElementWiseOperation.EQUAL,
        ).get_output(0)
        step_mask = network.add_shuffle(is_step)
        step_mask.reshape_dims = (1, 1)

        feature_mask = np.zeros((1, 3072), dtype=np.bool_)
        feature_mask[0, feature] = True
        correction_mask = network.add_elementwise(
            step_mask.get_output(0),
            constant(network, feature_mask, dtype=trt.bool),
            trt.ElementWiseOperation.AND,
        ).get_output(0)
        replacement = np.zeros((1, 3072), dtype=np.uint32)
        replacement[0, feature] = value_bits
        select = network.add_select(
            correction_mask,
            constant(network, replacement.view(np.float32), dtype=trt.bfloat16),
            corrected,
        )
        select.name = (
            f"openpi_droid_adaptive_modulation_bf16_correction_step_{step}_layer_{layer}_{role}"
        )
        corrected = select.get_output(0)
    return corrected


def materialize_identity(network, value, *, name: str | None = None):
    """Materialize a tensor dtype boundary with a rank-preserving IEinSum."""
    rank = len(tuple(value.shape))
    labels = "abcdefghijklmno"[:rank]
    identity = constant(
        network,
        np.asarray([1.0], dtype=np.float32),
        dtype=value.dtype,
    )
    return network.add_einsum([value, identity], f"{labels},z->{labels}").get_output(0)


def gelu_tanh(network, inp, *, rounding_output_name: str | None = None):
    """Flax tanh-approximate GELU in the expert activation dtype."""
    dtype = inp.dtype
    x2 = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD).get_output(0)
    x3 = network.add_elementwise(x2, inp, trt.ElementWiseOperation.PROD).get_output(0)
    cubic_scale = scalar_like(network, inp, 0.044715, dtype=dtype)
    cubic = network.add_elementwise(x3, cubic_scale, trt.ElementWiseOperation.PROD).get_output(0)
    if rounding_output_name is not None:
        # TensorRT otherwise fuses the BF16 pointwise chain while retaining
        # extra precision across its declared BF16 intermediates. XLA rounds
        # this cubic term to BF16. A scalar IEinSum materializes the boundary
        # without adding an externally visible engine tensor.
        cubic = materialize_identity(network, cubic, name=rounding_output_name)
    inner = network.add_elementwise(inp, cubic, trt.ElementWiseOperation.SUM).get_output(0)
    root_scale = scalar_like(network, inp, math.sqrt(2.0 / math.pi), dtype=dtype)
    inner = network.add_elementwise(inner, root_scale, trt.ElementWiseOperation.PROD).get_output(0)
    # Pinned XLA widens only the tanh itself to FP32, then rounds the result
    # back to the BF16 expert dtype before the remaining GELU operations.
    tanh = network.add_activation(
        cast(network, inner, trt.float32), trt.ActivationType.TANH
    ).get_output(0)
    tanh = cast(network, tanh, dtype)
    one = scalar_like(network, inp, 1.0, dtype=dtype)
    factor = network.add_elementwise(tanh, one, trt.ElementWiseOperation.SUM).get_output(0)
    half = scalar_like(network, inp, 0.5, dtype=dtype)
    factor = network.add_elementwise(factor, half, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(inp, factor, trt.ElementWiseOperation.PROD).get_output(0)


def _last_axis_mask(tensor) -> int:
    rank = len(tuple(tensor.shape))
    if rank <= 0:
        raise ValueError("OpenPI normalization requires a non-scalar tensor")
    return 1 << (rank - 1)


def adaptive_rms_norm(
    network,
    inp,
    cond,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
    *,
    epsilon: float = 1e-6,
):
    """π0.5 adaptive RMSNorm; returns ``(normalized, residual_gate)``."""
    width = int(np.asarray(dense_bias).size // 3)
    output_dtype = inp.dtype
    # Upstream constructs this Dense with ``dtype=x.dtype``. The timestep MLP
    # condition is FP32, but both it and the parameters are cast to the expert
    # activation type for this projection.
    cond_for_dense = cast(network, cond, output_dtype)
    if output_dtype == trt.bfloat16:
        modulation = _xla_bf16_m1_linear(
            network,
            cond_for_dense,
            np.asarray(dense_weight),
            np.asarray(dense_bias),
        )
    else:
        modulation = linear(
            network,
            cond_for_dense,
            np.asarray(dense_weight),
            np.asarray(dense_bias),
            dtype=output_dtype,
        )
    modulation = network.add_shuffle(modulation)
    modulation.reshape_dims = (1, 1, 3 * width)
    modulation = modulation.get_output(0)
    scale = network.add_slice(modulation, (0, 0, 0), (1, 1, width), (1, 1, 1)).get_output(0)
    shift = network.add_slice(modulation, (0, 0, width), (1, 1, width), (1, 1, 1)).get_output(0)
    gate = network.add_slice(modulation, (0, 0, 2 * width), (1, 1, width), (1, 1, 1)).get_output(0)

    x = cast(network, inp, trt.float32)
    axes = _last_axis_mask(x)
    square = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, axes, True).get_output(0)
    eps = scalar_like(network, variance, epsilon, dtype=trt.float32)
    denom = network.add_elementwise(variance, eps, trt.ElementWiseOperation.SUM).get_output(0)
    denom = network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
    denom = network.add_unary(denom, trt.UnaryOperation.RECIP).get_output(0)
    normed = network.add_elementwise(x, denom, trt.ElementWiseOperation.PROD).get_output(0)
    one = scalar_like(network, scale, 1.0, dtype=output_dtype)
    scale = network.add_elementwise(scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    scale = cast(network, scale, trt.float32)
    shift = cast(network, shift, trt.float32)
    normed = network.add_elementwise(normed, scale, trt.ElementWiseOperation.PROD).get_output(0)
    normed = network.add_elementwise(normed, shift, trt.ElementWiseOperation.SUM).get_output(0)
    return cast(network, normed, output_dtype), cast(network, gate, output_dtype)


def _create_openpi_post_attention_rms_norm_plugin(*, epsilon: float):
    """Create the family-owned plugin for the audited fusion.98 seam."""
    creator = require_openpi_plugin_creator("OpenPIPostAttentionRmsNorm", trt=trt)

    epsilon_value = np.asarray([epsilon], dtype=np.float32)
    plugin = creator.create_plugin(
        "openpi_post_attention_rms_norm",
        trt.PluginFieldCollection(
            [trt.PluginField("epsilon", epsilon_value, trt.PluginFieldType.FLOAT32)]
        ),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create OpenPI post-attention RMSNorm plugin")
    return plugin


def _create_openpi_final_adaptive_rms_norm_plugin(*, epsilon: float):
    """Create the fixed-shape pi05-DROID fusion.144 plugin."""
    if np.float32(epsilon) != np.float32(1.0e-6):
        raise ValueError("OpenPI final adaptive RMSNorm requires the audited epsilon 1e-6")
    creator = require_openpi_plugin_creator("OpenPIFinalAdaptiveRmsNorm", trt=trt)
    epsilon_value = np.asarray([epsilon], dtype=np.float32)
    plugin = creator.create_plugin(
        "openpi_final_adaptive_rms_norm",
        trt.PluginFieldCollection(
            [trt.PluginField("epsilon", epsilon_value, trt.PluginFieldType.FLOAT32)]
        ),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create OpenPI final adaptive RMSNorm plugin")
    return plugin


def _create_openpi_action_layer0_mlp_closure_plugin():
    """Create the fixed-shape pi05-DROID layer-0 MLP closure plugin."""
    creator = require_openpi_plugin_creator("OpenPIActionLayer0MlpClosure", trt=trt)
    plugin = creator.create_plugin(
        "openpi_action_layer0_mlp_closure",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create OpenPI action layer-0 MLP closure plugin")
    return plugin


def _create_openpi_action_output_projection_plugin():
    """Create the fixed-shape pi05-DROID action-output plugin."""
    creator = require_openpi_plugin_creator("OpenPIActionOutputProjection", trt=trt)
    plugin = creator.create_plugin(
        "openpi_action_output_projection",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create OpenPI action-output projection plugin")
    return plugin


def _create_openpi_action_attention_context_plugin():
    """Create the fixed-shape pi05-DROID action-attention plugin."""
    creator = require_openpi_plugin_creator("OpenPIActionAttentionContext", trt=trt)
    plugin = creator.create_plugin(
        "openpi_action_attention_context",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create OpenPI action-attention context plugin")
    return plugin


def _create_openpi_rope_qk_plugin():
    """Create the family-owned libdevice-compatible Q/K RoPE plugin."""
    creator = require_openpi_plugin_creator("OpenPIRopeQK", trt=trt)

    plugin = creator.create_plugin(
        "openpi_action_rope_qk",
        trt.PluginFieldCollection([]),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("failed to create the OpenPI action RoPE TensorRT plugin")
    return plugin


def apply_action_rope_qk(
    network,
    query,
    key,
    position_ids,
    *,
    max_positions: int,
    head_dim: int,
):
    """Apply exact XLA/libdevice RoPE jointly to production action Q/K."""
    if (
        query.dtype == trt.bfloat16
        and key.dtype == trt.bfloat16
        and tuple(query.shape) == (1, 8, 15, 256)
        and tuple(key.shape) == (1, 1, 15, 256)
        and tuple(position_ids.shape) == (1, 15)
    ):
        plugin = _create_openpi_rope_qk_plugin()
        layer = network.add_plugin_v3([query, key, position_ids], [], plugin)
        return layer.get_output(0), layer.get_output(1)

    return (
        apply_rope(
            network,
            query,
            position_ids,
            max_positions=max_positions,
            head_dim=head_dim,
        ),
        apply_rope(
            network,
            key,
            position_ids,
            max_positions=max_positions,
            head_dim=head_dim,
        ),
    )


def pre_attention_adaptive_rms_norm(
    network,
    inp,
    cond,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
    *,
    epsilon: float = 1e-6,
    layer_name: str,
    droid_timestep=None,
    droid_layer: int | None = None,
):
    """Match the pinned pi05-DROID pre-attention adaptive RMSNorm.

    The pre-attention and post-attention XLA fusions use the same fixed
    two-warp width-1024 reduction. Reuse the audited post-attention plugin
    with an exact zero gated contribution so the residual path is an identity;
    this preserves the XLA association without adding another runtime plugin.
    The production builder calls this helper only through its explicit DROID
    selector, and every tensor/parameter dimension is checked again here.
    """
    weight = np.asarray(dense_weight)
    bias = np.asarray(dense_bias)
    expected_shapes = {
        "input": (1, 15, 1024),
        "condition": (1, 1024),
        "weight": (1024, 3072),
        "bias": (3072,),
    }
    observed_shapes = {
        "input": tuple(inp.shape),
        "condition": tuple(cond.shape),
        "weight": tuple(weight.shape),
        "bias": tuple(bias.shape),
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            "OpenPI pre-attention adaptive RMSNorm requires fixed shapes "
            f"{expected_shapes}, got {observed_shapes}"
        )
    if inp.dtype != trt.bfloat16 or cond.dtype != trt.float32:
        raise ValueError(
            "OpenPI pre-attention adaptive RMSNorm requires BF16 input and "
            f"FP32 condition, got {inp.dtype} and {cond.dtype}"
        )
    if np.float32(epsilon) != np.float32(1.0e-6):
        raise ValueError("OpenPI pre-attention adaptive RMSNorm requires epsilon 1e-6")
    if not layer_name:
        raise ValueError("OpenPI pre-attention adaptive RMSNorm requires a non-empty layer name")

    cond_for_dense = cast(network, cond, trt.bfloat16)
    modulation = _xla_bf16_m1_linear(network, cond_for_dense, weight, bias)
    if (droid_timestep is None) != (droid_layer is None):
        raise ValueError(
            "OpenPI pre-attention modulation corrections require timestep and layer together"
        )
    if droid_timestep is not None:
        modulation = correct_droid_adaptive_modulation_bf16_boundaries(
            network,
            modulation,
            droid_timestep,
            layer=droid_layer,
            role="attn",
        )
    modulation_shuffle = network.add_shuffle(modulation)
    modulation_shuffle.reshape_dims = (1, 1, 3072)
    modulation = modulation_shuffle.get_output(0)
    scale = network.add_slice(modulation, (0, 0, 0), (1, 1, 1024), (1, 1, 1)).get_output(0)
    shift = network.add_slice(modulation, (0, 0, 1024), (1, 1, 1024), (1, 1, 1)).get_output(0)
    gate = network.add_slice(modulation, (0, 0, 2048), (1, 1, 1024), (1, 1, 1)).get_output(0)

    vector_inputs = []
    for value in (scale, shift):
        shuffle = network.add_shuffle(value)
        shuffle.reshape_dims = (1024,)
        vector_inputs.append(shuffle.get_output(0))
    # Reuse the finite activation as the update and multiply it by +0. This
    # produces a signed zero matching each input element, so the residual add
    # remains an exact identity even if a future qualified input contains -0.
    zero_gate = constant(
        network,
        np.zeros((1024,), dtype=np.float32),
        dtype=trt.bfloat16,
    )
    plugin = _create_openpi_post_attention_rms_norm_plugin(epsilon=epsilon)
    layer = network.add_plugin_v3([inp, inp, zero_gate, *vector_inputs], [], plugin)
    layer.name = layer_name
    return layer.get_output(0), gate


def post_attention_adaptive_rms_norm(
    network,
    residual,
    update,
    residual_gate,
    cond,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
    *,
    epsilon: float = 1e-6,
    droid_timestep=None,
    droid_layer: int | None = None,
):
    """Match XLA fusion.98's gated residual and adaptive RMSNorm seam.

    The pinned BF16 graph rounds the gated attention update, rounds its
    residual addition, then reduces the resulting width-1024 row using a
    fixed two-warp topology before adaptive scale/shift. TensorRT's native
    pointwise/reduction fusion does not preserve that association, so the
    qualified BF16 path uses the family-owned plugin. Other precisions retain
    the ordinary graph implementation.
    """
    width = int(np.asarray(dense_bias).size // 3)
    production_shape = (
        width == 1024
        and tuple(residual.shape) == (1, 15, 1024)
        and tuple(update.shape) == (1, 15, 1024)
        and tuple(residual_gate.shape) == (1, 1, 1024)
        and tuple(cond.shape) == (1, 1024)
    )
    correction_requested = droid_timestep is not None or droid_layer is not None
    if (droid_timestep is None) != (droid_layer is None):
        raise ValueError(
            "OpenPI post-attention modulation corrections require timestep and layer together"
        )
    if correction_requested and (residual.dtype != trt.bfloat16 or not production_shape):
        raise ValueError(
            "OpenPI post-attention modulation corrections require the fixed BF16 DROID shape"
        )
    if residual.dtype != trt.bfloat16 or not production_shape:
        hidden = gated_residual(network, residual, update, residual_gate)
        return adaptive_rms_norm(
            network,
            hidden,
            cond,
            dense_weight,
            dense_bias,
            epsilon=epsilon,
        )

    cond_for_dense = cast(network, cond, residual.dtype)
    modulation = _xla_bf16_m1_linear(
        network,
        cond_for_dense,
        np.asarray(dense_weight),
        np.asarray(dense_bias),
    )
    if droid_timestep is not None:
        modulation = correct_droid_adaptive_modulation_bf16_boundaries(
            network,
            modulation,
            droid_timestep,
            layer=droid_layer,
            role="ffw",
        )
    modulation_shuffle = network.add_shuffle(modulation)
    modulation_shuffle.reshape_dims = (1, 1, 3 * width)
    modulation = modulation_shuffle.get_output(0)
    scale = network.add_slice(modulation, (0, 0, 0), (1, 1, width), (1, 1, 1)).get_output(0)
    shift = network.add_slice(modulation, (0, 0, width), (1, 1, width), (1, 1, 1)).get_output(0)
    gate = network.add_slice(modulation, (0, 0, 2 * width), (1, 1, width), (1, 1, 1)).get_output(0)

    vector_inputs = []
    for value in (residual_gate, scale, shift):
        shuffle = network.add_shuffle(value)
        shuffle.reshape_dims = (width,)
        vector_inputs.append(shuffle.get_output(0))
    plugin = _create_openpi_post_attention_rms_norm_plugin(epsilon=epsilon)
    normed = network.add_plugin_v3([residual, update, *vector_inputs], [], plugin).get_output(0)
    return normed, gate


def final_adaptive_rms_norm(
    network,
    inp,
    cond,
    dense_weight: np.ndarray,
    dense_bias: np.ndarray,
    *,
    epsilon: float = 1e-6,
):
    """Apply the exact pinned pi05-DROID final norm when its contract matches.

    The XLA final norm is a distinct eight-warp fusion that combines the
    condition projection with all 15 RMS reductions. Tiny test profiles,
    non-BF16 graphs, and any unaudited shape retain the ordinary TensorRT graph
    implementation.
    """
    weight = np.asarray(dense_weight)
    bias = np.asarray(dense_bias)
    production_shape = (
        tuple(inp.shape) == (1, 15, 1024)
        and tuple(cond.shape) == (1, 1024)
        and weight.shape == (1024, 3072)
        and bias.shape == (3072,)
    )
    exact_epsilon = np.float32(epsilon) == np.float32(1.0e-6)
    if (
        inp.dtype != trt.bfloat16
        or cond.dtype != trt.float32
        or not production_shape
        or not exact_epsilon
    ):
        normalized, _ = adaptive_rms_norm(
            network,
            inp,
            cond,
            weight,
            bias,
            epsilon=epsilon,
        )
        return normalized

    bias_tensor = constant(network, bias, dtype=trt.bfloat16)
    weight_tensor = constant(network, weight, dtype=trt.bfloat16)
    plugin = _create_openpi_final_adaptive_rms_norm_plugin(epsilon=epsilon)
    # Preserve fusion.144's audited argument order: hidden, bias, weight,
    # condition. The plugin owns no parameter storage of its own.
    layer = network.add_plugin_v3([inp, bias_tensor, weight_tensor, cond], [], plugin)
    layer.name = "openpi_final_adaptive_rms_norm"
    return layer.get_output(0)


def action_layer0_mlp_closure(
    network,
    post_attention,
    normed_ffw,
    ffw_gate,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    layer_name: str,
):
    """Apply the audited fixed-shape action MLP closure.

    This helper is intentionally fail-closed. The production builder reaches
    it only for the qualified pi05-DROID layer contract; tiny profiles and
    non-BF16 graphs retain the ordinary TensorRT graph. The plugin ABI name
    retains ``Layer0`` because that was the first isolated seam qualified.
    """
    tensor_shapes = {
        "post_attention": tuple(post_attention.shape),
        "normed_ffw": tuple(normed_ffw.shape),
        "ffw_gate": tuple(ffw_gate.shape),
    }
    expected_tensor_shapes = {
        "post_attention": (1, 15, 1024),
        "normed_ffw": (1, 15, 1024),
        "ffw_gate": (1, 1, 1024),
    }
    if tensor_shapes != expected_tensor_shapes:
        raise ValueError(
            "OpenPI action layer-0 MLP closure requires fixed tensor shapes "
            f"{expected_tensor_shapes}, got {tensor_shapes}"
        )
    for name, tensor in (
        ("post_attention", post_attention),
        ("normed_ffw", normed_ffw),
        ("ffw_gate", ffw_gate),
    ):
        if tensor.dtype != trt.bfloat16:
            raise ValueError(
                f"OpenPI action layer-0 MLP closure requires BF16 {name}, got {tensor.dtype}"
            )

    weights = {
        "gate_weight": np.asarray(gate_weight),
        "up_weight": np.asarray(up_weight),
        "down_weight": np.asarray(down_weight),
    }
    expected_weight_shapes = {
        "gate_weight": (1024, 4096),
        "up_weight": (1024, 4096),
        "down_weight": (4096, 1024),
    }
    observed_weight_shapes = {name: tuple(value.shape) for name, value in weights.items()}
    if observed_weight_shapes != expected_weight_shapes:
        raise ValueError(
            "OpenPI action layer-0 MLP closure requires fixed weight shapes "
            f"{expected_weight_shapes}, got {observed_weight_shapes}"
        )
    if not layer_name:
        raise ValueError("OpenPI action MLP closure requires a non-empty layer name")

    weight_tensors = [
        constant(network, weights[name], dtype=trt.bfloat16)
        for name in ("gate_weight", "up_weight", "down_weight")
    ]
    plugin = _create_openpi_action_layer0_mlp_closure_plugin()
    layer = network.add_plugin_v3(
        [post_attention, normed_ffw, ffw_gate, *weight_tensors], [], plugin
    )
    layer.name = layer_name
    return layer.get_output(0)


def action_output_projection(
    network,
    hidden,
    weight: np.ndarray,
    bias: np.ndarray,
):
    """Apply the audited padded pi05-DROID action-output projection."""
    weight = np.asarray(weight)
    bias = np.asarray(bias)
    expected_shapes = {
        "hidden": (1, 15, 1024),
        "weight": (1024, 32),
        "bias": (32,),
    }
    observed_shapes = {
        "hidden": tuple(hidden.shape),
        "weight": tuple(weight.shape),
        "bias": tuple(bias.shape),
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            "OpenPI action-output projection requires fixed shapes "
            f"{expected_shapes}, got {observed_shapes}"
        )
    if hidden.dtype != trt.bfloat16:
        raise ValueError(
            f"OpenPI action-output projection requires BF16 hidden input, got {hidden.dtype}"
        )

    weight_tensor = constant(network, weight, dtype=trt.bfloat16)
    bias_tensor = constant(network, bias, dtype=trt.bfloat16)
    plugin = _create_openpi_action_output_projection_plugin()
    layer = network.add_plugin_v3([hidden, weight_tensor, bias_tensor], [], plugin)
    layer.name = "openpi_action_output_projection"
    output = layer.get_output(0)
    if tuple(output.shape) != (1, 15, 32) or output.dtype != trt.bfloat16:
        raise RuntimeError(
            "OpenPI action-output plugin returned an invalid contract: "
            f"shape={tuple(output.shape)}, dtype={output.dtype}"
        )
    return output


def action_attention_context(
    network,
    query,
    key,
    value,
    attention_mask,
    *,
    layer_name: str,
):
    """Apply the audited fixed-shape pi05-DROID attention contraction.

    The plugin reproduces the pinned XLA QK GEMM, masked FP32 softmax, and PV
    GEMM, including the physical 983-to-984 padding. It is intentionally
    fail-closed and is reachable from the action builder only for the
    recursively qualified pi05-DROID BF16 geometry.
    """
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "attention_mask": attention_mask,
    }
    observed_shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
    expected_shapes = {
        "query": (1, 8, 15, 256),
        "key": (1, 1, 983, 256),
        "value": (1, 1, 983, 256),
        "attention_mask": (1, 1, 15, 983),
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            "OpenPI action-attention context requires fixed tensor shapes "
            f"{expected_shapes}, got {observed_shapes}"
        )
    for name in ("query", "key", "value"):
        if tensors[name].dtype != trt.bfloat16:
            raise ValueError(
                f"OpenPI action-attention context requires BF16 {name}, got {tensors[name].dtype}"
            )
    if attention_mask.dtype != trt.bool:
        raise ValueError(
            "OpenPI action-attention context requires a boolean attention mask, "
            f"got {attention_mask.dtype}"
        )
    if not layer_name:
        raise ValueError("OpenPI action-attention context requires a non-empty layer name")

    plugin = _create_openpi_action_attention_context_plugin()
    layer = network.add_plugin_v3([query, key, value, attention_mask], [], plugin)
    layer.name = layer_name
    output = layer.get_output(0)
    if tuple(output.shape) != (1, 15, 2048):
        raise RuntimeError(
            "OpenPI action-attention context plugin returned unexpected shape "
            f"{tuple(output.shape)}"
        )
    return output


def gated_residual(
    network,
    residual,
    update,
    gate=None,
    *,
    rounding_output_name: str | None = None,
):
    if gate is not None:
        update = network.add_elementwise(update, gate, trt.ElementWiseOperation.PROD).get_output(0)
        if rounding_output_name is not None:
            update = materialize_identity(network, update, name=rounding_output_name)
    return network.add_elementwise(residual, update, trt.ElementWiseOperation.SUM).get_output(0)


def rope_tables(max_positions: int, head_dim: int, theta: float = 10_000.0):
    """Return half-width FP32 cos/sin tables matching OpenPI's JAX RoPE."""
    if head_dim % 2:
        raise ValueError("OpenPI RoPE head_dim must be even")
    exponents = (2.0 / head_dim) * np.arange(head_dim // 2, dtype=np.float32)
    timescale = np.float32(theta) ** exponents
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    radians = positions / timescale[None, :]
    return np.cos(radians).astype(np.float32), np.sin(radians).astype(np.float32)


def apply_rope(network, tensor, position_ids, *, max_positions: int, head_dim: int):
    """Apply OpenPI's split-half RoPE to ``[B,H,S,D]`` in FP32."""
    output_dtype = tensor.dtype
    cos_values, sin_values = rope_tables(max_positions, head_dim)
    cos_table = constant(network, cos_values, dtype=trt.float32)
    sin_table = constant(network, sin_values, dtype=trt.float32)
    positions = position_ids
    if positions.dtype != trt.int32:
        positions = cast(network, positions, trt.int32)
    cos = network.add_gather(cos_table, positions, 0).get_output(0)
    sin = network.add_gather(sin_table, positions, 0).get_output(0)
    cos_shuffle = network.add_shuffle(cos)
    cos_shuffle.reshape_dims = (1, 1, -1, head_dim // 2)
    sin_shuffle = network.add_shuffle(sin)
    sin_shuffle.reshape_dims = (1, 1, -1, head_dim // 2)
    cos = cos_shuffle.get_output(0)
    sin = sin_shuffle.get_output(0)

    x = cast(network, tensor, trt.float32)
    shape = tuple(x.shape)
    first = network.add_slice(
        x, (0, 0, 0, 0), (shape[0], shape[1], shape[2], head_dim // 2), (1, 1, 1, 1)
    ).get_output(0)
    second = network.add_slice(
        x, (0, 0, 0, head_dim // 2), (shape[0], shape[1], shape[2], head_dim // 2), (1, 1, 1, 1)
    ).get_output(0)
    first_cos = network.add_elementwise(first, cos, trt.ElementWiseOperation.PROD).get_output(0)
    second_sin = network.add_elementwise(second, sin, trt.ElementWiseOperation.PROD).get_output(0)
    out_first = network.add_elementwise(
        first_cos, second_sin, trt.ElementWiseOperation.SUB
    ).get_output(0)
    second_cos = network.add_elementwise(second, cos, trt.ElementWiseOperation.PROD).get_output(0)
    first_sin = network.add_elementwise(first, sin, trt.ElementWiseOperation.PROD).get_output(0)
    out_second = network.add_elementwise(
        second_cos, first_sin, trt.ElementWiseOperation.SUM
    ).get_output(0)
    concat = network.add_concatenation([out_first, out_second])
    concat.axis = 3
    return cast(network, concat.get_output(0), output_dtype)


def attention_from_rotated(
    network,
    q,
    k,
    v,
    mask,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    fp32_logits: bool = True,
):
    """Attention for already-RoPE'd ``[B,H,S,D]`` Q/K tensors.

    The audited OpenPI implementation forms logits and performs softmax in
    FP32, then casts probabilities back to the activation dtype before the
    value product.  Keep that path as the default.  The native TensorRT
    IAttention path remains available for later tactic qualification, and the
    primitive implementation also lets the family build with TensorRT 10.x
    releases that predate IAttention.
    """
    q_dtype = q.dtype
    scale = scalar_like(network, q, 1.0 / math.sqrt(head_dim), dtype=q_dtype)
    q_scaled = network.add_elementwise(q, scale, trt.ElementWiseOperation.PROD).get_output(0)

    if fp32_logits:
        if num_query_heads % num_kv_heads:
            raise ValueError("OpenPI query heads must be divisible by K/V heads")
        batch, _, query_length, _ = tuple(q.shape)
        key_length = int(tuple(k.shape)[2])
        groups = num_query_heads // num_kv_heads

        q_tokens = network.add_shuffle(q_scaled)
        q_tokens.first_transpose = (0, 2, 1, 3)
        q_tokens.reshape_dims = (
            batch,
            query_length,
            num_kv_heads,
            groups,
            head_dim,
        )
        k_tokens = network.add_shuffle(k)
        k_tokens.first_transpose = (0, 2, 1, 3)
        v_tokens = network.add_shuffle(v)
        v_tokens.first_transpose = (0, 2, 1, 3)
        q32 = cast(network, q_tokens.get_output(0), trt.float32)
        k32 = cast(network, k_tokens.get_output(0), trt.float32)
        logits = network.add_einsum([q32, k32], "btkgh,bskh->bkgts").get_output(0)
        if mask is not None:
            if mask.dtype != trt.bool:
                raise ValueError("OpenPI attention mask must be boolean")
            mask_tokens = network.add_shuffle(mask)
            mask_tokens.reshape_dims = (
                batch,
                1,
                1,
                query_length,
                key_length,
            )
            big_negative = scalar_like(network, logits, -2.3819763e38, dtype=trt.float32)
            logits = network.add_select(mask_tokens.get_output(0), logits, big_negative).get_output(
                0
            )
        softmax = network.add_softmax(logits)
        softmax.axes = 1 << (len(tuple(logits.shape)) - 1)
        probabilities = cast(network, softmax.get_output(0), q_dtype)
        # XLA pads the BF16 probability/value contraction to a multiple of
        # eight along S before its cuBLAS call. Matching that shape makes TRT's
        # IEinSum contraction bit-exact for the pinned S=983 action step.
        padded_key_length = ((key_length + 7) // 8) * 8
        pad = padded_key_length - key_length
        values = v_tokens.get_output(0)
        if pad:
            probability_padding = constant(
                network,
                np.zeros(
                    (batch, num_kv_heads, groups, query_length, pad),
                    dtype=np.float32,
                ),
                dtype=q_dtype,
            )
            probability_concat = network.add_concatenation([probabilities, probability_padding])
            probability_concat.axis = 4
            probabilities = probability_concat.get_output(0)
            value_padding = constant(
                network,
                np.zeros((batch, pad, num_kv_heads, head_dim), dtype=np.float32),
                dtype=q_dtype,
            )
            value_concat = network.add_concatenation([values, value_padding])
            value_concat.axis = 1
            values = value_concat.get_output(0)
        context = network.add_einsum([probabilities, values], "bkgts,bskh->btkgh").get_output(0)
        context_rows = network.add_shuffle(context)
        context_rows.reshape_dims = (
            batch,
            query_length,
            num_query_heads * head_dim,
        )
        return context_rows.get_output(0)

    if hasattr(network, "add_attention"):
        # IAttention accepts an additive mask. Convert the public boolean mask
        # explicitly so padding semantics do not depend on TensorRT version.
        additive_mask = None
        if mask is not None:
            if mask.dtype != trt.bool:
                raise ValueError("OpenPI attention mask must be boolean")
            zero = scalar_like(network, q, 0.0, dtype=q_dtype)
            big_negative = scalar_like(network, q, -2.3819763e38, dtype=q_dtype)
            additive_mask = network.add_select(mask, zero, big_negative).get_output(0)
        attn = network.add_attention(q_scaled, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
        attn.decomposable = True
        if additive_mask is not None:
            attn.mask = additive_mask
        context = attn.get_output(0)
    else:
        logits = network.add_matrix_multiply(
            q_scaled,
            trt.MatrixOperation.NONE,
            k,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        if mask is not None:
            if mask.dtype != trt.bool:
                raise ValueError("OpenPI attention mask must be boolean")
            big_negative = scalar_like(network, logits, -2.3819763e38, dtype=q_dtype)
            logits = network.add_select(mask, logits, big_negative).get_output(0)
        softmax = network.add_softmax(logits)
        softmax.axes = 1 << (len(tuple(logits.shape)) - 1)
        context = network.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            v,
            trt.MatrixOperation.NONE,
        ).get_output(0)
    shape = tuple(context.shape)
    shuffle = network.add_shuffle(context)
    shuffle.first_transpose = (0, 2, 1, 3)
    shuffle.reshape_dims = (shape[0], shape[2], num_query_heads * head_dim)
    return shuffle.get_output(0)


def add_sinusoidal_embedding(
    network,
    timestep,
    dimension: int,
    *,
    min_period: float = 4e-3,
    max_period: float = 4.0,
):
    """Build the π0/π0.5 scalar timestep embedding in FP32.

    Keep the association used by the pinned optimized XLA graph: multiply the
    scalar timestep by ``2*pi`` first, then multiply by the reciprocal period.
    Precomputing one combined scale changes enough FP32 angles to perturb the
    conditioner permanently.
    """
    inverse_period_values = sinusoidal_inverse_periods_numpy(
        dimension,
        min_period=min_period,
        max_period=max_period,
    )
    inverse_period = constant(network, inverse_period_values, dtype=trt.float32)
    time32 = cast(network, timestep, trt.float32)
    scaled_time = network.add_elementwise(
        time32,
        constant(network, np.float32(2.0 * math.pi), dtype=trt.float32),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    angles = network.add_elementwise(
        inverse_period, scaled_time, trt.ElementWiseOperation.PROD
    ).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    concat = network.add_concatenation([sin, cos])
    concat.axis = 0
    output = network.add_shuffle(concat.get_output(0))
    output.reshape_dims = (1, dimension)
    return output.get_output(0)
