# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TensorRT graph vocabulary owned by Wan2.2 TI2V."""

from __future__ import annotations

import math

import numpy as np
import ml_dtypes

from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt()

_EMULATE_BF16_GEMM = False
_USE_SOURCE_ATTENTION_PLUGIN = False
_USE_SOURCE_LINEAR_PLUGIN = False
_USE_CUDA_BF16_BARRIERS = False
_USE_DIT_CUDA_NUMERICS = False
_USE_DIT_BF16_LINEAR = False
_USE_DIT_TIME_SILU = False
_USE_DIT_TIME_LINEAR2 = False
_USE_DIT_TIME_PROJECTION = False
_USE_DIT_BLOCK_LAYER_NORM = False
_USE_DIT_ADAPTIVE_NORM = False
_USE_DIT_RMS_NORM = False
_USE_DIT_SELF_GATED_RESIDUAL = False
_USE_DIT_FFN_GATED_RESIDUAL = False
_USE_DIT_CROSS_AFFINE_LAYER_NORM = False
_USE_DIT_FINAL_PROJECTION = False


def set_bf16_gemm_emulation(enabled: bool) -> None:
    global _EMULATE_BF16_GEMM
    _EMULATE_BF16_GEMM = bool(enabled)


def set_source_attention_plugin(enabled: bool) -> None:
    global _USE_SOURCE_ATTENTION_PLUGIN, _USE_SOURCE_LINEAR_PLUGIN
    _USE_SOURCE_ATTENTION_PLUGIN = bool(enabled)
    _USE_SOURCE_LINEAR_PLUGIN = bool(enabled)


def set_cuda_bf16_barriers(enabled: bool) -> None:
    """Force source-eager BF16 materialization without ATen or PyTorch."""

    global _USE_CUDA_BF16_BARRIERS
    _USE_CUDA_BF16_BARRIERS = bool(enabled)


def set_dit_cuda_numerics(enabled: bool) -> None:
    """Use CUDA plugins for DiT operations whose source semantics exceed TRT layers."""

    global _USE_DIT_CUDA_NUMERICS
    _USE_DIT_CUDA_NUMERICS = bool(enabled)


def set_dit_bf16_linear(enabled: bool) -> None:
    """Enable the qualified production BF16 linear plugin independently for A/B."""

    global _USE_DIT_BF16_LINEAR
    _USE_DIT_BF16_LINEAR = bool(enabled)


def set_dit_time_silu(enabled: bool) -> None:
    """Enable the source-exact FP32 time SiLU independently for A/B."""

    global _USE_DIT_TIME_SILU
    _USE_DIT_TIME_SILU = bool(enabled)


def set_dit_time_linear2(enabled: bool) -> None:
    """Enable the exact second FP32 time linear independently for A/B."""

    global _USE_DIT_TIME_LINEAR2
    _USE_DIT_TIME_LINEAR2 = bool(enabled)


def set_dit_time_projection(enabled: bool) -> None:
    """Enable the exact FP32 time projection independently for A/B."""

    global _USE_DIT_TIME_PROJECTION
    _USE_DIT_TIME_PROJECTION = bool(enabled)


def set_dit_block_layer_norm(enabled: bool) -> None:
    """Enable the source-exact fixed-profile non-affine block LayerNorm for A/B."""

    global _USE_DIT_BLOCK_LAYER_NORM
    _USE_DIT_BLOCK_LAYER_NORM = bool(enabled)


def set_dit_adaptive_norm(enabled: bool) -> None:
    """Enable the source-exact non-FMA adaptive norm independently for A/B."""

    global _USE_DIT_ADAPTIVE_NORM
    _USE_DIT_ADAPTIVE_NORM = bool(enabled)


def set_dit_rms_norm(enabled: bool) -> None:
    """Enable the source-exact fixed-profile Q/K RMSNorm independently for A/B."""

    global _USE_DIT_RMS_NORM
    _USE_DIT_RMS_NORM = bool(enabled)


def set_dit_self_gated_residual(enabled: bool) -> None:
    """Enable the source-exact self-attention gated residual independently for A/B."""

    global _USE_DIT_SELF_GATED_RESIDUAL
    _USE_DIT_SELF_GATED_RESIDUAL = bool(enabled)


def set_dit_ffn_gated_residual(enabled: bool) -> None:
    """Enable the source-exact FFN gated residual independently for A/B."""

    global _USE_DIT_FFN_GATED_RESIDUAL
    _USE_DIT_FFN_GATED_RESIDUAL = bool(enabled)


def set_dit_cross_affine_layer_norm(enabled: bool) -> None:
    """Enable source-exact cross-attention affine LayerNorm independently for A/B."""

    global _USE_DIT_CROSS_AFFINE_LAYER_NORM
    _USE_DIT_CROSS_AFFINE_LAYER_NORM = bool(enabled)


def set_dit_final_projection(enabled: bool) -> None:
    """Enable the source-exact fixed-profile FP32 output projection for A/B."""

    global _USE_DIT_FINAL_PROJECTION
    _USE_DIT_FINAL_PROJECTION = bool(enabled)


def bf16_barrier(network, tensor, label: str = "value"):
    if not _USE_CUDA_BF16_BARRIERS:
        return cast(network, tensor, trt.bfloat16)
    tensor = cast(network, tensor, trt.bfloat16)
    creator = trt.get_plugin_registry().get_creator("Wan22Umt5Bf16Barrier", "1", "")
    if creator is None:
        raise RuntimeError("Wan2.2 BF16 barrier plugin creator is not registered")
    instance = f"wan22_dit_{label}_{network.num_layers}"
    plugin = creator.create_plugin(instance, trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([tensor], plugin)
    if layer is None:
        raise RuntimeError("Could not add the Wan2.2 BF16 barrier plugin")
    return layer.get_output(0)


def _bf16_rounded_fp32(value):
    return np.asarray(value, dtype=ml_dtypes.bfloat16).astype(np.float32)


def constant(network, value, *, dtype=np.float32):
    array = np.ascontiguousarray(value, dtype=dtype)
    return network.add_constant(tuple(array.shape), array).get_output(0)


def source_patch_embedding(
    network, latent, weight, bias, patch_size, num_patches: int, hidden_size: int
):
    if not _USE_DIT_CUDA_NUMERICS:
        return None
    # The cuDNN graph is qualified only for the fixed production profile.  Keep
    # small component probes on the existing unfold+MM fallback rather than
    # claiming source exactness for an unqualified convolution shape.
    production_shape = (1, 48, 31, 44, 80)
    latent_shape = tuple(int(value) for value in latent.shape)
    if (
        latent_shape != production_shape
        or tuple(patch_size) != (1, 2, 2)
        or num_patches != 27_280
        or hidden_size != 3_072
    ):
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitPatchEmbedding", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitPatchEmbedding plugin creator is not registered")
    # Match autocast before entering cuDNN.  The plugin itself accepts only
    # BF16 and performs FP32 convolution accumulation.
    latent = cast(network, latent, trt.bfloat16)
    weight_tensor = cast(
        network, constant(network, np.asarray(weight, dtype=np.float32)), trt.bfloat16
    )
    bias_tensor = cast(network, constant(network, np.asarray(bias, dtype=np.float32)), trt.bfloat16)
    plugin = creator.create_plugin("wan2_2_dit_patch_embedding", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2([latent, weight_tensor, bias_tensor], plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitPatchEmbedding plugin")
    rows = network.add_shuffle(layer.get_output(0))
    rows.first_transpose = trt.Permutation([0, 2, 3, 4, 1])
    rows.reshape_dims = (num_patches, hidden_size)
    return rows.get_output(0)


def source_time_linear1(network, x, weight, bias):
    """Use the qualified source-exact first time linear on the fixed TI2V profile."""

    if not _USE_DIT_CUDA_NUMERICS:
        return None
    if (
        tuple(int(value) for value in x.shape) != (27_280, 256)
        or tuple(np.asarray(weight).shape) != (3_072, 256)
        or tuple(np.asarray(bias).shape) != (3_072,)
    ):
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitTimeLinear1", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitTimeLinear1 plugin creator is not registered")
    inputs = [
        cast(network, x, trt.float32),
        constant(network, np.asarray(weight, dtype=np.float32)),
        constant(network, np.asarray(bias, dtype=np.float32)),
    ]
    plugin = creator.create_plugin("wan2_2_dit_time_linear1", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitTimeLinear1 plugin")
    return layer.get_output(0)


def source_time_linear2(network, x, weight, bias):
    """Use the call92-qualified exact second time linear on the fixed profile."""

    if not _USE_DIT_CUDA_NUMERICS or not _USE_DIT_TIME_LINEAR2:
        return None
    if (
        tuple(int(value) for value in x.shape) != (27_280, 3_072)
        or tuple(np.asarray(weight).shape) != (3_072, 3_072)
        or tuple(np.asarray(bias).shape) != (3_072,)
    ):
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitTimeLinear2", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitTimeLinear2 plugin creator is not registered")
    inputs = [
        cast(network, x, trt.float32),
        constant(network, np.asarray(weight, dtype=np.float32)),
        constant(network, np.asarray(bias, dtype=np.float32)),
    ]
    plugin = creator.create_plugin("wan2_2_dit_time_linear2", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitTimeLinear2 plugin")
    return layer.get_output(0)


def source_time_projection(network, x, weight, bias):
    """Use the call92-qualified exact time projection on the fixed profile."""

    if not _USE_DIT_CUDA_NUMERICS or not _USE_DIT_TIME_PROJECTION:
        return None
    if (
        tuple(int(value) for value in x.shape) != (27_280, 3_072)
        or tuple(np.asarray(weight).shape) != (18_432, 3_072)
        or tuple(np.asarray(bias).shape) != (18_432,)
    ):
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitTimeProjection", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitTimeProjection plugin creator is not registered")
    inputs = [
        cast(network, x, trt.float32),
        constant(network, np.asarray(weight, dtype=np.float32)),
        constant(network, np.asarray(bias, dtype=np.float32)),
    ]
    plugin = creator.create_plugin("wan2_2_dit_time_projection", trt.PluginFieldCollection([]))
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitTimeProjection plugin")
    return layer.get_output(0)


def source_final_projection(network, x, weight, bias):
    """Use the call92-qualified strict-FP32 output projection on the fixed profile."""

    if not _USE_DIT_CUDA_NUMERICS or not _USE_DIT_FINAL_PROJECTION:
        return None
    if (
        tuple(int(value) for value in x.shape) != (27_280, 3_072)
        or tuple(np.asarray(weight).shape) != (192, 3_072)
        or tuple(np.asarray(bias).shape) != (192,)
    ):
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitFinalProjectionFp32", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitFinalProjectionFp32 plugin creator is not registered")
    inputs = [
        cast(network, x, trt.float32),
        constant(network, np.asarray(weight, dtype=np.float32)),
        constant(network, np.asarray(bias, dtype=np.float32)),
    ]
    plugin = creator.create_plugin(
        "wan2_2_dit_final_projection_fp32", trt.PluginFieldCollection([])
    )
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError("Could not add Wan22DitFinalProjectionFp32 plugin")
    return layer.get_output(0)


def source_bf16_linear(network, x, weight, bias):
    """Use source-exact cuBLASLt for the five fixed DiT BF16 linear shapes."""

    if not _USE_DIT_CUDA_NUMERICS or not _USE_DIT_BF16_LINEAR or bias is None:
        return None
    x_shape = tuple(int(value) for value in x.shape)
    weight_values = np.asarray(weight, dtype=np.float32)
    bias_values = np.asarray(bias, dtype=np.float32)
    if len(x_shape) != 2 or weight_values.ndim != 2 or bias_values.ndim != 1:
        return None
    m, k = x_shape
    n, weight_k = weight_values.shape
    qualified_shapes = {
        (27_280, 3_072, 3_072),
        (27_280, 3_072, 14_336),
        (27_280, 14_336, 3_072),
        (512, 4_096, 3_072),
        (512, 3_072, 3_072),
    }
    if k != weight_k or bias_values.shape != (n,) or (m, k, n) not in qualified_shapes:
        return None
    creator = trt.get_plugin_registry().get_creator("Wan22DitBf16Linear", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitBf16Linear plugin creator is not registered")
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField("m", np.array([m], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("n", np.array([n], dtype=np.int32), trt.PluginFieldType.INT32),
            trt.PluginField("k", np.array([k], dtype=np.int32), trt.PluginFieldType.INT32),
        ]
    )
    plugin = creator.create_plugin(f"wan2_2_dit_bf16_linear_{network.num_layers}", fields)
    if plugin is None:
        raise RuntimeError(f"Could not create Wan22DitBf16Linear plugin for {(m, k, n)}")
    inputs = [
        cast(network, x, trt.bfloat16),
        cast(network, constant(network, weight_values), trt.bfloat16),
        cast(network, constant(network, bias_values), trt.bfloat16),
    ]
    layer = network.add_plugin_v2(inputs, plugin)
    if layer is None:
        raise RuntimeError(f"Could not add Wan22DitBf16Linear plugin for {(m, k, n)}")
    return layer.get_output(0)


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def bf16_constant(network, value):
    return cast(network, constant(network, value), trt.bfloat16)


def linear(network, x, weight, bias=None, *, bf16=True):
    """Linear with PyTorch [out,in] weights."""

    if bf16:
        source_output = source_bf16_linear(network, x, weight, bias)
        if source_output is not None:
            return source_output

    if not bf16 and _USE_SOURCE_LINEAR_PLUGIN and bias is not None:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceFloatLinear", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceFloatLinear plugin creator is not registered")
        x = cast(network, x, trt.float32)
        weight_tensor = constant(network, np.asarray(weight, dtype=np.float32))
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32))
        plugin = creator.create_plugin("wan2_2_source_float_linear", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([x, weight_tensor, bias_tensor], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceFloatLinear plugin")
        return layer.get_output(0)

    if bf16 and _USE_SOURCE_LINEAR_PLUGIN and bias is not None:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceLinear", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceLinear plugin creator is not registered")
        weight_tensor = constant(network, np.asarray(weight, dtype=np.float32))
        bias_tensor = constant(network, np.asarray(bias, dtype=np.float32))
        plugin = creator.create_plugin("wan2_2_source_linear", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([x, weight_tensor, bias_tensor], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceLinear plugin")
        return layer.get_output(0)

    emulate = bf16 and _EMULATE_BF16_GEMM
    rhs_values = _bf16_rounded_fp32(weight) if emulate else np.asarray(weight, dtype=np.float32)
    rhs = constant(network, rhs_values.T)
    if emulate:
        x = cast(network, cast(network, x, trt.bfloat16), trt.float32)
    elif bf16:
        x = cast(network, x, trt.bfloat16)
        rhs = cast(network, rhs, trt.bfloat16)
    layer = network.add_matrix_multiply(
        x,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    y = layer.get_output(0)
    if bias is not None:
        bias_values = _bf16_rounded_fp32(bias) if emulate else np.asarray(bias, dtype=np.float32)
        b = constant(network, bias_values.reshape(1, -1))
        b = cast(network, b, y.dtype)
        y = network.add_elementwise(y, b, trt.ElementWiseOperation.SUM).get_output(0)
    if emulate:
        y = cast(network, y, trt.bfloat16)
    if bf16:
        y = bf16_barrier(network, y, "linear")
    return y


def layer_norm(network, x, hidden_size: int, eps: float, *, round_bf16=False):
    x = cast(network, x, trt.float32)
    if (
        _USE_DIT_CUDA_NUMERICS
        and _USE_DIT_BLOCK_LAYER_NORM
        and tuple(int(value) for value in x.shape) == (27_280, 3_072)
        and hidden_size == 3_072
        and eps == 1.0e-6
    ):
        creator = trt.get_plugin_registry().get_creator("Wan22DitLayerNormFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitLayerNormFp32 plugin creator is not registered")
        plugin = creator.create_plugin(
            f"wan2_2_dit_block_layer_norm_{network.num_layers}",
            trt.PluginFieldCollection([]),
        )
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitLayerNormFp32 plugin")
        y = layer.get_output(0)
        if round_bf16:
            y = cast(network, bf16_barrier(network, y, "layer_norm"), trt.float32)
        return y
    gamma = constant(network, np.ones((1, hidden_size), dtype=np.float32))
    beta = constant(network, np.zeros((1, hidden_size), dtype=np.float32))
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceLayerNorm", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceLayerNorm plugin creator is not registered")
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32
                ),
                trt.PluginField("affine", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField(
                    "round_bf16",
                    np.array([int(round_bf16)], dtype=np.int32),
                    trt.PluginFieldType.INT32,
                ),
            ]
        )
        plugin = creator.create_plugin("wan2_2_source_layer_norm", fields)
        layer = network.add_plugin_v2([x, gamma, beta], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceLayerNorm plugin")
        return layer.get_output(0)
    norm = network.add_normalization_v2(x, gamma, beta, 1 << 1)
    norm.epsilon = eps
    y = norm.get_output(0)
    if round_bf16:
        y = cast(network, bf16_barrier(network, y, "layer_norm"), trt.float32)
    return y


def affine_layer_norm(network, x, weight, bias, hidden_size: int, eps: float):
    x = cast(network, x, trt.float32)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
    beta = constant(network, np.asarray(bias, dtype=np.float32).reshape(1, hidden_size))
    if (
        _USE_DIT_CUDA_NUMERICS
        and _USE_DIT_CROSS_AFFINE_LAYER_NORM
        and tuple(int(value) for value in x.shape) == (27_280, 3_072)
        and tuple(np.asarray(weight).shape) == (3_072,)
        and tuple(np.asarray(bias).shape) == (3_072,)
        and hidden_size == 3_072
        and eps == 1.0e-6
    ):
        creator = trt.get_plugin_registry().get_creator("Wan22DitLayerNormFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitLayerNormFp32 plugin creator is not registered")
        plugin = creator.create_plugin(
            f"wan2_2_dit_cross_affine_layer_norm_{network.num_layers}",
            trt.PluginFieldCollection([]),
        )
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitLayerNormFp32 plugin")
        normalized = layer.get_output(0)
        scaled = network.add_elementwise(
            normalized, gamma, trt.ElementWiseOperation.PROD
        ).get_output(0)
        return network.add_elementwise(scaled, beta, trt.ElementWiseOperation.SUM).get_output(0)
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceLayerNorm", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceLayerNorm plugin creator is not registered")
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32
                ),
                trt.PluginField("affine", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField(
                    "round_bf16", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32
                ),
            ]
        )
        plugin = creator.create_plugin("wan2_2_source_affine_layer_norm", fields)
        layer = network.add_plugin_v2([x, gamma, beta], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceLayerNorm plugin")
        return layer.get_output(0)
    norm = network.add_normalization_v2(x, gamma, beta, 1 << 1)
    norm.epsilon = eps
    return norm.get_output(0)


def rms_norm(
    network,
    x,
    weight,
    hidden_size: int,
    eps: float,
    *,
    debug_weight_name: str | None = None,
):
    """Match upstream WanRMSNorm's mixed BF16/FP32 boundary.

    The source casts the normalized values back to the linear output dtype
    (BF16), then multiplies by an FP32 parameter.  PyTorch type promotion makes
    that affine result FP32; Q/K are quantized to V's BF16 dtype only when they
    enter attention (after RoPE for self-attention).
    """

    def expose_gamma(gamma):
        if debug_weight_name is None:
            return
        creator = trt.get_plugin_registry().get_creator("Wan22DitFp32Barrier", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitFp32Barrier plugin creator is not registered")
        plugin = creator.create_plugin(
            f"wan2_2_dit_debug_rms_gamma_{network.num_layers}",
            trt.PluginFieldCollection([]),
        )
        layer = network.add_plugin_v2([gamma], plugin)
        if layer is None:
            raise RuntimeError("Could not expose RMSNorm gamma through FP32 barrier")
        debug = layer.get_output(0)
        debug.name = debug_weight_name
        network.mark_output(debug)

    if (
        _USE_DIT_CUDA_NUMERICS
        and _USE_DIT_RMS_NORM
        and tuple(int(value) for value in x.shape) in {(27_280, 3_072), (512, 3_072)}
        and tuple(np.asarray(weight).shape) == (3_072,)
        and hidden_size == 3_072
        and eps == 1.0e-6
    ):
        creator = trt.get_plugin_registry().get_creator("Wan22DitRmsNormFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitRmsNormFp32 plugin creator is not registered")
        gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
        expose_gamma(gamma)
        inputs = [cast(network, x, trt.bfloat16), gamma]
        plugin = creator.create_plugin(
            f"wan2_2_dit_rms_norm_{network.num_layers}", trt.PluginFieldCollection([])
        )
        layer = network.add_plugin_v2(inputs, plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitRmsNormFp32 plugin")
        return layer.get_output(0)
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceRmsNorm", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceRmsNorm plugin creator is not registered")
        x = cast(network, x, trt.bfloat16)
        gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
        expose_gamma(gamma)
        fields = trt.PluginFieldCollection(
            [trt.PluginField("eps", np.array([eps], dtype=np.float32), trt.PluginFieldType.FLOAT32)]
        )
        plugin = creator.create_plugin("wan2_2_source_rms_norm", fields)
        layer = network.add_plugin_v2([x, gamma], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceRmsNorm plugin")
        return layer.get_output(0)

    x_fp32 = cast(network, x, trt.float32)
    squared = network.add_elementwise(x_fp32, x_fp32, trt.ElementWiseOperation.PROD).get_output(0)
    mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
    eps_t = constant(network, np.array([[eps]], dtype=np.float32))
    variance = network.add_elementwise(mean, eps_t, trt.ElementWiseOperation.SUM).get_output(0)
    inv = network.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
    inv = network.add_unary(inv, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(x_fp32, inv, trt.ElementWiseOperation.PROD).get_output(0)
    normalized = bf16_barrier(network, normalized, "rms_norm")
    normalized = cast(network, normalized, trt.float32)
    gamma = constant(network, np.asarray(weight, dtype=np.float32).reshape(1, hidden_size))
    expose_gamma(gamma)
    return network.add_elementwise(normalized, gamma, trt.ElementWiseOperation.PROD).get_output(0)


def adaptive_norm(network, normalized, shift, scale):
    normalized = cast(network, normalized, trt.float32)
    if (
        _USE_DIT_CUDA_NUMERICS
        and _USE_DIT_ADAPTIVE_NORM
        and tuple(int(value) for value in normalized.shape) == (27_280, 3_072)
        and tuple(int(value) for value in shift.shape) == (1, 3_072)
        and tuple(int(value) for value in scale.shape) == (1, 3_072)
    ):
        creator = trt.get_plugin_registry().get_creator("Wan22DitAdaptiveNormFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitAdaptiveNormFp32 plugin creator is not registered")
        inputs = [
            normalized,
            cast(network, shift, trt.float32),
            cast(network, scale, trt.float32),
        ]
        plugin = creator.create_plugin(
            f"wan2_2_dit_adaptive_norm_{network.num_layers}",
            trt.PluginFieldCollection([]),
        )
        layer = network.add_plugin_v2(inputs, plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitAdaptiveNormFp32 plugin")
        return layer.get_output(0)
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceAdaptiveNorm", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceAdaptiveNorm plugin creator is not registered")
        shift = cast(network, shift, trt.float32)
        scale = cast(network, scale, trt.float32)
        plugin = creator.create_plugin("wan2_2_source_adaptive_norm", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([normalized, shift, scale], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceAdaptiveNorm plugin")
        return layer.get_output(0)
    one = constant(network, np.ones((1, 1), dtype=np.float32))
    scale_plus_one = network.add_elementwise(scale, one, trt.ElementWiseOperation.SUM).get_output(0)
    y = network.add_elementwise(
        normalized, scale_plus_one, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(y, shift, trt.ElementWiseOperation.SUM).get_output(0)


def gelu_tanh(network, x):
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceGelu", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceGelu plugin creator is not registered")
        x = cast(network, x, trt.bfloat16)
        plugin = creator.create_plugin("wan2_2_source_gelu", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceGelu plugin")
        return layer.get_output(0)
    if _USE_DIT_CUDA_NUMERICS:
        creator = trt.get_plugin_registry().get_creator("Wan22DitGelu", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitGelu plugin creator is not registered")
        x = cast(network, x, trt.bfloat16)
        plugin = creator.create_plugin(
            f"wan2_2_dit_gelu_{network.num_layers}", trt.PluginFieldCollection([])
        )
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitGelu plugin")
        return layer.get_output(0)
    output = network.add_activation(x, trt.ActivationType.GELU_TANH).get_output(0)
    return bf16_barrier(network, output, "gelu")


def silu(network, x):
    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceSilu", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceSilu plugin creator is not registered")
        x = cast(network, x, trt.float32)
        plugin = creator.create_plugin("wan2_2_source_silu", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceSilu plugin")
        return layer.get_output(0)
    if _USE_DIT_CUDA_NUMERICS and _USE_DIT_TIME_SILU:
        creator = trt.get_plugin_registry().get_creator("Wan22DitSiluFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitSiluFp32 plugin creator is not registered")
        x = cast(network, x, trt.float32)
        plugin = creator.create_plugin(
            f"wan2_2_dit_silu_fp32_{network.num_layers}", trt.PluginFieldCollection([])
        )
        layer = network.add_plugin_v2([x], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitSiluFp32 plugin")
        return layer.get_output(0)
    sigmoid = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(x, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)


def rows_to_heads(network, x, seq: int, heads: int, head_dim: int):
    reshape = network.add_shuffle(x)
    reshape.reshape_dims = (seq, heads, head_dim)
    reshape.second_transpose = trt.Permutation([1, 0, 2])
    batched = network.add_shuffle(reshape.get_output(0))
    batched.reshape_dims = (1, heads, seq, head_dim)
    return batched.get_output(0)


def heads_to_rows(network, x, seq: int, hidden_size: int):
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = trt.Permutation([0, 2, 1, 3])
    shuffle.reshape_dims = (seq, hidden_size)
    return shuffle.get_output(0)


def source_cudnn_sdpa(
    network,
    q,
    k,
    v,
    *,
    q_seq: int,
    kv_seq: int,
    heads: int,
    head_dim: int,
    scale: float | None,
    fp32_accumulation: bool,
):
    """Use source-exact cuDNN SDPA for the two qualified 720p contracts."""

    if not _USE_DIT_CUDA_NUMERICS or scale is not None or fp32_accumulation:
        return None
    common_shape = q_seq == 27_280 and heads == 24 and head_dim == 128
    if not common_shape or kv_seq not in (27_280, 512):
        return None
    hidden_size = heads * head_dim
    if (
        tuple(int(value) for value in q.shape) != (q_seq, hidden_size)
        or tuple(int(value) for value in k.shape) != (kv_seq, hidden_size)
        or tuple(int(value) for value in v.shape) != (kv_seq, hidden_size)
    ):
        return None
    if q.dtype != trt.bfloat16 or k.dtype != trt.bfloat16 or v.dtype != trt.bfloat16:
        return None

    attention_kind = 0 if kv_seq == 27_280 else 1
    kind_name = "self" if attention_kind == 0 else "cross"

    # The official tensors are contiguous BSHD, then transposed to logical
    # BHSD as a view.  Preserve that physical layout for the cuDNN plugin; the
    # existing rows_to_heads helper instead materializes a physical BHSD tensor.
    q_bshd = network.add_shuffle(q)
    q_bshd.reshape_dims = (1, q_seq, heads, head_dim)
    k_bshd = network.add_shuffle(k)
    k_bshd.reshape_dims = (1, kv_seq, heads, head_dim)
    v_bshd = network.add_shuffle(v)
    v_bshd.reshape_dims = (1, kv_seq, heads, head_dim)

    creator = trt.get_plugin_registry().get_creator("Wan22DitCudnnSdpa", "1", "")
    if creator is None:
        raise RuntimeError("Wan22DitCudnnSdpa plugin creator is not registered")
    field_values = {
        "attention_kind": np.array([attention_kind], dtype=np.int32),
        "batch": np.array([1], dtype=np.int32),
        "heads": np.array([heads], dtype=np.int32),
        "q_sequence": np.array([q_seq], dtype=np.int32),
        "kv_sequence": np.array([kv_seq], dtype=np.int32),
        "head_dimension": np.array([head_dim], dtype=np.int32),
    }
    fields = trt.PluginFieldCollection(
        [
            trt.PluginField(name, value, trt.PluginFieldType.INT32)
            for name, value in field_values.items()
        ]
    )
    instance = f"wan2_2_dit_{kind_name}_cudnn_sdpa_{network.num_layers}"
    plugin = creator.create_plugin(instance, fields)
    if plugin is None:
        raise RuntimeError(f"Could not create Wan22DitCudnnSdpa {kind_name} plugin")
    layer = network.add_plugin_v2(
        [q_bshd.get_output(0), k_bshd.get_output(0), v_bshd.get_output(0)], plugin
    )
    if layer is None:
        raise RuntimeError(f"Could not add Wan22DitCudnnSdpa {kind_name} plugin")

    rows = network.add_shuffle(layer.get_output(0))
    rows.reshape_dims = (q_seq, hidden_size)
    return rows.get_output(0)


def rotary(network, x, cos_half, sin_half, seq: int, heads: int, head_dim: int):
    """Apply source-compatible RoPE before returning BF16 attention rows.

    Upstream Wan2.2 promotes Q/K to FP64 for its complex multiply and only
    converts the rotated values to BF16 at the attention boundary.  TensorRT
    does not expose an FP64 rotary layer, so FP32 is the closest supported
    boundary.  Performing the rotation directly in BF16 rounds both the
    trigonometric tables and every multiply/add, and that error compounds over
    the 30 denoiser blocks.
    """

    if _USE_SOURCE_LINEAR_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceRotary", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceRotary plugin creator is not registered")
        x = cast(network, x, trt.float32)
        cos64 = np.asarray(cos_half, dtype=np.float64).reshape(seq, head_dim // 2)
        sin64 = np.asarray(sin_half, dtype=np.float64).reshape(seq, head_dim // 2)
        cos_high = cos64.astype(np.float32)
        sin_high = sin64.astype(np.float32)
        cos_low = (cos64 - cos_high.astype(np.float64)).astype(np.float32)
        sin_low = (sin64 - sin_high.astype(np.float64)).astype(np.float32)
        inputs = [
            x,
            constant(network, cos_high),
            constant(network, sin_high),
            constant(network, cos_low),
            constant(network, sin_low),
        ]
        plugin = creator.create_plugin("wan2_2_source_rotary", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2(inputs, plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceRotary plugin")
        return layer.get_output(0)

    if _USE_DIT_CUDA_NUMERICS:
        creator = trt.get_plugin_registry().get_creator("Wan22DitRotary", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitRotary plugin creator is not registered")
        x = cast(network, x, trt.float32)
        cos64 = np.asarray(cos_half, dtype=np.float64).reshape(seq, head_dim // 2)
        sin64 = np.asarray(sin_half, dtype=np.float64).reshape(seq, head_dim // 2)
        cos_high = cos64.astype(np.float32)
        sin_high = sin64.astype(np.float32)
        cos_low = (cos64 - cos_high.astype(np.float64)).astype(np.float32)
        sin_low = (sin64 - sin_high.astype(np.float64)).astype(np.float32)
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "heads", np.array([heads], dtype=np.int32), trt.PluginFieldType.INT32
                ),
                trt.PluginField(
                    "head_dim",
                    np.array([head_dim], dtype=np.int32),
                    trt.PluginFieldType.INT32,
                ),
            ]
        )
        plugin = creator.create_plugin(f"wan2_2_dit_rotary_{network.num_layers}", fields)
        inputs = [
            x,
            constant(network, cos_high),
            constant(network, sin_high),
            constant(network, cos_low),
            constant(network, sin_low),
        ]
        layer = network.add_plugin_v2(inputs, plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitRotary plugin")
        return layer.get_output(0)

    x = rows_to_heads(network, x, seq, heads, head_dim)
    x = cast(network, x, trt.float32)
    cos_t = constant(network, np.asarray(cos_half, dtype=np.float32).reshape(1, seq, head_dim // 2))
    sin_t = constant(network, np.asarray(sin_half, dtype=np.float32).reshape(1, seq, head_dim // 2))
    layer = network.add_rotary_embedding(x, cos_t, sin_t, True, head_dim)
    rotated = bf16_barrier(network, layer.get_output(0), "rotary")
    return heads_to_rows(network, rotated, seq, heads * head_dim)


def attention(
    network,
    q,
    k,
    v,
    *,
    q_seq: int,
    kv_seq: int,
    heads: int,
    head_dim: int,
    scale: float | None = None,
    fp32_accumulation: bool = False,
):
    # Upstream attention explicitly converts Q/K to V's dtype immediately
    # before scaled dot-product attention.  WanRMSNorm returns FP32 Q/K while
    # V comes from an autocast BF16 linear.
    q = cast(network, q, v.dtype)
    k = cast(network, k, v.dtype)
    source_result = source_cudnn_sdpa(
        network,
        q,
        k,
        v,
        q_seq=q_seq,
        kv_seq=kv_seq,
        heads=heads,
        head_dim=head_dim,
        scale=scale,
        fp32_accumulation=fp32_accumulation,
    )
    if source_result is not None:
        return source_result
    q4 = rows_to_heads(network, q, q_seq, heads, head_dim)
    k4 = rows_to_heads(network, k, kv_seq, heads, head_dim)
    v4 = rows_to_heads(network, v, kv_seq, heads, head_dim)
    if _USE_SOURCE_ATTENTION_PLUGIN:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceAttention", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceAttention plugin creator is not registered")
        plugin = creator.create_plugin("wan2_2_source_attention", trt.PluginFieldCollection([]))
        layer = network.add_plugin_v2([q4, k4, v4], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceAttention plugin")
        return heads_to_rows(network, layer.get_output(0), q_seq, heads * head_dim)
    output_dtype = q4.dtype
    if fp32_accumulation:
        q4 = cast(network, q4, trt.float32)
        k4 = cast(network, k4, trt.float32)
        v4 = cast(network, v4, trt.float32)
    factor = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    factor_t = constant(network, np.array([[[[factor]]]], dtype=np.float32))
    factor_t = cast(network, factor_t, q4.dtype)
    q4 = network.add_elementwise(q4, factor_t, trt.ElementWiseOperation.PROD).get_output(0)
    if fp32_accumulation:
        logits = network.add_matrix_multiply(
            q4,
            trt.MatrixOperation.NONE,
            k4,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        softmax = network.add_softmax(logits)
        softmax.axes = 1 << 3
        result = network.add_matrix_multiply(
            softmax.get_output(0),
            trt.MatrixOperation.NONE,
            v4,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        result = cast(network, result, output_dtype)
        if output_dtype == trt.bfloat16:
            result = bf16_barrier(network, result, "attention")
        return heads_to_rows(network, result, q_seq, heads * head_dim)
    layer = network.add_attention(
        q4,
        k4,
        v4,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    layer.decomposable = True
    result = bf16_barrier(network, layer.get_output(0), "attention")
    return heads_to_rows(network, result, q_seq, heads * head_dim)


def add_fp32_residual(
    network,
    x,
    update,
    gate=None,
    *,
    round_bf16=False,
    source_exact_gated_stage=None,
):
    if source_exact_gated_stage not in (None, "self_attention", "ffn"):
        raise ValueError(f"Unknown source-exact gated residual stage: {source_exact_gated_stage}")
    source_exact_gated = (
        source_exact_gated_stage == "self_attention" and _USE_DIT_SELF_GATED_RESIDUAL
    ) or (source_exact_gated_stage == "ffn" and _USE_DIT_FFN_GATED_RESIDUAL)
    if (
        _USE_DIT_CUDA_NUMERICS
        and source_exact_gated
        and gate is not None
        and not round_bf16
        and tuple(x.shape) == (27_280, 3_072)
        and tuple(update.shape) == (27_280, 3_072)
        and tuple(gate.shape) == (1, 3_072)
    ):
        creator = trt.get_plugin_registry().get_creator("Wan22DitGatedResidualFp32", "1", "")
        if creator is None:
            raise RuntimeError("Wan22DitGatedResidualFp32 plugin creator is not registered")
        x = cast(network, x, trt.float32)
        update = cast(network, update, trt.float32)
        gate = cast(network, gate, trt.float32)
        plugin = creator.create_plugin(
            f"wan2_2_dit_self_gated_residual_{network.num_layers}",
            trt.PluginFieldCollection([]),
        )
        layer = network.add_plugin_v2([x, update, gate], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22DitGatedResidualFp32 plugin")
        return layer.get_output(0)
    if _USE_SOURCE_LINEAR_PLUGIN and not round_bf16:
        creator = trt.get_plugin_registry().get_creator("Wan22SourceResidual", "1", "")
        if creator is None:
            raise RuntimeError("Wan22SourceResidual plugin creator is not registered")
        use_gate = gate is not None
        x = cast(network, x, trt.float32)
        if gate is None:
            gate = constant(network, np.ones((1, x.shape[-1]), dtype=np.float32))
        else:
            gate = cast(network, gate, trt.float32)
        fields = trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "gated",
                    np.array([int(use_gate)], dtype=np.int32),
                    trt.PluginFieldType.INT32,
                )
            ]
        )
        plugin = creator.create_plugin("wan2_2_source_residual", fields)
        layer = network.add_plugin_v2([x, update, gate], plugin)
        if layer is None:
            raise RuntimeError("Could not add Wan22SourceResidual plugin")
        return layer.get_output(0)
    x = cast(network, x, trt.float32)
    update = cast(network, update, trt.float32)
    if gate is not None:
        update = network.add_elementwise(update, gate, trt.ElementWiseOperation.PROD).get_output(0)
    result = network.add_elementwise(x, update, trt.ElementWiseOperation.SUM).get_output(0)
    if round_bf16:
        result = cast(network, result, trt.bfloat16)
    return result
