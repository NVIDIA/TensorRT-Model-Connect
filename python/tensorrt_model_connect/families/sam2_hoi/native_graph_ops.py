# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT Network Definition helpers owned by the SAM2-HOI family.

The reviewed checkpoint stores every learned tensor as FP32.  CUDA autocast
executes most of the image encoder and detector in BF16, so a direct TensorRT
graph must materialize BF16 weights instead of relying on weakly typed layer
precision.  TensorRT's NumPy bridge does not infer ``ml_dtypes.bfloat16``;
``trt.Weights(DataType.BF16, pointer, count)`` is therefore used explicitly and
the backing arrays are retained until the serialized plan has been built.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


_BF16_WEIGHT_REFS: list[np.ndarray] = []


def _trt():
    from tensorrt_model_connect import trt_compat

    return trt_compat.get_trt()


def normalize_precision(precision: str) -> str:
    normalized = str(precision).strip().lower()
    if normalized not in {"fp32", "bf16"}:
        raise ValueError(
            f"SAM2 HOI native TensorRT builders support fp32 or bf16, got {precision!r}"
        )
    return normalized


def runtime_dtype(precision: str):
    trt = _trt()
    return trt.bfloat16 if normalize_precision(precision) == "bf16" else trt.float32


def reset_weight_refs() -> None:
    """Release buffers retained for a previous completed plan build."""

    _BF16_WEIGHT_REFS.clear()


def make_weights(values: Any | None, *, precision: str = "fp32"):
    """Create TensorRT weights with an explicit FP32 or BF16 storage type."""

    trt = _trt()
    if values is None:
        return trt.Weights()
    array = np.ascontiguousarray(values, dtype=np.float32)
    if normalize_precision(precision) == "fp32":
        return trt.Weights(array)

    import ml_dtypes

    bf16 = np.ascontiguousarray(array.astype(ml_dtypes.bfloat16))
    _BF16_WEIGHT_REFS.append(bf16)
    return trt.Weights(trt.bfloat16, bf16.ctypes.data, bf16.size)


def cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    layer = network.add_cast(tensor, dtype)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI cast")
    return layer.get_output(0)


def add_constant(
    network,
    shape: tuple[int, ...],
    values: Any,
    *,
    precision: str = "fp32",
):
    array = np.asarray(values, dtype=np.float32).reshape(shape)
    layer = network.add_constant(
        shape,
        make_weights(array, precision=precision),
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI constant")
    return layer.get_output(0)


def add_int32_constant(network, shape: tuple[int, ...], values: Any):
    trt = _trt()
    array = np.ascontiguousarray(values, dtype=np.int32).reshape(shape)
    layer = network.add_constant(shape, trt.Weights(array))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI int32 constant")
    return layer.get_output(0)


def add_linear(
    network,
    inp,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str,
):
    """Apply a PyTorch Linear whose checkpoint weight is [out, in]."""

    trt = _trt()
    output_width, input_width = tuple(weight_out_in.shape)
    rank = len(tuple(inp.shape))
    rhs_shape = (1,) * max(0, rank - 2) + (input_width, output_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(weight_out_in, dtype=np.float32).T.reshape(rhs_shape),
        precision=precision,
    )
    out = network.add_matrix_multiply(
        inp,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    if out is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI matrix multiply")
    result = out.get_output(0)
    if bias is None:
        return result
    bias_shape = (1,) * max(0, rank - 1) + (output_width,)
    bias_tensor = add_constant(
        network,
        bias_shape,
        np.asarray(bias, dtype=np.float32).reshape(bias_shape),
        precision=precision,
    )
    summed = network.add_elementwise(result, bias_tensor, trt.ElementWiseOperation.SUM)
    if summed is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI linear bias")
    return summed.get_output(0)


def add_conv2d(
    network,
    inp,
    weight: np.ndarray,
    bias: np.ndarray | None,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
    precision: str,
):
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 4:
        raise ValueError(f"SAM2 HOI Conv2d weight must have rank 4, got {weight.shape}")
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=(kernel_h, kernel_w),
        kernel=make_weights(weight, precision=precision),
        bias=make_weights(bias, precision=precision),
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI convolution")
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups
    return layer.get_output(0)


def fold_batch_norm(
    weight: np.ndarray,
    bias: np.ndarray | None,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold an eval-mode BatchNorm2d into its preceding convolution."""

    weight = np.asarray(weight, dtype=np.float32)
    conv_bias = (
        np.zeros(weight.shape[0], dtype=np.float32)
        if bias is None
        else np.asarray(bias, dtype=np.float32)
    )
    scale = np.asarray(gamma, dtype=np.float32) / np.sqrt(
        np.asarray(running_variance, dtype=np.float32) + np.float32(epsilon)
    )
    folded_weight = weight * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    folded_bias = (conv_bias - np.asarray(running_mean, dtype=np.float32)) * scale + np.asarray(
        beta, dtype=np.float32
    )
    return (
        np.ascontiguousarray(folded_weight, dtype=np.float32),
        np.ascontiguousarray(folded_bias, dtype=np.float32),
    )


def batch_norm_affine_parameters(
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the FP32 channel scale and shift for eval-mode BatchNorm2d."""

    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    running_mean = np.asarray(running_mean, dtype=np.float32)
    running_variance = np.asarray(running_variance, dtype=np.float32)
    scale = gamma / np.sqrt(running_variance + np.float32(epsilon))
    shift = beta - running_mean * scale
    return (
        np.ascontiguousarray(scale, dtype=np.float32),
        np.ascontiguousarray(shift, dtype=np.float32),
    )


def batch_norm_affine_parameters_from_invstd(
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    invstd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eval-mode affine values from an externally computed FP32 invstd."""

    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    running_mean = np.ascontiguousarray(running_mean, dtype=np.float32)
    invstd = np.ascontiguousarray(invstd, dtype=np.float32)
    if not (
        gamma.ndim == 1
        and beta.shape == gamma.shape
        and running_mean.shape == gamma.shape
        and invstd.shape == gamma.shape
    ):
        raise ValueError("SAM2 HOI BatchNorm exact affine parameter shape drift")
    scale = np.multiply(gamma, invstd, dtype=np.float32)
    shift = np.subtract(
        beta,
        np.multiply(running_mean, scale, dtype=np.float32),
        dtype=np.float32,
    )
    return (
        np.ascontiguousarray(scale, dtype=np.float32),
        np.ascontiguousarray(shift, dtype=np.float32),
    )


def add_batch_norm2d_affine(
    network,
    inp,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_variance: np.ndarray,
    *,
    epsilon: float,
    output_dtype,
):
    """Apply eval-mode BatchNorm2d in FP32, then cast to ``output_dtype``.

    This keeps the source autocast boundary intact: the convolution result is
    rounded before BatchNorm, while the frozen statistics and affine parameters
    remain FP32.  Folding these values into BF16 convolution weights changes
    both the weight rounding point and the resulting activation.
    """

    trt = _trt()
    if len(tuple(inp.shape)) != 4:
        raise ValueError(f"SAM2 HOI BatchNorm2d input must have rank 4, got {inp.shape}")
    channels = int(np.asarray(gamma).size)
    parameter_shape = (1, channels, 1, 1)
    scale_values, shift_values = batch_norm_affine_parameters(
        gamma,
        beta,
        running_mean,
        running_variance,
        epsilon=epsilon,
    )
    compute = cast(network, inp, trt.float32)
    scale = add_constant(
        network,
        parameter_shape,
        scale_values.reshape(parameter_shape),
        precision="fp32",
    )
    shift = add_constant(
        network,
        parameter_shape,
        shift_values.reshape(parameter_shape),
        precision="fp32",
    )
    scaled = network.add_elementwise(compute, scale, trt.ElementWiseOperation.PROD)
    if scaled is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI BatchNorm2d scale")
    shifted = network.add_elementwise(
        scaled.get_output(0),
        shift,
        trt.ElementWiseOperation.SUM,
    )
    if shifted is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI BatchNorm2d shift")
    return cast(network, shifted.get_output(0), output_dtype)


def add_batch_norm2d_affine_from_invstd(
    network,
    inp,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    invstd: np.ndarray,
    *,
    output_dtype,
):
    """Apply eval-mode BatchNorm using CUDA-produced builder-time invstd constants."""

    trt = _trt()
    if len(tuple(inp.shape)) != 4:
        raise ValueError(f"SAM2 HOI BatchNorm2d input must have rank 4, got {inp.shape}")
    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)
    running_mean = np.ascontiguousarray(running_mean, dtype=np.float32)
    invstd = np.ascontiguousarray(invstd, dtype=np.float32)
    if not (
        gamma.ndim == 1
        and beta.shape == gamma.shape
        and running_mean.shape == gamma.shape
        and invstd.shape == gamma.shape
    ):
        raise ValueError("SAM2 HOI BatchNorm exact affine parameter shape drift")
    channels = int(gamma.size)
    parameter_shape = (1, channels, 1, 1)
    compute = cast(network, inp, trt.float32)
    mean_tensor = add_constant(
        network,
        parameter_shape,
        running_mean.reshape(parameter_shape),
        precision="fp32",
    )
    gamma_tensor = add_constant(
        network,
        parameter_shape,
        gamma.reshape(parameter_shape),
        precision="fp32",
    )
    invstd_tensor = add_constant(
        network,
        parameter_shape,
        invstd.reshape(parameter_shape),
        precision="fp32",
    )
    beta_tensor = add_constant(
        network,
        parameter_shape,
        beta.reshape(parameter_shape),
        precision="fp32",
    )
    centered = network.add_elementwise(
        compute,
        mean_tensor,
        trt.ElementWiseOperation.SUB,
    )
    if centered is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BatchNorm2d centering")
    gamma_centered = network.add_elementwise(
        gamma_tensor,
        centered.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    if gamma_centered is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BatchNorm2d gamma")
    scaled = network.add_elementwise(
        gamma_centered.get_output(0),
        invstd_tensor,
        trt.ElementWiseOperation.PROD,
    )
    if scaled is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BatchNorm2d invstd")
    shifted = network.add_elementwise(
        scaled.get_output(0),
        beta_tensor,
        trt.ElementWiseOperation.SUM,
    )
    if shifted is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI exact BatchNorm2d beta")
    return cast(network, shifted.get_output(0), output_dtype)


def add_activation(network, inp, kind: str):
    trt = _trt()
    normalized = kind.lower()
    if normalized == "relu":
        layer = network.add_activation(inp, trt.ActivationType.RELU)
        return layer.get_output(0)
    if normalized == "silu":
        # PyTorch evaluates SiLU in the FP32 opmath type before rounding the
        # result back to BF16.  Rounding sigmoid(x) to BF16 before multiplying
        # changes values near BF16 ties, so keep the complete source formula in
        # FP32 and cast only its final result.
        output_dtype = inp.dtype
        compute = cast(network, inp, trt.float32)
        negative = network.add_unary(compute, trt.UnaryOperation.NEG).get_output(0)
        exponential = network.add_unary(negative, trt.UnaryOperation.EXP).get_output(0)
        rank = max(1, len(tuple(inp.shape)))
        one = add_constant(
            network,
            (1,) * rank,
            np.asarray([1.0]),
            precision="fp32",
        )
        denominator = network.add_elementwise(
            one,
            exponential,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        result = network.add_elementwise(
            compute,
            denominator,
            trt.ElementWiseOperation.DIV,
        ).get_output(0)
        return cast(network, result, output_dtype)
    if normalized == "gelu":
        # nn.GELU() uses the exact erf formulation by default.
        output_dtype = inp.dtype
        compute = cast(network, inp, trt.float32)
        rank = max(1, len(tuple(inp.shape)))
        shape = (1,) * rank
        inv_sqrt_two = add_constant(
            network, shape, np.asarray([1.0 / np.sqrt(2.0)]), precision="fp32"
        )
        scaled = network.add_elementwise(
            compute, inv_sqrt_two, trt.ElementWiseOperation.PROD
        ).get_output(0)
        erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        one = add_constant(network, shape, np.asarray([1.0]), precision="fp32")
        one_plus = network.add_elementwise(one, erf, trt.ElementWiseOperation.SUM).get_output(0)
        half = add_constant(network, shape, np.asarray([0.5]), precision="fp32")
        half_x = network.add_elementwise(half, compute, trt.ElementWiseOperation.PROD).get_output(0)
        result = network.add_elementwise(
            half_x, one_plus, trt.ElementWiseOperation.PROD
        ).get_output(0)
        return cast(
            network,
            result,
            output_dtype,
        )
    raise ValueError(f"Unsupported SAM2 HOI activation {kind!r}")


def add_layer_norm(
    network,
    inp,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    epsilon: float,
    precision: str,
):
    """Add native TensorRT LayerNorm over the final tensor dimension."""

    rank = len(tuple(inp.shape))
    width = int(np.asarray(gamma).size)
    parameter_shape = (1,) * max(0, rank - 1) + (width,)
    scale = add_constant(
        network,
        parameter_shape,
        np.asarray(gamma).reshape(parameter_shape),
        precision=precision,
    )
    shift = add_constant(
        network,
        parameter_shape,
        np.asarray(beta).reshape(parameter_shape),
        precision=precision,
    )
    layer = network.add_normalization_v2(inp, scale, shift, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI LayerNorm")
    layer.epsilon = float(epsilon)
    return layer.get_output(0)


def plugin_creator(name: str, *, version: str = "1", namespace: str = ""):
    trt = _trt()
    registry = trt.get_plugin_registry()
    if registry is None:
        raise RuntimeError("TensorRT plugin registry is unavailable")
    get_creator = getattr(registry, "get_creator", None)
    if get_creator is None:
        get_creator = getattr(registry, "get_plugin_creator", None)
    if get_creator is None:
        raise RuntimeError("TensorRT plugin registry has no creator lookup API")
    creator = get_creator(name, version, namespace)
    if creator is None:
        raise RuntimeError(
            f"SAM2 HOI TensorRT plugin creator {name!r} version {version!r} is unavailable"
        )
    return creator


def add_plugin(
    network,
    name: str,
    inputs: Iterable[Any],
    *,
    instance_name: str,
    version: str = "1",
):
    trt = _trt()
    creator = plugin_creator(name, version=version)
    plugin = creator.create_plugin(instance_name, trt.PluginFieldCollection([]))
    if plugin is None:
        raise RuntimeError(f"Could not create SAM2 HOI TensorRT plugin {name!r}")
    layer = network.add_plugin_v2(list(inputs), plugin)
    if layer is None:
        raise RuntimeError(f"Could not add SAM2 HOI TensorRT plugin {name!r}")
    return layer.get_output(0)


def add_resize(
    network,
    inp,
    shape: tuple[int, ...],
    *,
    mode: str,
    coordinate_transformation: str | None = None,
):
    trt = _trt()
    layer = network.add_resize(inp)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI resize")
    normalized = mode.lower()
    if normalized == "nearest":
        layer.resize_mode = trt.InterpolationMode.NEAREST
    elif normalized in {"linear", "bilinear"}:
        layer.resize_mode = trt.InterpolationMode.LINEAR
    elif normalized == "cubic":
        layer.resize_mode = trt.InterpolationMode.CUBIC
    else:
        raise ValueError(f"Unsupported SAM2 HOI resize mode {mode!r}")
    layer.shape = shape
    if coordinate_transformation is not None and hasattr(layer, "coordinate_transformation"):
        transformations = {
            "align_corners": trt.ResizeCoordinateTransformation.ALIGN_CORNERS,
            "asymmetric": trt.ResizeCoordinateTransformation.ASYMMETRIC,
            "half_pixel": trt.ResizeCoordinateTransformation.HALF_PIXEL,
        }
        layer.coordinate_transformation = transformations[coordinate_transformation]
    return layer.get_output(0)


def mark_output(network, tensor, name: str, *, dtype=None) -> None:
    if dtype is not None:
        tensor = cast(network, tensor, dtype)
    tensor.name = name
    network.mark_output(tensor)


__all__ = [
    "add_activation",
    "add_batch_norm2d_affine",
    "add_batch_norm2d_affine_from_invstd",
    "add_constant",
    "add_conv2d",
    "add_int32_constant",
    "add_layer_norm",
    "add_linear",
    "add_plugin",
    "add_resize",
    "cast",
    "batch_norm_affine_parameters",
    "batch_norm_affine_parameters_from_invstd",
    "fold_batch_norm",
    "make_weights",
    "mark_output",
    "normalize_precision",
    "plugin_creator",
    "reset_weight_refs",
    "runtime_dtype",
]
