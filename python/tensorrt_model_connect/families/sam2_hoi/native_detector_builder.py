# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT Network Definition builder for the SAM2-HOI detector.

The graph is the fixed-B1, three-level Co-DINO detector embedded in the
reviewed SAM2-HOI checkpoint.  Model construction only reads checkpoint
arrays.  Every executable operator is expressed as a TensorRT network layer
or one of the family-owned exact-arithmetic plugins.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat

from . import native_graph_ops as graph_ops


_PREFIX = "image_encoder.hoi_head.query_head."
_TRANSFORMER = _PREFIX + "transformer."
_LEVEL_SHAPES = ((128, 128), (64, 64), (32, 32))
_LEVEL_STARTS = (0, 128 * 128, 128 * 128 + 64 * 64)
_SPATIAL_TOKENS = sum(height * width for height, width in _LEVEL_SHAPES)
_HIDDEN = 256
_FFN_HIDDEN = 2048
_NUM_HEADS = 8
_HEAD_WIDTH = 32
_NUM_LEVELS = 3
_NUM_POINTS = 4
_NUM_ENCODER_LAYERS = 6
_NUM_DECODER_LAYERS = 6
_NUM_QUERIES = 1500
_NUM_CLASSES = 4
_WORKSPACE_BYTES = 8 << 30
_BUILDER_OPTIMIZATION_LEVEL = 5
_AVG_TIMING_ITERATIONS = 8
_MAX_AUX_STREAMS = 0

# PyTorch's BF16 Linear keeps the learned bias inside the FP32-accumulating
# CUDA kernel before its single BF16 output rounding.  TensorRT 11.1 can
# instead fuse the generic MatrixMultiply/cast/add sequence into a tactic that
# rounds the GEMM before adding the bias.  Exact-input L4 qualification proved
# that only these fixed sites need the source-equivalent fused-bias schedule.
_FUSED_BF16_LINEAR_PREFIXES = frozenset(
    {
        *(
            _TRANSFORMER + f"encoder.layers.{layer}.attentions.0.output_proj"
            for layer in range(_NUM_ENCODER_LAYERS)
        ),
        *(
            _TRANSFORMER + f"decoder.layers.{layer}.attentions.1.output_proj"
            for layer in range(_NUM_DECODER_LAYERS)
        ),
        _TRANSFORMER + "enc_output",
        _PREFIX + f"reg_branches.{_NUM_DECODER_LAYERS}.0",
        _PREFIX + f"reg_branches.{_NUM_DECODER_LAYERS}.2",
    }
)


def _expected_weight_shapes() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {
        _TRANSFORMER + "level_embeds": (_NUM_LEVELS, _HIDDEN),
        _TRANSFORMER + "enc_output.weight": (_HIDDEN, _HIDDEN),
        _TRANSFORMER + "enc_output.bias": (_HIDDEN,),
        _TRANSFORMER + "enc_output_norm.weight": (_HIDDEN,),
        _TRANSFORMER + "enc_output_norm.bias": (_HIDDEN,),
        _TRANSFORMER + "query_embed.weight": (_NUM_QUERIES, _HIDDEN),
        _TRANSFORMER + "decoder.ref_point_head.0.weight": (_HIDDEN, 2 * _HIDDEN),
        _TRANSFORMER + "decoder.ref_point_head.0.bias": (_HIDDEN,),
        _TRANSFORMER + "decoder.ref_point_head.2.weight": (_HIDDEN, _HIDDEN),
        _TRANSFORMER + "decoder.ref_point_head.2.bias": (_HIDDEN,),
        _TRANSFORMER + "decoder.norm.weight": (_HIDDEN,),
        _TRANSFORMER + "decoder.norm.bias": (_HIDDEN,),
    }
    for layer in range(_NUM_ENCODER_LAYERS):
        base = _TRANSFORMER + f"encoder.layers.{layer}."
        attention = base + "attentions.0."
        expected.update(
            {
                attention + "sampling_offsets.weight": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS * 2,
                    _HIDDEN,
                ),
                attention + "sampling_offsets.bias": (_NUM_HEADS * _NUM_LEVELS * _NUM_POINTS * 2,),
                attention + "attention_weights.weight": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS,
                    _HIDDEN,
                ),
                attention + "attention_weights.bias": (_NUM_HEADS * _NUM_LEVELS * _NUM_POINTS,),
                attention + "value_proj.weight": (_HIDDEN, _HIDDEN),
                attention + "value_proj.bias": (_HIDDEN,),
                attention + "output_proj.weight": (_HIDDEN, _HIDDEN),
                attention + "output_proj.bias": (_HIDDEN,),
                base + "ffns.0.layers.0.0.weight": (_FFN_HIDDEN, _HIDDEN),
                base + "ffns.0.layers.0.0.bias": (_FFN_HIDDEN,),
                base + "ffns.0.layers.1.weight": (_HIDDEN, _FFN_HIDDEN),
                base + "ffns.0.layers.1.bias": (_HIDDEN,),
                base + "norms.0.weight": (_HIDDEN,),
                base + "norms.0.bias": (_HIDDEN,),
                base + "norms.1.weight": (_HIDDEN,),
                base + "norms.1.bias": (_HIDDEN,),
            }
        )
    for layer in range(_NUM_DECODER_LAYERS):
        base = _TRANSFORMER + f"decoder.layers.{layer}."
        self_attention = base + "attentions.0.attn."
        cross_attention = base + "attentions.1."
        expected.update(
            {
                self_attention + "in_proj_weight": (3 * _HIDDEN, _HIDDEN),
                self_attention + "in_proj_bias": (3 * _HIDDEN,),
                self_attention + "out_proj.weight": (_HIDDEN, _HIDDEN),
                self_attention + "out_proj.bias": (_HIDDEN,),
                cross_attention + "sampling_offsets.weight": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS * 2,
                    _HIDDEN,
                ),
                cross_attention + "sampling_offsets.bias": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS * 2,
                ),
                cross_attention + "attention_weights.weight": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS,
                    _HIDDEN,
                ),
                cross_attention + "attention_weights.bias": (
                    _NUM_HEADS * _NUM_LEVELS * _NUM_POINTS,
                ),
                cross_attention + "value_proj.weight": (_HIDDEN, _HIDDEN),
                cross_attention + "value_proj.bias": (_HIDDEN,),
                cross_attention + "output_proj.weight": (_HIDDEN, _HIDDEN),
                cross_attention + "output_proj.bias": (_HIDDEN,),
                base + "ffns.0.layers.0.0.weight": (_FFN_HIDDEN, _HIDDEN),
                base + "ffns.0.layers.0.0.bias": (_FFN_HIDDEN,),
                base + "ffns.0.layers.1.weight": (_HIDDEN, _FFN_HIDDEN),
                base + "ffns.0.layers.1.bias": (_HIDDEN,),
                base + "norms.0.weight": (_HIDDEN,),
                base + "norms.0.bias": (_HIDDEN,),
                base + "norms.1.weight": (_HIDDEN,),
                base + "norms.1.bias": (_HIDDEN,),
                base + "norms.2.weight": (_HIDDEN,),
                base + "norms.2.bias": (_HIDDEN,),
            }
        )
    for branch in range(_NUM_DECODER_LAYERS + 1):
        cls = _PREFIX + f"cls_branches.{branch}."
        reg = _PREFIX + f"reg_branches.{branch}."
        expected.update(
            {
                cls + "weight": (_NUM_CLASSES, _HIDDEN),
                cls + "bias": (_NUM_CLASSES,),
                reg + "0.weight": (_HIDDEN, _HIDDEN),
                reg + "0.bias": (_HIDDEN,),
                reg + "2.weight": (_HIDDEN, _HIDDEN),
                reg + "2.bias": (_HIDDEN,),
                reg + "4.weight": (4, _HIDDEN),
                reg + "4.bias": (4,),
            }
        )
    return expected


_EXPECTED_WEIGHT_SHAPES = _expected_weight_shapes()


def detector_required_weight_keys() -> tuple[str, ...]:
    """Return the complete learned-tensor contract for the native graph."""

    return tuple(_EXPECTED_WEIGHT_SHAPES)


def _validate_weights(weights: Mapping[str, np.ndarray]) -> None:
    missing = [key for key in _EXPECTED_WEIGHT_SHAPES if key not in weights]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f" (and {len(missing) - 8} more)"
        raise RuntimeError(f"SAM2 HOI detector checkpoint is missing {preview}{suffix}")
    for key, expected in _EXPECTED_WEIGHT_SHAPES.items():
        actual = tuple(np.asarray(weights[key]).shape)
        if actual != expected:
            raise RuntimeError(
                f"SAM2 HOI detector parameter {key!r} expected shape {expected}, got {actual}"
            )


def _sine_position_encoding(height: int, width: int) -> np.ndarray:
    """Exact all-valid SinePositionalEncoding used by the reviewed head."""

    y = np.arange(1, height + 1, dtype=np.float32).reshape(height, 1)
    x = np.arange(1, width + 1, dtype=np.float32).reshape(1, width)
    epsilon = np.float32(1.0e-6)
    scale = np.float32(2.0 * np.pi)
    y = np.broadcast_to(y / (np.float32(height) + epsilon) * scale, (height, width))
    x = np.broadcast_to(x / (np.float32(width) + epsilon) * scale, (height, width))
    indices = np.arange(128, dtype=np.float32)
    dim_t = np.float32(20.0) ** (2.0 * np.floor(indices / 2.0) / 128.0)

    def encode(coordinates: np.ndarray) -> np.ndarray:
        phase = coordinates[..., None] / dim_t
        encoded = np.empty((height, width, 128), dtype=np.float32)
        encoded[..., 0::2] = np.sin(phase[..., 0::2])
        encoded[..., 1::2] = np.cos(phase[..., 1::2])
        return encoded

    # MMDetection concatenates y before x and returns NCHW.
    return np.ascontiguousarray(
        np.concatenate((encode(y), encode(x)), axis=-1).transpose(2, 0, 1)[None]
    )


def _sine_position_table(network, length: int):
    """Build one exact CUDA SinePositionalEncoding coordinate table.

    PyTorch and TensorRT use the same CUDA arithmetic for the fixed FP32
    normalize, power, sine, and cosine sequence.  Host-side libm differs by a
    few FP32 ulps, which can cross a later BF16 GEMM rounding boundary, so keep
    this fixed table in the TensorRT graph rather than baking a NumPy result.
    """

    trt = trt_compat.get_trt()
    coordinates = graph_ops.add_constant(
        network,
        (length,),
        np.arange(1, length + 1, dtype=np.float32),
        precision="fp32",
    )
    extent = graph_ops.add_constant(network, (1,), [float(length)], precision="fp32")
    epsilon = graph_ops.add_constant(network, (1,), [1.0e-6], precision="fp32")
    denominator = _elementwise(network, extent, epsilon, trt.ElementWiseOperation.SUM)
    coordinates = _elementwise(network, coordinates, denominator, trt.ElementWiseOperation.DIV)
    scale = graph_ops.add_constant(network, (1,), [2.0 * np.pi], precision="fp32")
    coordinates = _elementwise(network, coordinates, scale, trt.ElementWiseOperation.PROD)

    indices = np.arange(128, dtype=np.float32)
    exponent = 2.0 * np.floor(indices / 2.0) / 128.0
    exponent = graph_ops.add_constant(network, (128,), exponent, precision="fp32")
    temperature = graph_ops.add_constant(network, (1,), [20.0], precision="fp32")
    dimension = _elementwise(network, temperature, exponent, trt.ElementWiseOperation.POW)
    phase = _elementwise(
        network,
        _reshape(network, coordinates, (length, 1)),
        _reshape(network, dimension, (1, 128)),
        trt.ElementWiseOperation.DIV,
    )
    even = _slice(network, phase, (0, 0), (length, 64), (1, 2))
    odd = _slice(network, phase, (0, 1), (length, 64), (1, 2))
    sine = _unary(network, even, trt.UnaryOperation.SIN)
    cosine = _unary(network, odd, trt.UnaryOperation.COS)
    sine = _reshape(network, sine, (length, 64, 1))
    cosine = _reshape(network, cosine, (length, 64, 1))
    return _reshape(
        network,
        _concat(network, (sine, cosine), axis=2),
        (length, 128),
    )


def _sine_position_encoding_tensor(network, height: int, width: int):
    trt = trt_compat.get_trt()
    y_table = _sine_position_table(network, height)
    x_table = y_table if height == width else _sine_position_table(network, width)
    y_table = _reshape(network, y_table, (1, height, 1, 128))
    x_table = _reshape(network, x_table, (1, 1, width, 128))
    y_repeat = graph_ops.add_constant(
        network, (1, 1, width, 1), np.ones((1, 1, width, 1)), precision="fp32"
    )
    x_repeat = graph_ops.add_constant(
        network, (1, height, 1, 1), np.ones((1, height, 1, 1)), precision="fp32"
    )
    y_grid = _elementwise(network, y_table, y_repeat, trt.ElementWiseOperation.PROD)
    x_grid = _elementwise(network, x_table, x_repeat, trt.ElementWiseOperation.PROD)
    return _reshape(
        network,
        _concat(network, (y_grid, x_grid), axis=3),
        (1, height * width, _HIDDEN),
    )


def _encoder_reference_points() -> np.ndarray:
    references = []
    for height, width in _LEVEL_SHAPES:
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32) + np.float32(0.5),
            np.arange(width, dtype=np.float32) + np.float32(0.5),
            indexing="ij",
        )
        references.append(np.stack((x.reshape(-1) / width, y.reshape(-1) / height), axis=-1))
    merged = np.concatenate(references, axis=0)[None, :, None, :]
    return np.ascontiguousarray(np.repeat(merged, _NUM_LEVELS, axis=2), dtype=np.float32)


def _encoder_proposal_coordinates() -> tuple[np.ndarray, np.ndarray]:
    proposals = []
    for level, (height, width) in enumerate(_LEVEL_SHAPES):
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        centers = np.stack(
            ((x + np.float32(0.5)) / width, (y + np.float32(0.5)) / height),
            axis=-1,
        )
        size = np.full_like(centers, np.float32(0.05 * (2.0**level)))
        proposals.append(np.concatenate((centers, size), axis=-1).reshape(-1, 4))
    normalized = np.concatenate(proposals, axis=0)[None]
    valid = np.all((normalized > 0.01) & (normalized < 0.99), axis=-1, keepdims=True)
    return np.ascontiguousarray(normalized, dtype=np.float32), valid.astype(np.float32)


def _encoder_proposals() -> tuple[np.ndarray, np.ndarray]:
    normalized, valid = _encoder_proposal_coordinates()
    logits = np.log(normalized / (np.float32(1.0) - normalized))
    logits = np.where(valid.astype(bool), logits, np.float32(np.inf))
    return np.ascontiguousarray(logits, dtype=np.float32), valid


def _reshape(network, tensor, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI reshape")
    layer.reshape_dims = shape
    return layer.get_output(0)


def _transpose(network, tensor, permutation: tuple[int, ...]):
    trt = trt_compat.get_trt()
    layer = network.add_shuffle(tensor)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI transpose")
    layer.first_transpose = trt.Permutation(list(permutation))
    return layer.get_output(0)


def _slice(network, tensor, start, shape, stride):
    layer = network.add_slice(tensor, start, shape, stride)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI slice")
    return layer.get_output(0)


def _concat(network, tensors, axis: int):
    layer = network.add_concatenation(list(tensors))
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI concatenation")
    layer.axis = axis
    return layer.get_output(0)


def _elementwise(network, left, right, operation):
    layer = network.add_elementwise(left, right, operation)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI elementwise layer")
    return layer.get_output(0)


def _force_cast(network, tensor, dtype):
    layer = network.add_cast(tensor, dtype)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI precision boundary")
    return layer.get_output(0)


def _typed_elementwise(network, left, right, operation, dtype):
    left = _force_cast(network, left, dtype)
    right = _force_cast(network, right, dtype)
    output = _elementwise(network, left, right, operation)
    return _force_cast(network, output, dtype)


def _unary(network, tensor, operation):
    layer = network.add_unary(tensor, operation)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI unary layer")
    return layer.get_output(0)


def _clip(network, tensor, minimum: float, maximum: float):
    trt = trt_compat.get_trt()
    layer = network.add_activation(tensor, trt.ActivationType.CLIP)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI clamp")
    layer.alpha = float(minimum)
    layer.beta = float(maximum)
    return layer.get_output(0)


def _linear(network, tensor, weights, prefix: str, *, precision: str):
    return _linear_arrays(
        network,
        tensor,
        np.asarray(weights[prefix + ".weight"], dtype=np.float32),
        np.asarray(weights[prefix + ".bias"], dtype=np.float32),
        precision=precision,
        label=prefix,
    )


def _linear_arrays(network, tensor, weight, bias, *, precision: str, label: str):
    trt = trt_compat.get_trt()
    if precision == "bf16":
        input_cast = network.add_cast(tensor, trt.bfloat16)
        if input_cast is None:
            raise RuntimeError(f"TensorRT failed to cast SAM2 HOI input {label!r}")
        tensor = input_cast.get_output(0)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    output_width, input_width = weight.shape
    rank = len(tuple(tensor.shape))
    rhs_shape = (1,) * max(0, rank - 2) + (input_width, output_width)
    rhs = graph_ops.add_constant(
        network,
        rhs_shape,
        weight.T.reshape(rhs_shape),
        precision=precision,
    )
    layer = network.add_matrix_multiply(
        tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to create SAM2 HOI linear {label!r}")
    # TensorRT 11.x may infer an FP32 accumulator output for a BF16 GEMM even
    # in a strongly typed graph.  PyTorch autocast returns BF16 from Linear,
    # so close that precision boundary before the learned BF16 bias addition.
    result = layer.get_output(0)
    if precision == "bf16":
        cast_layer = network.add_cast(result, trt.bfloat16)
        if cast_layer is None:
            raise RuntimeError(f"TensorRT failed to cast SAM2 HOI linear {label!r}")
        result = cast_layer.get_output(0)
    else:
        result = graph_ops.cast(network, result, tensor.dtype)
    bias_shape = (1,) * max(0, rank - 1) + (output_width,)
    bias_tensor = graph_ops.add_constant(
        network,
        bias_shape,
        bias.reshape(bias_shape),
        precision=precision,
    )
    return _typed_elementwise(
        network,
        result,
        bias_tensor,
        trt.ElementWiseOperation.SUM,
        graph_ops.runtime_dtype(precision),
    )


def _linear_via_fused_1x1_conv(
    network,
    tensor,
    weights,
    prefix: str,
    *,
    precision: str,
):
    """Use source-equivalent fused-bias BF16 arithmetic at proven Linear sites."""

    if prefix not in _FUSED_BF16_LINEAR_PREFIXES:
        raise ValueError(f"SAM2 HOI fused BF16 Linear is not qualified for {prefix!r}")
    if precision != "bf16":
        return _linear(network, tensor, weights, prefix, precision=precision)

    trt = trt_compat.get_trt()
    weight = np.asarray(weights[prefix + ".weight"], dtype=np.float32)
    bias = np.asarray(weights[prefix + ".bias"], dtype=np.float32)
    output_width, input_width = weight.shape
    shape = tuple(tensor.shape)
    if len(shape) != 3 or int(shape[0]) != 1 or int(shape[2]) != input_width:
        raise RuntimeError(
            f"SAM2 HOI fused BF16 Linear {prefix!r} expected [1, L, {input_width}], got {shape}"
        )
    length = int(shape[1])
    tensor = _force_cast(network, tensor, trt.bfloat16)
    channels_first = _transpose(network, tensor, (0, 2, 1))
    image = _reshape(network, channels_first, (1, input_width, length, 1))
    layer = network.add_convolution_nd(
        image,
        output_width,
        (1, 1),
        trt.Weights(),
        trt.Weights(),
    )
    if layer is None:
        raise RuntimeError(f"TensorRT failed to create SAM2 HOI fused BF16 Linear {prefix!r}")
    layer.name = prefix + ".fused_1x1_conv"
    weight_tensor = graph_ops.add_constant(
        network,
        (output_width, input_width, 1, 1),
        weight.reshape(output_width, input_width, 1, 1),
        precision="bf16",
    )
    bias_tensor = graph_ops.add_constant(
        network,
        (output_width,),
        bias,
        precision="bf16",
    )
    layer.set_input(1, weight_tensor)
    layer.set_input(2, bias_tensor)
    result = _reshape(network, layer.get_output(0), (1, output_width, length))
    result = _transpose(network, result, (0, 2, 1))
    return _force_cast(network, result, trt.bfloat16)


def _plugin(
    network,
    name: str,
    inputs,
    *,
    instance_name: str,
    output_dtype,
    input_dtypes=None,
):
    if input_dtypes is None:
        input_dtypes = (output_dtype,) * len(inputs)
    if len(input_dtypes) != len(inputs):
        raise ValueError(f"SAM2 HOI plugin {name!r} has an invalid type contract")
    typed_inputs = []
    for tensor, dtype in zip(inputs, input_dtypes):
        typed_inputs.append(_force_cast(network, tensor, dtype))
    output = graph_ops.add_plugin(network, name, typed_inputs, instance_name=instance_name)
    output.name = instance_name + ".raw_output"
    # IPluginV2DynamicExt exposes its precise type through configure-time
    # descriptors.  TensorRT 11.x initially labels its network output FP32;
    # make the model's semantic output type explicit for strongly typed users.
    cast_layer = network.add_cast(output, output_dtype)
    if cast_layer is None:
        raise RuntimeError(f"TensorRT failed to type SAM2 HOI plugin {name!r}")
    return cast_layer.get_output(0)


def _layer_norm(network, tensor, weights, prefix: str, *, precision: str, name: str):
    gamma = np.asarray(weights[prefix + ".weight"], dtype=np.float32)
    beta = np.asarray(weights[prefix + ".bias"], dtype=np.float32)
    if precision == "bf16":
        trt = trt_compat.get_trt()
        gamma_tensor = graph_ops.add_constant(network, (_HIDDEN,), gamma, precision="fp32")
        beta_tensor = graph_ops.add_constant(network, (_HIDDEN,), beta, precision="fp32")
        return _plugin(
            network,
            "Sam2HoiLayerNorm256",
            (tensor, gamma_tensor, beta_tensor),
            instance_name=name,
            output_dtype=trt.float32,
            input_dtypes=(tensor.dtype, trt.float32, trt.float32),
        )
    return graph_ops.add_layer_norm(
        network,
        tensor,
        gamma,
        beta,
        epsilon=1.0e-5,
        precision=precision,
    )


def _softmax(network, tensor, *, precision: str, name: str):
    if precision == "bf16":
        return _plugin(
            network,
            "Sam2HoiSoftmax",
            (tensor,),
            instance_name=name,
            output_dtype=graph_ops.runtime_dtype(precision),
        )
    layer = network.add_softmax(tensor)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI softmax")
    layer.axes = 1 << (len(tuple(tensor.shape)) - 1)
    return layer.get_output(0)


def _sigmoid(network, tensor, *, precision: str, name: str):
    trt = trt_compat.get_trt()
    if precision == "bf16":
        return _plugin(
            network,
            "Sam2HoiSigmoid",
            (tensor,),
            instance_name=name,
            output_dtype=tensor.dtype,
        )
    layer = network.add_activation(tensor, trt.ActivationType.SIGMOID)
    if layer is None:
        raise RuntimeError("TensorRT failed to create a SAM2 HOI sigmoid")
    return layer.get_output(0)


def _mha_scale(network, tensor, *, precision: str, name: str):
    trt = trt_compat.get_trt()
    if precision == "bf16":
        return _plugin(
            network,
            "Sam2HoiMhaScale",
            (tensor,),
            instance_name=name,
            output_dtype=graph_ops.runtime_dtype(precision),
        )
    scale = graph_ops.add_constant(
        network,
        (1,) * len(tuple(tensor.shape)),
        np.asarray([_HEAD_WIDTH**-0.5], dtype=np.float32),
        precision="fp32",
    )
    return _elementwise(network, tensor, scale, trt.ElementWiseOperation.PROD)


def _regression_branch(network, tensor, weights, branch: int, *, precision: str):
    prefix = _PREFIX + f"reg_branches.{branch}"
    linear = _linear_via_fused_1x1_conv if branch == _NUM_DECODER_LAYERS else _linear
    hidden = linear(network, tensor, weights, prefix + ".0", precision=precision)
    hidden = graph_ops.add_activation(network, hidden, "relu")
    hidden = linear(network, hidden, weights, prefix + ".2", precision=precision)
    hidden = graph_ops.add_activation(network, hidden, "relu")
    return _linear(network, hidden, weights, prefix + ".4", precision=precision)


def _feed_forward(network, tensor, weights, prefix: str, *, precision: str):
    trt = trt_compat.get_trt()
    hidden = _linear(network, tensor, weights, prefix + ".layers.0.0", precision=precision)
    hidden = graph_ops.add_activation(network, hidden, "relu")
    hidden = _linear(network, hidden, weights, prefix + ".layers.1", precision=precision)
    return _typed_elementwise(network, tensor, hidden, trt.ElementWiseOperation.SUM, tensor.dtype)


def _msda(
    network,
    query,
    value,
    reference_points,
    weights,
    prefix: str,
    *,
    four_dimensional_references: bool,
    precision: str,
    name: str,
):
    trt = trt_compat.get_trt()
    work_dtype = graph_ops.runtime_dtype(precision)
    projected_value = _linear(network, value, weights, prefix + ".value_proj", precision=precision)
    projected_value = _reshape(
        network, projected_value, (1, _SPATIAL_TOKENS, _NUM_HEADS, _HEAD_WIDTH)
    )
    offsets = _linear(network, query, weights, prefix + ".sampling_offsets", precision=precision)
    offsets = _reshape(
        network,
        offsets,
        (1, int(query.shape[1]), _NUM_HEADS, _NUM_LEVELS, _NUM_POINTS, 2),
    )
    attention = _linear(network, query, weights, prefix + ".attention_weights", precision=precision)
    attention = _reshape(
        network,
        attention,
        (1, int(query.shape[1]), _NUM_HEADS, _NUM_LEVELS * _NUM_POINTS),
    )
    attention = _softmax(network, attention, precision=precision, name=name + ".softmax")
    attention = _reshape(
        network,
        attention,
        (1, int(query.shape[1]), _NUM_HEADS, _NUM_LEVELS, _NUM_POINTS),
    )

    if four_dimensional_references:
        xy = _slice(
            network,
            reference_points,
            (0, 0, 0, 0),
            (1, int(query.shape[1]), _NUM_LEVELS, 2),
            (1, 1, 1, 1),
        )
        wh = _slice(
            network,
            reference_points,
            (0, 0, 0, 2),
            (1, int(query.shape[1]), _NUM_LEVELS, 2),
            (1, 1, 1, 1),
        )
        xy = _reshape(network, xy, (1, int(query.shape[1]), 1, _NUM_LEVELS, 1, 2))
        wh = _reshape(network, wh, (1, int(query.shape[1]), 1, _NUM_LEVELS, 1, 2))
        point_count = graph_ops.add_constant(
            network, (1, 1, 1, 1, 1, 1), [_NUM_POINTS], precision=precision
        )
        half = graph_ops.add_constant(network, (1, 1, 1, 1, 1, 1), [0.5], precision="fp32")
        delta = _typed_elementwise(
            network, offsets, point_count, trt.ElementWiseOperation.DIV, work_dtype
        )
        delta = _typed_elementwise(network, delta, wh, trt.ElementWiseOperation.PROD, trt.float32)
        delta = _typed_elementwise(network, delta, half, trt.ElementWiseOperation.PROD, trt.float32)
        locations = _typed_elementwise(
            network, xy, delta, trt.ElementWiseOperation.SUM, trt.float32
        )
    else:
        base = _reshape(
            network,
            reference_points,
            (1, int(query.shape[1]), 1, _NUM_LEVELS, 1, 2),
        )
        normalizer = np.asarray(
            [[width, height] for height, width in _LEVEL_SHAPES], dtype=np.float32
        ).reshape(1, 1, 1, _NUM_LEVELS, 1, 2)
        normalizer_tensor = graph_ops.add_constant(
            network, tuple(normalizer.shape), normalizer, precision=precision
        )
        delta = _typed_elementwise(
            network, offsets, normalizer_tensor, trt.ElementWiseOperation.DIV, work_dtype
        )
        locations = _typed_elementwise(
            network, base, delta, trt.ElementWiseOperation.SUM, work_dtype
        )

    sampled = _plugin(
        network,
        "Sam2HoiMsDeformAttn",
        (projected_value, locations, attention),
        instance_name=name + ".msda",
        output_dtype=work_dtype,
    )
    output = _linear_via_fused_1x1_conv(
        network,
        sampled,
        weights,
        prefix + ".output_proj",
        precision=precision,
    )
    return output


def _encoder(
    network,
    features,
    weights,
    *,
    precision: str,
):
    trt = trt_compat.get_trt()
    rows = []
    positions = []
    level_embeds = np.asarray(weights[_TRANSFORMER + "level_embeds"], dtype=np.float32)
    for level, (feature, (height, width)) in enumerate(zip(features, _LEVEL_SHAPES)):
        row_major = _transpose(network, feature, (0, 2, 3, 1))
        rows.append(_reshape(network, row_major, (1, height * width, _HIDDEN)))
        position = _sine_position_encoding_tensor(network, height, width)
        level_embed = graph_ops.add_constant(
            network,
            (1, 1, _HIDDEN),
            level_embeds[level].reshape(1, 1, _HIDDEN),
            precision="fp32",
        )
        position = _typed_elementwise(
            network, position, level_embed, trt.ElementWiseOperation.SUM, trt.float32
        )
        positions.append(position)
    hidden = _concat(network, rows, axis=1)
    position = _concat(network, positions, axis=1)
    references = graph_ops.add_constant(
        network,
        (1, _SPATIAL_TOKENS, _NUM_LEVELS, 2),
        _encoder_reference_points(),
        precision=precision,
    )

    for layer in range(_NUM_ENCODER_LAYERS):
        base = _TRANSFORMER + f"encoder.layers.{layer}"
        query = _typed_elementwise(
            network, hidden, position, trt.ElementWiseOperation.SUM, trt.float32
        )
        attention = _msda(
            network,
            query,
            hidden,
            references,
            weights,
            base + ".attentions.0",
            four_dimensional_references=False,
            precision=precision,
            name=f"encoder.{layer}.attention",
        )
        hidden = _typed_elementwise(
            network, hidden, attention, trt.ElementWiseOperation.SUM, hidden.dtype
        )
        hidden = _layer_norm(
            network,
            hidden,
            weights,
            base + ".norms.0",
            precision=precision,
            name=f"encoder.{layer}.norm0",
        )
        hidden = _feed_forward(network, hidden, weights, base + ".ffns.0", precision=precision)
        hidden = _layer_norm(
            network,
            hidden,
            weights,
            base + ".norms.1",
            precision=precision,
            name=f"encoder.{layer}.norm1",
        )
    return hidden


def _inverse_sigmoid(network, tensor, *, epsilon: float):
    trt = trt_compat.get_trt()
    clipped = _clip(network, tensor, 0.0, 1.0)
    low = _clip(network, clipped, epsilon, 1.0)
    one = graph_ops.add_constant(network, (1,) * len(tuple(tensor.shape)), [1.0], precision="fp32")
    inverse = _elementwise(network, one, clipped, trt.ElementWiseOperation.SUB)
    inverse = _clip(network, inverse, epsilon, 1.0)
    ratio = _elementwise(network, low, inverse, trt.ElementWiseOperation.DIV)
    return _unary(network, ratio, trt.UnaryOperation.LOG)


def _interleaved_sine_cosine(network, phase, *, query_count: int):
    trt = trt_compat.get_trt()
    even = _slice(network, phase, (0, 0, 0), (1, query_count, 64), (1, 1, 2))
    odd = _slice(network, phase, (0, 0, 1), (1, query_count, 64), (1, 1, 2))
    sine = _unary(network, even, trt.UnaryOperation.SIN)
    cosine = _unary(network, odd, trt.UnaryOperation.COS)
    sine = _reshape(network, sine, (1, query_count, 64, 1))
    cosine = _reshape(network, cosine, (1, query_count, 64, 1))
    return _reshape(network, _concat(network, (sine, cosine), axis=3), (1, query_count, 128))


def _reference_query_position(network, references, weights, *, precision: str):
    trt = trt_compat.get_trt()
    query_count = int(references.shape[1])
    scale = graph_ops.add_constant(network, (1, 1, 1), [2.0 * np.pi], precision="fp32")
    scaled = _elementwise(network, references, scale, trt.ElementWiseOperation.PROD)
    indices = np.arange(128, dtype=np.float32)
    dim_t = np.float32(10000.0) ** (2.0 * np.floor(indices / 2.0) / 128.0)
    denominator = graph_ops.add_constant(
        network, (1, 1, 128), dim_t.reshape(1, 1, 128), precision="fp32"
    )
    encoded = []
    # DINO concatenates y, x, width, height.
    for coordinate in (1, 0, 2, 3):
        value = _slice(
            network,
            scaled,
            (0, 0, coordinate),
            (1, query_count, 1),
            (1, 1, 1),
        )
        phase = _elementwise(network, value, denominator, trt.ElementWiseOperation.DIV)
        encoded.append(_interleaved_sine_cosine(network, phase, query_count=query_count))
    position = _concat(network, encoded, axis=2)
    position = graph_ops.cast(network, position, graph_ops.runtime_dtype(precision))
    position = _linear(
        network,
        position,
        weights,
        _TRANSFORMER + "decoder.ref_point_head.0",
        precision=precision,
    )
    position = graph_ops.add_activation(network, position, "relu")
    return _linear(
        network,
        position,
        weights,
        _TRANSFORMER + "decoder.ref_point_head.2",
        precision=precision,
    )


def _self_attention(
    network,
    hidden,
    position,
    weights,
    prefix: str,
    *,
    precision: str,
    name: str,
):
    trt = trt_compat.get_trt()
    query_key = _typed_elementwise(
        network, hidden, position, trt.ElementWiseOperation.SUM, hidden.dtype
    )
    projection_weight = np.asarray(weights[prefix + ".in_proj_weight"], dtype=np.float32)
    projection_bias = np.asarray(weights[prefix + ".in_proj_bias"], dtype=np.float32)
    projected = []
    for index, source in enumerate((query_key, query_key, hidden)):
        projected.append(
            _linear_arrays(
                network,
                source,
                projection_weight[index * _HIDDEN : (index + 1) * _HIDDEN],
                projection_bias[index * _HIDDEN : (index + 1) * _HIDDEN],
                precision=precision,
                label=f"{prefix}.in_proj.{index}",
            )
        )
    heads = []
    for tensor in projected:
        tensor = _reshape(network, tensor, (1, _NUM_QUERIES, _NUM_HEADS, _HEAD_WIDTH))
        tensor = _transpose(network, tensor, (0, 2, 1, 3))
        heads.append(_reshape(network, tensor, (_NUM_HEADS, _NUM_QUERIES, _HEAD_WIDTH)))
    query = _mha_scale(network, heads[0], precision=precision, name=name + ".scale")
    logits_layer = network.add_matrix_multiply(
        query,
        trt.MatrixOperation.NONE,
        heads[1],
        trt.MatrixOperation.TRANSPOSE,
    )
    if logits_layer is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI self-attention logits")
    logits = logits_layer.get_output(0)
    probabilities = _softmax(network, logits, precision=precision, name=name + ".softmax")
    values_layer = network.add_matrix_multiply(
        probabilities,
        trt.MatrixOperation.NONE,
        heads[2],
        trt.MatrixOperation.NONE,
    )
    if values_layer is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI self-attention values")
    weighted_values = _force_cast(
        network, values_layer.get_output(0), graph_ops.runtime_dtype(precision)
    )
    merged = _transpose(network, weighted_values, (1, 0, 2))
    merged = _reshape(network, merged, (1, _NUM_QUERIES, _HIDDEN))
    output = _linear(network, merged, weights, prefix + ".out_proj", precision=precision)
    residual = _typed_elementwise(
        network, hidden, output, trt.ElementWiseOperation.SUM, hidden.dtype
    )
    return residual


def _decoder(
    network,
    memory,
    references,
    weights,
    *,
    precision: str,
):
    trt = trt_compat.get_trt()
    query = graph_ops.add_constant(
        network,
        (1, _NUM_QUERIES, _HIDDEN),
        np.asarray(weights[_TRANSFORMER + "query_embed.weight"], dtype=np.float32).reshape(
            1, _NUM_QUERIES, _HIDDEN
        ),
        # The learned content query is expanded outside any autocast-enabled
        # operator in the source decoder and therefore remains FP32.
        precision="fp32",
    )
    final_reference = references
    for layer in range(_NUM_DECODER_LAYERS):
        base = _TRANSFORMER + f"decoder.layers.{layer}"
        position = _reference_query_position(network, references, weights, precision=precision)
        query = _self_attention(
            network,
            query,
            position,
            weights,
            base + ".attentions.0.attn",
            precision=precision,
            name=f"decoder.{layer}.self_attention",
        )
        query = _layer_norm(
            network,
            query,
            weights,
            base + ".norms.0",
            precision=precision,
            name=f"decoder.{layer}.norm0",
        )

        repeated_references = _reshape(network, references, (1, _NUM_QUERIES, 1, 4))
        repeated_references = _concat(network, (repeated_references,) * _NUM_LEVELS, axis=2)
        cross_query = _typed_elementwise(
            network, query, position, trt.ElementWiseOperation.SUM, query.dtype
        )
        cross = _msda(
            network,
            cross_query,
            memory,
            repeated_references,
            weights,
            base + ".attentions.1",
            four_dimensional_references=True,
            precision=precision,
            name=f"decoder.{layer}.cross_attention",
        )
        query = _typed_elementwise(network, query, cross, trt.ElementWiseOperation.SUM, query.dtype)
        query = _layer_norm(
            network,
            query,
            weights,
            base + ".norms.1",
            precision=precision,
            name=f"decoder.{layer}.norm1",
        )
        query = _feed_forward(network, query, weights, base + ".ffns.0", precision=precision)
        query = _layer_norm(
            network,
            query,
            weights,
            base + ".norms.2",
            precision=precision,
            name=f"decoder.{layer}.norm2",
        )
        final_reference = references
        if layer + 1 < _NUM_DECODER_LAYERS:
            delta = _regression_branch(network, query, weights, layer, precision=precision)
            delta = graph_ops.cast(network, delta, trt.float32)
            reference_logits = _inverse_sigmoid(network, references, epsilon=1.0e-3)
            refined = _elementwise(network, delta, reference_logits, trt.ElementWiseOperation.SUM)
            references = _sigmoid(
                network,
                refined,
                precision=precision,
                name=f"decoder.{layer}.reference_sigmoid",
            )

    query = _layer_norm(
        network,
        query,
        weights,
        _TRANSFORMER + "decoder.norm",
        precision=precision,
        name="decoder.final_norm",
    )
    return query, final_reference


def _encoder_proposal_logits_tensor(network, normalized, valid, dependency):
    """Reproduce the source CUDA logit transform for fixed proposals."""

    trt = trt_compat.get_trt()
    proposals = graph_ops.add_constant(
        network, (1, _SPATIAL_TOKENS, 4), normalized, precision="fp32"
    )
    dependency = _slice(network, dependency, (0, 0, 0), (1, 1, 1), (1, 1, 1))
    dependency = graph_ops.cast(network, dependency, trt.float32)
    runtime_zero = graph_ops.add_constant(network, (1, 1, 1), [0.0], precision="fp32")
    runtime_zero = _elementwise(network, dependency, runtime_zero, trt.ElementWiseOperation.PROD)
    proposals = _elementwise(network, proposals, runtime_zero, trt.ElementWiseOperation.SUM)
    one = graph_ops.add_constant(network, (1, 1, 1), [1.0], precision="fp32")
    inverse = _elementwise(network, one, proposals, trt.ElementWiseOperation.SUB)
    ratio = _elementwise(network, proposals, inverse, trt.ElementWiseOperation.DIV)
    logits = _unary(network, ratio, trt.UnaryOperation.LOG)

    valid_tensor = graph_ops.add_constant(network, (1, _SPATIAL_TOKENS, 1), valid, precision="fp32")
    zero = graph_ops.add_constant(network, (1, 1, 1), [0.0], precision="fp32")
    condition = _elementwise(network, valid_tensor, zero, trt.ElementWiseOperation.GREATER)
    condition = _concat(network, (condition,) * 4, axis=2)
    infinity = graph_ops.add_constant(network, (1, 1, 1), [np.inf], precision="fp32")
    select = network.add_select(condition, logits, infinity)
    if select is None:
        raise RuntimeError("TensorRT failed to mask SAM2 HOI encoder proposals")
    return select.get_output(0), valid_tensor


def _initial_references(network, memory, weights, *, precision: str):
    trt = trt_compat.get_trt()
    normalized, valid = _encoder_proposal_coordinates()
    proposal_tensor, valid_tensor = _encoder_proposal_logits_tensor(
        network, normalized, valid, memory
    )
    output_memory = _typed_elementwise(
        network, memory, valid_tensor, trt.ElementWiseOperation.PROD, trt.float32
    )
    output_memory = _linear_via_fused_1x1_conv(
        network,
        output_memory,
        weights,
        _TRANSFORMER + "enc_output",
        precision=precision,
    )
    output_memory = _layer_norm(
        network,
        output_memory,
        weights,
        _TRANSFORMER + "enc_output_norm",
        precision=precision,
        name="encoder.output_norm",
    )
    cls = _linear(
        network,
        output_memory,
        weights,
        _PREFIX + f"cls_branches.{_NUM_DECODER_LAYERS}",
        precision=precision,
    )
    reg = _regression_branch(
        network,
        output_memory,
        weights,
        _NUM_DECODER_LAYERS,
        precision=precision,
    )
    reg = graph_ops.cast(network, reg, trt.float32)
    coordinates = _elementwise(network, reg, proposal_tensor, trt.ElementWiseOperation.SUM)

    maximum = network.add_reduce(cls, trt.ReduceOperation.MAX, 1 << 2, keep_dims=False)
    if maximum is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI proposal class reduction")
    topk = network.add_topk(maximum.get_output(0), trt.TopKOperation.MAX, _NUM_QUERIES, 1 << 1)
    if topk is None:
        raise RuntimeError("TensorRT failed to create SAM2 HOI top-1500 proposals")
    indices = topk.get_output(1)
    gather = network.add_gather(coordinates, indices, axis=1)
    if gather is None:
        raise RuntimeError("TensorRT failed to gather SAM2 HOI proposal coordinates")
    gather.num_elementwise_dims = 1
    gathered = gather.get_output(0)
    references = _sigmoid(
        network,
        gathered,
        precision=precision,
        name="encoder.proposal_sigmoid",
    )
    return references


def _build_outputs(network, query, reference, weights, *, precision: str):
    trt = trt_compat.get_trt()
    class_logits = _linear(
        network,
        query,
        weights,
        _PREFIX + f"cls_branches.{_NUM_DECODER_LAYERS - 1}",
        precision=precision,
    )
    class_scores = _sigmoid(network, class_logits, precision=precision, name="output.class_sigmoid")
    box_delta = _regression_branch(
        network,
        query,
        weights,
        _NUM_DECODER_LAYERS - 1,
        precision=precision,
    )
    box_delta = graph_ops.cast(network, box_delta, trt.float32)
    reference_logits = _inverse_sigmoid(network, reference, epsilon=1.0e-3)
    box_logits = _elementwise(network, box_delta, reference_logits, trt.ElementWiseOperation.SUM)
    boxes = _sigmoid(network, box_logits, precision=precision, name="output.box_sigmoid")
    graph_ops.mark_output(network, class_scores, "class_scores", dtype=trt.float32)
    graph_ops.mark_output(network, boxes, "boxes_cxcywh", dtype=trt.float32)
    graph_ops.mark_output(network, query, "query_embeddings", dtype=trt.float32)


def build_hoi_detector_engine(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build the exact fixed-shape raw HOI detector with TensorRT APIs."""

    precision = graph_ops.normalize_precision(precision)
    _validate_weights(weights)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    graph_ops.reset_weight_refs()
    try:
        network = builder.create_network(
            trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
        )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
        config.builder_optimization_level = _BUILDER_OPTIMIZATION_LEVEL
        config.avg_timing_iterations = _AVG_TIMING_ITERATIONS
        if _MAX_AUX_STREAMS >= 0:
            config.max_aux_streams = _MAX_AUX_STREAMS
        dtype = graph_ops.runtime_dtype(precision)
        features = []
        for level, (height, width) in enumerate(_LEVEL_SHAPES):
            feature = network.add_input(
                f"detector_feature_{level}", dtype, (1, _HIDDEN, height, width)
            )
            if feature is None:
                raise RuntimeError(f"Could not add SAM2 HOI detector input {level}")
            features.append(feature)
        memory = _encoder(network, features, weights, precision=precision)
        references = _initial_references(network, memory, weights, precision=precision)
        query, final_reference = _decoder(network, memory, references, weights, precision=precision)
        _build_outputs(network, query, final_reference, weights, precision=precision)
        if verbose:
            print(
                f"[trtmc build] Building direct TensorRT SAM2-HOI {precision} detector "
                f"from {network.num_layers} layers ...",
                file=sys.stderr,
            )
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                if layer.type != trt.LayerType.PLUGIN_V2:
                    continue
                input_types = [str(layer.get_input(i).dtype) for i in range(layer.num_inputs)]
                output_types = [str(layer.get_output(i).dtype) for i in range(layer.num_outputs)]
                print(
                    f"[trtmc build] plugin layer {index}: {input_types} -> {output_types}",
                    file=sys.stderr,
                )
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT failed to build the SAM2 HOI detector plan")
        payload = bytes(plan)
        if not payload:
            raise RuntimeError("TensorRT produced an empty SAM2 HOI detector plan")
        return payload
    finally:
        graph_ops.reset_weight_refs()


__all__ = ["build_hoi_detector_engine", "detector_required_weight_keys"]
