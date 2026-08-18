# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT construction of the reviewed SAM2.1 HOI tracker stages.

Prompt encoding, the two-way mask decoder, recurrent memory attention, and
spatial-memory encoding are rebuilt from checkpoint arrays with TensorRT
Network Definition layers.  The C++ runtime continues to own object
association and temporal-memory policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import sys
from typing import Any

import numpy as np

from . import native_graph_ops as graph_ops


PROMPT_TRACKER_SECTION = "sam2_hoi_prompt_tracker_engine_plan"
RECURRENT_TRACKER_SECTION = "sam2_hoi_recurrent_tracker_engine_plan"
MEMORY_ENCODER_SECTION = "sam2_hoi_memory_encoder_engine_plan"

_BATCH = 2
_IMAGE_SIZE = 1024
_GRID = 64
_TOKENS = _GRID * _GRID
_LOW_RESOLUTION = 256
_HIDDEN = 256
_MEMORY_CHANNELS = 64
_POINTER_TOKENS = _HIDDEN // _MEMORY_CHANNELS
_MEMORY_FRAMES = 7
_MAX_POINTERS = 16
_DECODER_HEADS = 8
_MASK_TOKENS = 4
_OUTPUT_TOKENS = 2 + _MASK_TOKENS
_WORKSPACE_BYTES = 8 << 30
_SUPPORTED_PRECISIONS = frozenset({"fp32", "bf16"})


@dataclass(frozen=True)
class NativeBinding:
    """One runtime binding in a direct TensorRT tracker plan."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    profile: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None


@dataclass(frozen=True)
class NativeEngineBindings:
    """Inputs and outputs used to verify the native builder ABI."""

    section: str
    inputs: tuple[NativeBinding, ...]
    outputs: tuple[NativeBinding, ...]


def _normalize_precision(precision: str) -> str:
    normalized = str(precision).strip().lower()
    if normalized not in _SUPPORTED_PRECISIONS:
        supported = ", ".join(sorted(_SUPPORTED_PRECISIONS))
        raise ValueError(f"SAM2 HOI native tracker supports {{{supported}}}, got {precision!r}")
    return normalized


def tracker_binding_specs(precision: str = "fp32") -> tuple[NativeEngineBindings, ...]:
    """Return the fail-closed public ABI for all three tracker plans."""

    work = "bfloat16" if _normalize_precision(precision) == "bf16" else "float32"
    feature_inputs = (
        NativeBinding("tracker_feature_0", work, (1, 32, 256, 256)),
        NativeBinding("tracker_feature_1", work, (1, 64, 128, 128)),
        NativeBinding("tracker_feature_2", "float32", (1, 256, 64, 64)),
    )
    head_outputs = (
        NativeBinding("pred_masks", "float32", (2, 1, 256, 256)),
        NativeBinding("object_pointer", "float32", (2, 256)),
        NativeBinding("object_score_logits", "float32", (2, 1)),
    )
    memory_profile = (
        (2, 1, 64, 64, 64),
        (2, 3, 64, 64, 64),
        (2, 7, 64, 64, 64),
    )
    memory_offset_profile = ((2, 1), (2, 3), (2, 7))
    pointer_profile = ((2, 1, 256), (2, 2, 256), (2, 16, 256))
    pointer_offset_profile = ((2, 1), (2, 2), (2, 16))
    prompt = NativeEngineBindings(
        PROMPT_TRACKER_SECTION,
        feature_inputs
        + (
            NativeBinding("point_coords", "float32", (2, 3, 2)),
            NativeBinding("point_labels", "int32", (2, 3)),
        ),
        head_outputs + (NativeBinding("selected_iou", "float32", (2, 1)),),
    )
    recurrent = NativeEngineBindings(
        RECURRENT_TRACKER_SECTION,
        feature_inputs
        + (
            NativeBinding("tracker_position_2", "float32", (1, 256, 64, 64)),
            NativeBinding("memory_features", work, (2, -1, 64, 64, 64), memory_profile),
            NativeBinding("memory_position", work, (2, -1, 64, 64, 64), memory_profile),
            NativeBinding(
                "memory_temporal_offsets",
                "int32",
                (2, -1),
                memory_offset_profile,
            ),
            NativeBinding("object_pointers", "float32", (2, -1, 256), pointer_profile),
            NativeBinding(
                "object_pointer_temporal_offsets",
                "float32",
                (2, -1),
                pointer_offset_profile,
            ),
            NativeBinding("object_pointer_time_denominator", "float32", (1,)),
        ),
        head_outputs + (NativeBinding("selected_iou", "float32", (2, 3)),),
    )
    memory_dtype = work
    memory = NativeEngineBindings(
        MEMORY_ENCODER_SECTION,
        (
            NativeBinding("tracker_feature_2", "float32", (1, 256, 64, 64)),
            NativeBinding("pred_masks", "float32", (2, 1, 256, 256)),
            NativeBinding("object_score_logits", "float32", (2, 1)),
            NativeBinding("is_mask_from_points", "int32", (2, 1)),
        ),
        (
            NativeBinding("new_memory_features", memory_dtype, (2, 64, 64, 64)),
            NativeBinding("new_memory_position", memory_dtype, (2, 64, 64, 64)),
        ),
    )
    return prompt, recurrent, memory


def _trt():
    from tensorrt_model_connect import trt_compat

    return trt_compat.get_trt()


def _new_network(*, verbose: bool):
    from tensorrt_model_connect import trt_compat

    trt = _trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _WORKSPACE_BYTES)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 5
    if hasattr(config, "avg_timing_iterations"):
        config.avg_timing_iterations = 8
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    return trt, builder, network, config


def _trt_dtype(trt: Any, name: str):
    mapping = {"float32": trt.float32, "bfloat16": trt.bfloat16, "int32": trt.int32}
    return mapping[name]


def _input(network: Any, binding: NativeBinding):
    tensor = network.add_input(binding.name, _trt_dtype(_trt(), binding.dtype), binding.shape)
    if tensor is None:
        raise RuntimeError(f"Could not add SAM2 HOI native tracker input {binding.name!r}")
    return tensor


def _mark(network: Any, tensor: Any, binding: NativeBinding) -> None:
    tensor = _cast(network, tensor, _trt_dtype(_trt(), binding.dtype))
    tensor.name = binding.name
    network.mark_output(tensor)


def _add_profiles(builder: Any, config: Any, bindings: NativeEngineBindings) -> None:
    dynamic = [binding for binding in bindings.inputs if binding.profile is not None]
    if not dynamic:
        return
    profile = builder.create_optimization_profile()
    for binding in dynamic:
        assert binding.profile is not None
        if profile.set_shape(binding.name, *binding.profile) is False:
            raise RuntimeError(f"Could not set SAM2 HOI profile for {binding.name!r}")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError(f"Could not add TensorRT profile for {bindings.section}")


def _serialize(builder: Any, network: Any, config: Any, section: str, *, verbose: bool) -> bytes:
    if verbose:
        print(
            f"[trtmc build] Building direct TensorRT {section} from {network.num_layers} layers ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build direct tracker plan {section}")
    payload = bytes(plan)
    if not payload:
        raise RuntimeError(f"TensorRT produced an empty direct tracker plan {section}")
    return payload


def _constant(network: Any, values: Any, shape: tuple[int, ...] | None = None, *, dtype=np.float32):
    trt = _trt()
    array = np.ascontiguousarray(values, dtype=dtype)
    if shape is None:
        shape = tuple(int(dim) for dim in array.shape)
    array = np.ascontiguousarray(array.reshape(shape), dtype=dtype)
    return network.add_constant(shape, trt.Weights(array)).get_output(0)


def _cast(network: Any, tensor: Any, dtype: Any):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _constant_for_dtype(network: Any, values: Any, shape: tuple[int, ...], dtype: Any):
    constant = _constant(network, values, shape)
    return _cast(network, constant, dtype)


def _reshape(network: Any, tensor: Any, shape: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    layer.reshape_dims = shape
    return layer.get_output(0)


def _transpose(network: Any, tensor: Any, order: tuple[int, ...]):
    layer = network.add_shuffle(tensor)
    layer.first_transpose = _trt().Permutation(order)
    return layer.get_output(0)


def _expand_batch2(network: Any, tensor: Any):
    if int(tensor.shape[0]) == _BATCH:
        return tensor
    if int(tensor.shape[0]) != 1:
        raise ValueError(f"SAM2 HOI tracker cannot expand feature batch {tensor.shape[0]} to two")
    concat = network.add_concatenation([tensor, tensor])
    concat.axis = 0
    return concat.get_output(0)


def _fp32_sum(network: Any, lhs: Any, rhs: Any):
    trt = _trt()
    return network.add_elementwise(
        _cast(network, lhs, trt.float32),
        _cast(network, rhs, trt.float32),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _sum_as(network: Any, lhs: Any, rhs: Any, dtype: Any):
    trt = _trt()
    return network.add_elementwise(
        _cast(network, lhs, dtype),
        _cast(network, rhs, dtype),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _weight(weights: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.asarray(weights[key])
    except KeyError as error:
        raise KeyError(
            f"SAM2 HOI tracker checkpoint is missing required parameter {key}"
        ) from error
    if tuple(value.shape) != shape:
        raise ValueError(f"SAM2 HOI tracker weight {key} has shape {value.shape}, expected {shape}")
    return np.ascontiguousarray(value, dtype=np.float32)


_ARCHITECTURE_WEIGHT_SHAPES = (
    ("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", (2, 128)),
    ("sam_prompt_encoder.point_embeddings.3.weight", (1, _HIDDEN)),
    ("sam_mask_decoder.mask_tokens.weight", (_MASK_TOKENS, _HIDDEN)),
    (
        "sam_mask_decoder.transformer.layers.1.self_attn.q_proj.weight",
        (_HIDDEN, _HIDDEN),
    ),
    ("sam_mask_decoder.iou_prediction_head.layers.2.weight", (_MASK_TOKENS, _HIDDEN)),
    ("sam_mask_decoder.output_upscaling.3.weight", (64, 32, 2, 2)),
    (
        "memory_attention.layers.3.cross_attn_image.k_proj.weight",
        (_HIDDEN, _MEMORY_CHANNELS),
    ),
    ("memory_attention.layers.3.linear1.weight", (2048, _HIDDEN)),
    ("memory_attention.norm.weight", (_HIDDEN,)),
    ("maskmem_tpos_enc", (_MEMORY_FRAMES, 1, 1, _MEMORY_CHANNELS)),
    ("obj_ptr_tpos_proj.weight", (_MEMORY_CHANNELS, _HIDDEN)),
    ("memory_encoder.mask_downsampler.encoder.12.weight", (_HIDDEN, _HIDDEN, 1, 1)),
    ("memory_encoder.fuser.layers.1.dwconv.weight", (_HIDDEN, 1, 7, 7)),
    ("memory_encoder.out_proj.weight", (_MEMORY_CHANNELS, _HIDDEN, 1, 1)),
    ("no_mem_embed", (1, 1, _HIDDEN)),
    ("no_obj_ptr", (1, _HIDDEN)),
    ("no_obj_embed_spatial", (1, _MEMORY_CHANNELS)),
)

_ARCHITECTURE_LAYER_INDICES = (
    ("sam_mask_decoder.transformer.layers.", (0, 1)),
    ("memory_attention.layers.", (0, 1, 2, 3)),
    ("memory_encoder.fuser.layers.", (0, 1)),
)


def _validate_native_tracker_weights(weights: Mapping[str, np.ndarray]) -> None:
    """Reject checkpoint architectures that do not match this fixed graph."""

    if not isinstance(weights, Mapping):
        raise TypeError("SAM2 HOI native tracker weights must be a parameter mapping")
    for key, shape in _ARCHITECTURE_WEIGHT_SHAPES:
        _weight(weights, key, shape)
    for prefix, expected in _ARCHITECTURE_LAYER_INDICES:
        actual = {
            int(suffix.partition(".")[0])
            for key in weights
            if key.startswith(prefix)
            and (suffix := key.removeprefix(prefix)).partition(".")[0].isdigit()
        }
        if actual != set(expected):
            raise ValueError(
                f"SAM2 HOI tracker checkpoint layers under {prefix!r} are "
                f"{tuple(sorted(actual))}, expected {expected}"
            )


def _linear(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    in_width: int,
    out_width: int,
    learned_dtype: Any,
):
    trt = _trt()
    tensor = _cast(network, tensor, learned_dtype)
    rank = len(tuple(tensor.shape))
    rhs_shape = (1,) * (rank - 2) + (in_width, out_width)
    rhs = _constant_for_dtype(
        network,
        _weight(weights, f"{prefix}.weight", (out_width, in_width)).T,
        rhs_shape,
        learned_dtype,
    )
    output = network.add_matrix_multiply(
        tensor,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    bias_shape = (1,) * (rank - 1) + (out_width,)
    bias = _constant_for_dtype(
        network,
        _weight(weights, f"{prefix}.bias", (out_width,)),
        bias_shape,
        learned_dtype,
    )
    return network.add_elementwise(output, bias, trt.ElementWiseOperation.SUM).get_output(0)


def _conv2d(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel: tuple[int, int],
    learned_dtype: Any,
    *,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    groups: int = 1,
):
    trt = _trt()
    tensor = _cast(network, tensor, learned_dtype)
    layer = network.add_convolution_nd(
        tensor,
        out_channels,
        kernel,
        trt.Weights(),
        trt.Weights(),
    )
    weight_shape = (out_channels, in_channels // groups, *kernel)
    layer.set_input(
        1,
        _constant_for_dtype(
            network,
            _weight(weights, f"{prefix}.weight", weight_shape),
            weight_shape,
            learned_dtype,
        ),
    )
    layer.set_input(
        2,
        _constant_for_dtype(
            network,
            _weight(weights, f"{prefix}.bias", (out_channels,)),
            (out_channels,),
            learned_dtype,
        ),
    )
    layer.stride_nd = stride
    layer.padding_nd = padding
    layer.num_groups = groups
    return layer.get_output(0)


def _layer_norm_last(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    width: int,
    *,
    epsilon: float = 1e-5,
):
    if width != _HIDDEN or epsilon != 1e-5:
        raise ValueError("SAM2 HOI tracker only qualifies nn.LayerNorm(width=256, epsilon=1e-5)")
    scale = _constant(
        network,
        _weight(weights, f"{prefix}.weight", (width,)),
        (width,),
    )
    bias = _constant(
        network,
        _weight(weights, f"{prefix}.bias", (width,)),
        (width,),
    )
    return graph_ops.add_plugin(
        network,
        "Sam2HoiLayerNorm256",
        (tensor, scale, bias),
        instance_name=prefix.replace(".", "_") + "_layer_norm",
    )


def _layer_norm_channels(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    channels: int,
    *,
    epsilon: float = 1e-6,
):
    """Reproduce the source ``LayerNorm2d`` autocast boundary exactly.

    Unlike ``nn.LayerNorm``, SAM2's channel-first helper computes its channel
    mean and subtraction in the incoming dtype.  With BF16 inputs the square,
    variance, reciprocal standard deviation, and learned affine then execute
    in FP32 and the module returns FP32.  Expressing those operations directly
    avoids both TensorRT INormalization's different reduction and the
    width-256 Welford plugin used for ordinary ``nn.LayerNorm``.
    """

    trt = _trt()
    if len(tuple(tensor.shape)) != 4 or int(tensor.shape[1]) != channels:
        raise ValueError(
            f"SAM2 HOI LayerNorm2d expected NCHW with {channels} channels, got {tensor.shape}"
        )
    mean = network.add_reduce(
        tensor,
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    centered = network.add_elementwise(
        tensor,
        mean,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    centered_fp32 = _cast(network, centered, trt.float32)
    square = network.add_elementwise(
        centered_fp32,
        centered_fp32,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    variance = network.add_reduce(
        square,
        trt.ReduceOperation.AVG,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    epsilon_tensor = _constant(
        network,
        np.full((1, 1, 1, 1), epsilon, dtype=np.float32),
    )
    variance = network.add_elementwise(
        variance,
        epsilon_tensor,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    standard_deviation = network.add_unary(
        variance,
        trt.UnaryOperation.SQRT,
    ).get_output(0)
    normalized = network.add_elementwise(
        centered_fp32,
        standard_deviation,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)
    parameter_shape = (1, channels, 1, 1)
    scale = _constant(
        network,
        _weight(weights, f"{prefix}.weight", (channels,)),
        parameter_shape,
    )
    bias = _constant(
        network,
        _weight(weights, f"{prefix}.bias", (channels,)),
        parameter_shape,
    )
    normalized = network.add_elementwise(
        normalized,
        scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    return network.add_elementwise(
        normalized,
        bias,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def _gelu(network: Any, tensor: Any):
    trt = _trt()
    tensor = _cast(network, tensor, trt.float32)
    inv_sqrt_two = _constant(
        network, np.array([1.0 / np.sqrt(2.0)], dtype=np.float32), (1,) * len(tuple(tensor.shape))
    )
    half = _constant(network, np.array([0.5], dtype=np.float32), (1,) * len(tuple(tensor.shape)))
    one = _constant(network, np.array([1.0], dtype=np.float32), (1,) * len(tuple(tensor.shape)))
    scaled = network.add_elementwise(
        tensor, inv_sqrt_two, trt.ElementWiseOperation.PROD
    ).get_output(0)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    shifted = network.add_elementwise(erf, one, trt.ElementWiseOperation.SUM).get_output(0)
    gated = network.add_elementwise(tensor, shifted, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(gated, half, trt.ElementWiseOperation.PROD).get_output(0)


def _attention_core(network: Any, query: Any, key: Any, value: Any):
    """Use TensorRT's decomposable scaled-dot-product attention primitive."""

    trt = _trt()
    head_width = int(query.shape[-1])
    scale = np.float32(1.0 / np.sqrt(head_width))
    scale_tensor = _constant_for_dtype(
        network,
        np.array([scale], dtype=np.float32),
        (1, 1, 1, 1),
        query.dtype,
    )
    query = network.add_elementwise(query, scale_tensor, trt.ElementWiseOperation.PROD).get_output(
        0
    )
    attention = network.add_attention(
        query,
        key,
        value,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    return attention.get_output(0)


@dataclass(frozen=True)
class _DecoderOutputs:
    masks: Any
    iou_scores: Any
    mask_tokens: Any
    object_score_logits: Any


@dataclass(frozen=True)
class _SelectedOutputs:
    mask: Any
    iou_score: Any
    mask_token: Any


@dataclass(frozen=True)
class _HeadOutputs:
    pred_masks: Any
    object_pointer: Any
    object_score_logits: Any
    selected_iou: Any


def _decoder_attention(
    network: Any,
    query: Any,
    key: Any,
    value: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    *,
    query_length: int,
    key_length: int,
    internal_width: int,
    learned_dtype: Any,
):
    trt = _trt()
    head_width = internal_width // _DECODER_HEADS
    query = _linear(
        network, query, weights, f"{prefix}.q_proj", _HIDDEN, internal_width, learned_dtype
    )
    key = _linear(network, key, weights, f"{prefix}.k_proj", _HIDDEN, internal_width, learned_dtype)
    value = _linear(
        network, value, weights, f"{prefix}.v_proj", _HIDDEN, internal_width, learned_dtype
    )

    def to_heads(tensor: Any, length: int):
        tensor = _reshape(network, tensor, (_BATCH, length, _DECODER_HEADS, head_width))
        return _transpose(network, tensor, (0, 2, 1, 3))

    query = to_heads(query, query_length)
    key = to_heads(key, key_length)
    value = to_heads(value, key_length)
    context = _attention_core(network, query, key, value)
    context = _transpose(network, context, (0, 2, 1, 3))
    context = _reshape(network, context, (_BATCH, query_length, internal_width))
    del trt
    return _linear(
        network,
        context,
        weights,
        f"{prefix}.out_proj",
        internal_width,
        _HIDDEN,
        learned_dtype,
    )


def _relu(network: Any, tensor: Any):
    return network.add_activation(tensor, _trt().ActivationType.RELU).get_output(0)


def _exact_sigmoid(network: Any, tensor: Any, *, instance_name: str):
    """Use the family plugin that mirrors PyTorch 2.7.1 CUDA sigmoid."""

    return graph_ops.add_plugin(
        network,
        "Sam2HoiSigmoid",
        (tensor,),
        instance_name=instance_name,
    )


def _transformer_mlp(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    learned_dtype: Any,
):
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.layers.0",
        _HIDDEN,
        2048,
        learned_dtype,
    )
    tensor = _relu(network, tensor)
    return _linear(
        network,
        tensor,
        weights,
        f"{prefix}.layers.1",
        2048,
        _HIDDEN,
        learned_dtype,
    )


def _three_layer_mlp(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    hidden_width: int,
    output_width: int,
    learned_dtype: Any,
):
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.layers.0",
        _HIDDEN,
        hidden_width,
        learned_dtype,
    )
    tensor = _relu(network, tensor)
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.layers.1",
        hidden_width,
        hidden_width,
        learned_dtype,
    )
    tensor = _relu(network, tensor)
    return _linear(
        network,
        tensor,
        weights,
        f"{prefix}.layers.2",
        hidden_width,
        output_width,
        learned_dtype,
    )


def _round_float32_to_bfloat16(values: Any) -> np.ndarray:
    """Return FP32 carrier values rounded to BF16 with round-to-nearest-even."""

    array = np.ascontiguousarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)
    expanded = rounded.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(expanded.view(np.float32))


def _dense_random_position_encoding_values(
    weights: Mapping[str, np.ndarray],
    *,
    bf16: bool,
) -> np.ndarray:
    """Generate the input-independent dense random PE with source arithmetic."""

    coordinates = (np.arange(_GRID, dtype=np.float32) + np.float32(0.5)) / np.float32(_GRID)
    x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid = np.float32(2.0) * np.stack((x_coordinates, y_coordinates), axis=-1) - np.float32(1.0)
    projection = _weight(
        weights,
        "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
        (2, _HIDDEN // 2),
    )
    if bf16:
        grid = _round_float32_to_bfloat16(grid)
        projection = _round_float32_to_bfloat16(projection)

    # K is exactly two.  Spelling out the reduction reproduces CUDA BF16 GEMM's
    # FP32 accumulation deterministically and avoids a host BLAS dependency.
    phases = grid[..., 0, None] * projection[0] + grid[..., 1, None] * projection[1]
    if bf16:
        phases = _round_float32_to_bfloat16(phases)
    phases = phases * np.float32(2.0 * np.pi)
    if bf16:
        phases = _round_float32_to_bfloat16(phases)

    sine = np.sin(phases)
    cosine = np.cos(phases)
    if bf16:
        sine = _round_float32_to_bfloat16(sine)
        cosine = _round_float32_to_bfloat16(cosine)
    encoded = np.concatenate((sine, cosine), axis=-1).transpose(2, 0, 1)
    return np.ascontiguousarray(encoded[None], dtype=np.float32)


def _dense_random_position_encoding(
    network: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    """Materialize PromptEncoder.get_dense_pe as a checkpoint-dependent constant."""

    trt = _trt()
    values = _dense_random_position_encoding_values(
        weights,
        bf16=learned_dtype == trt.bfloat16,
    )
    return _constant_for_dtype(
        network,
        values,
        (1, _HIDDEN, _GRID, _GRID),
        learned_dtype,
    )


def _k2_projection(
    network: Any,
    coordinates: Any,
    projection: Any,
    learned_dtype: Any,
):
    """Reproduce the source K=2 autocast GEMM with explicit FP32 accumulation."""

    trt = _trt()
    coordinates = _cast(network, _cast(network, coordinates, learned_dtype), trt.float32)
    projection = _cast(network, projection, trt.float32)
    batch = int(coordinates.shape[0])
    points = int(coordinates.shape[1])
    width = int(projection.shape[2])
    coordinate_x = network.add_slice(
        coordinates,
        (0, 0, 0),
        (batch, points, 1),
        (1, 1, 1),
    ).get_output(0)
    coordinate_y = network.add_slice(
        coordinates,
        (0, 0, 1),
        (batch, points, 1),
        (1, 1, 1),
    ).get_output(0)
    projection_x = network.add_slice(
        projection,
        (0, 0, 0),
        (1, 1, width),
        (1, 1, 1),
    ).get_output(0)
    projection_y = network.add_slice(
        projection,
        (0, 1, 0),
        (1, 1, width),
        (1, 1, 1),
    ).get_output(0)
    product_x = network.add_elementwise(
        coordinate_x,
        projection_x,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    product_y = network.add_elementwise(
        coordinate_y,
        projection_y,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    projected = network.add_elementwise(
        product_x,
        product_y,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return _cast(network, projected, learned_dtype)


def _point_prompt_embeddings(
    network: Any,
    point_coords: Any,
    point_labels: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    trt = _trt()
    half = _constant(network, np.full((1, 1, 2), 0.5, dtype=np.float32))
    image_size = _constant(
        network,
        np.full((1, 1, 2), float(_IMAGE_SIZE), dtype=np.float32),
    )
    two = _constant(network, np.full((1, 1, 1), 2.0, dtype=np.float32))
    one = _constant(network, np.full((1, 1, 1), 1.0, dtype=np.float32))
    tau = _constant(network, np.full((1, 1, 1), 2.0 * np.pi, dtype=np.float32))
    coordinates = network.add_elementwise(
        point_coords, half, trt.ElementWiseOperation.SUM
    ).get_output(0)
    coordinates = network.add_elementwise(
        coordinates, image_size, trt.ElementWiseOperation.DIV
    ).get_output(0)
    coordinates = network.add_elementwise(
        coordinates, two, trt.ElementWiseOperation.PROD
    ).get_output(0)
    coordinates = network.add_elementwise(
        coordinates, one, trt.ElementWiseOperation.SUB
    ).get_output(0)
    projection = _constant_for_dtype(
        network,
        _weight(
            weights,
            "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            (2, _HIDDEN // 2),
        ),
        (1, 2, _HIDDEN // 2),
        learned_dtype,
    )
    phases = _k2_projection(network, coordinates, projection, learned_dtype)
    phases_fp32 = network.add_elementwise(
        _cast(network, phases, trt.float32),
        tau,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    phases = _cast(network, phases_fp32, learned_dtype)
    sine = network.add_unary(phases, trt.UnaryOperation.SIN).get_output(0)
    cosine = network.add_unary(phases, trt.UnaryOperation.COS).get_output(0)
    encoded = network.add_concatenation([sine, cosine])
    encoded.axis = 2
    # The raw random PE is BF16 under CUDA autocast, but all learned prompt
    # embeddings remain FP32, promoting sparse prompt output to FP32.
    base = _cast(network, encoded.get_output(0), learned_dtype)
    base_fp32 = _cast(network, base, trt.float32)

    labels = _reshape(network, point_labels, (_BATCH, 3, 1))
    not_a_point = _constant(
        network,
        _weight(weights, "sam_prompt_encoder.not_a_point_embed.weight", (1, _HIDDEN)),
        (1, 1, _HIDDEN),
    )
    minus_one = _constant(network, np.full((1, 1, 1), -1, dtype=np.int32), dtype=np.int32)
    is_padding = network.add_elementwise(
        labels, minus_one, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    result = network.add_select(is_padding, not_a_point, base_fp32).get_output(0)
    for label in range(4):
        label_value = _constant(
            network,
            np.full((1, 1, 1), label, dtype=np.int32),
            dtype=np.int32,
        )
        matches = network.add_elementwise(
            labels, label_value, trt.ElementWiseOperation.EQUAL
        ).get_output(0)
        learned = _constant(
            network,
            _weight(weights, f"sam_prompt_encoder.point_embeddings.{label}.weight", (1, _HIDDEN)),
            (1, 1, _HIDDEN),
        )
        candidate = network.add_elementwise(
            base_fp32,
            learned,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        result = network.add_select(matches, candidate, result).get_output(0)

    padded = np.tile(
        _weight(weights, "sam_prompt_encoder.not_a_point_embed.weight", (1, _HIDDEN))[None],
        (_BATCH, 1, 1),
    )
    padding = _constant(network, padded, (_BATCH, 1, _HIDDEN))
    concat = network.add_concatenation([result, padding])
    concat.axis = 1
    return concat.get_output(0)


def _empty_prompt_embeddings(
    network: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    value = _weight(weights, "sam_prompt_encoder.not_a_point_embed.weight", (1, _HIDDEN))
    value = np.tile(value.reshape(1, 1, _HIDDEN), (_BATCH, 2, 1))
    del learned_dtype
    return _constant(network, value, (_BATCH, 2, _HIDDEN))


def _no_mask_dense_embeddings(
    network: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    value = _weight(weights, "sam_prompt_encoder.no_mask_embed.weight", (1, _HIDDEN))
    value = np.tile(value.reshape(1, _HIDDEN, 1, 1), (_BATCH, 1, _GRID, _GRID))
    del learned_dtype
    return _constant(network, value, (_BATCH, _HIDDEN, _GRID, _GRID))


def _two_way_transformer(
    network: Any,
    point_embeddings: Any,
    image_embeddings: Any,
    image_position: Any,
    weights: Mapping[str, np.ndarray],
    *,
    point_count: int,
    learned_dtype: Any,
):
    queries = point_embeddings
    keys = image_embeddings
    for layer_index in range(2):
        prefix = f"sam_mask_decoder.transformer.layers.{layer_index}"
        if layer_index == 0:
            queries = _decoder_attention(
                network,
                queries,
                queries,
                queries,
                weights,
                f"{prefix}.self_attn",
                query_length=point_count,
                key_length=point_count,
                internal_width=_HIDDEN,
                learned_dtype=learned_dtype,
            )
        else:
            query_with_position = _fp32_sum(network, queries, point_embeddings)
            attention = _decoder_attention(
                network,
                query_with_position,
                query_with_position,
                queries,
                weights,
                f"{prefix}.self_attn",
                query_length=point_count,
                key_length=point_count,
                internal_width=_HIDDEN,
                learned_dtype=learned_dtype,
            )
            queries = _fp32_sum(network, queries, attention)
        queries = _layer_norm_last(network, queries, weights, f"{prefix}.norm1", _HIDDEN)

        token_query = _fp32_sum(network, queries, point_embeddings)
        image_key = _fp32_sum(network, keys, image_position)
        attention = _decoder_attention(
            network,
            token_query,
            image_key,
            keys,
            weights,
            f"{prefix}.cross_attn_token_to_image",
            query_length=point_count,
            key_length=_TOKENS,
            internal_width=_HIDDEN // 2,
            learned_dtype=learned_dtype,
        )
        queries = _fp32_sum(network, queries, attention)
        queries = _layer_norm_last(network, queries, weights, f"{prefix}.norm2", _HIDDEN)

        mlp = _transformer_mlp(network, queries, weights, f"{prefix}.mlp", learned_dtype)
        queries = _fp32_sum(network, queries, mlp)
        queries = _layer_norm_last(network, queries, weights, f"{prefix}.norm3", _HIDDEN)

        token_key = _fp32_sum(network, queries, point_embeddings)
        image_query = _fp32_sum(network, keys, image_position)
        attention = _decoder_attention(
            network,
            image_query,
            token_key,
            queries,
            weights,
            f"{prefix}.cross_attn_image_to_token",
            query_length=_TOKENS,
            key_length=point_count,
            internal_width=_HIDDEN // 2,
            learned_dtype=learned_dtype,
        )
        keys = _fp32_sum(network, keys, attention)
        keys = _layer_norm_last(network, keys, weights, f"{prefix}.norm4", _HIDDEN)

    final_query = _fp32_sum(network, queries, point_embeddings)
    final_key = _fp32_sum(network, keys, image_position)
    attention = _decoder_attention(
        network,
        final_query,
        final_key,
        keys,
        weights,
        "sam_mask_decoder.transformer.final_attn_token_to_image",
        query_length=point_count,
        key_length=_TOKENS,
        internal_width=_HIDDEN // 2,
        learned_dtype=learned_dtype,
    )
    queries = _fp32_sum(network, queries, attention)
    queries = _layer_norm_last(
        network,
        queries,
        weights,
        "sam_mask_decoder.transformer.norm_final_attn",
        _HIDDEN,
    )
    return queries, keys


def _pixel_shuffle_deconvolution(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    in_channels: int,
    out_channels: int,
    input_size: int,
    learned_dtype: Any,
):
    """Express a kernel-2 stride-2 deconvolution as Conv plus pixel shuffle."""

    trt = _trt()
    source_weight = _weight(
        weights,
        f"{prefix}.weight",
        (in_channels, out_channels, 2, 2),
    )
    packed_weight = np.ascontiguousarray(
        source_weight.transpose(1, 2, 3, 0).reshape(out_channels * 4, in_channels, 1, 1)
    )
    packed_bias = np.repeat(
        _weight(weights, f"{prefix}.bias", (out_channels,)),
        4,
    )
    tensor = _cast(network, tensor, learned_dtype)
    convolution = network.add_convolution_nd(
        tensor,
        out_channels * 4,
        (1, 1),
        trt.Weights(),
        trt.Weights(),
    )
    convolution.set_input(
        1,
        _constant_for_dtype(
            network,
            packed_weight,
            (out_channels * 4, in_channels, 1, 1),
            learned_dtype,
        ),
    )
    convolution.set_input(
        2,
        _constant_for_dtype(network, packed_bias, (out_channels * 4,), learned_dtype),
    )
    packed = _reshape(
        network,
        convolution.get_output(0),
        (_BATCH, out_channels, 2, 2, input_size, input_size),
    )
    packed = _transpose(network, packed, (0, 1, 4, 2, 5, 3))
    return _reshape(
        network,
        packed,
        (_BATCH, out_channels, input_size * 2, input_size * 2),
    )


def _add_mask_decoder(
    network: Any,
    feature_0: Any,
    feature_1: Any,
    image_features: Any,
    sparse_prompt: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
) -> _DecoderOutputs:
    trt = _trt()
    feature_0 = _cast(network, _expand_batch2(network, feature_0), learned_dtype)
    feature_1 = _cast(network, _expand_batch2(network, feature_1), learned_dtype)
    image_features = _expand_batch2(network, image_features)
    dense_prompt = _no_mask_dense_embeddings(network, weights, learned_dtype)

    output_tokens = np.concatenate(
        (
            _weight(weights, "sam_mask_decoder.obj_score_token.weight", (1, _HIDDEN)),
            _weight(weights, "sam_mask_decoder.iou_token.weight", (1, _HIDDEN)),
            _weight(weights, "sam_mask_decoder.mask_tokens.weight", (_MASK_TOKENS, _HIDDEN)),
        ),
        axis=0,
    )
    output_tokens = np.tile(output_tokens[None], (_BATCH, 1, 1))
    output_tokens = _constant(
        network,
        output_tokens,
        (_BATCH, _OUTPUT_TOKENS, _HIDDEN),
    )
    token_concat = network.add_concatenation([output_tokens, sparse_prompt])
    token_concat.axis = 1
    point_embeddings = token_concat.get_output(0)
    point_count = _OUTPUT_TOKENS + int(sparse_prompt.shape[1])

    image_features = _fp32_sum(network, image_features, dense_prompt)
    image_rows = _transpose(network, image_features, (0, 2, 3, 1))
    image_rows = _reshape(network, image_rows, (_BATCH, _TOKENS, _HIDDEN))
    position = _dense_random_position_encoding(network, weights, learned_dtype)
    position_batch = network.add_concatenation([position, position])
    position_batch.axis = 0
    position = position_batch.get_output(0)
    position_rows = _transpose(network, position, (0, 2, 3, 1))
    position_rows = _reshape(network, position_rows, (_BATCH, _TOKENS, _HIDDEN))
    point_outputs, image_outputs = _two_way_transformer(
        network,
        point_embeddings,
        image_rows,
        position_rows,
        weights,
        point_count=point_count,
        learned_dtype=learned_dtype,
    )

    iou_token = network.add_slice(
        point_outputs,
        (0, 1, 0),
        (_BATCH, 1, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)
    mask_tokens = network.add_slice(
        point_outputs,
        (0, 2, 0),
        (_BATCH, _MASK_TOKENS, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)
    object_token = network.add_slice(
        point_outputs,
        (0, 0, 0),
        (_BATCH, 1, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)

    upscaled = _reshape(network, image_outputs, (_BATCH, _GRID, _GRID, _HIDDEN))
    upscaled = _transpose(network, upscaled, (0, 3, 1, 2))
    upscaled = _pixel_shuffle_deconvolution(
        network,
        upscaled,
        weights,
        "sam_mask_decoder.output_upscaling.0",
        _HIDDEN,
        64,
        _GRID,
        learned_dtype,
    )
    upscaled = _sum_as(network, upscaled, feature_1, learned_dtype)
    upscaled = _layer_norm_channels(
        network,
        upscaled,
        weights,
        "sam_mask_decoder.output_upscaling.1",
        64,
    )
    upscaled = _cast(network, _gelu(network, upscaled), learned_dtype)
    upscaled = _pixel_shuffle_deconvolution(
        network,
        upscaled,
        weights,
        "sam_mask_decoder.output_upscaling.3",
        64,
        32,
        _GRID * 2,
        learned_dtype,
    )
    upscaled = _sum_as(network, upscaled, feature_0, learned_dtype)
    upscaled = _cast(network, _gelu(network, upscaled), learned_dtype)

    hypernetwork_outputs = []
    for index in range(_MASK_TOKENS):
        token = network.add_slice(
            mask_tokens,
            (0, index, 0),
            (_BATCH, 1, _HIDDEN),
            (1, 1, 1),
        ).get_output(0)
        hypernetwork_outputs.append(
            _three_layer_mlp(
                network,
                token,
                weights,
                f"sam_mask_decoder.output_hypernetworks_mlps.{index}",
                _HIDDEN,
                32,
                learned_dtype,
            )
        )
    hypernetwork = network.add_concatenation(hypernetwork_outputs)
    hypernetwork.axis = 1
    upscaled = _reshape(
        network,
        upscaled,
        (_BATCH, 32, _LOW_RESOLUTION * _LOW_RESOLUTION),
    )
    masks = network.add_matrix_multiply(
        hypernetwork.get_output(0),
        trt.MatrixOperation.NONE,
        upscaled,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    masks = _reshape(
        network,
        masks,
        (_BATCH, _MASK_TOKENS, _LOW_RESOLUTION, _LOW_RESOLUTION),
    )
    iou_scores = _three_layer_mlp(
        network,
        iou_token,
        weights,
        "sam_mask_decoder.iou_prediction_head",
        _HIDDEN,
        _MASK_TOKENS,
        learned_dtype,
    )
    iou_scores = _exact_sigmoid(
        network,
        iou_scores,
        instance_name="sam_mask_decoder_iou_prediction_sigmoid",
    )
    iou_scores = _reshape(network, iou_scores, (_BATCH, _MASK_TOKENS))
    object_score = _three_layer_mlp(
        network,
        object_token,
        weights,
        "sam_mask_decoder.pred_obj_score_head",
        _HIDDEN,
        1,
        learned_dtype,
    )
    return _DecoderOutputs(
        masks=masks,
        iou_scores=iou_scores,
        mask_tokens=mask_tokens,
        object_score_logits=_reshape(network, object_score, (_BATCH, 1)),
    )


def _best_candidate(
    network: Any,
    masks: Any,
    scores: Any,
    tokens: Any,
    *,
    count: int,
) -> _SelectedOutputs:
    trt = _trt()
    best_score = network.add_slice(scores, (0, 0), (_BATCH, 1), (1, 1)).get_output(0)
    best_indices = _constant(
        network,
        np.zeros((1, 1), dtype=np.int32),
        dtype=np.int32,
    )
    for index in range(1, count):
        candidate = network.add_slice(scores, (0, index), (_BATCH, 1), (1, 1)).get_output(0)
        better = network.add_elementwise(
            candidate, best_score, trt.ElementWiseOperation.GREATER
        ).get_output(0)
        best_score = network.add_select(better, candidate, best_score).get_output(0)
        candidate_index = _constant(
            network,
            np.full((1, 1), index, dtype=np.int32),
            dtype=np.int32,
        )
        best_indices = network.add_select(better, candidate_index, best_indices).get_output(0)
    indices = _constant(
        network,
        np.arange(count, dtype=np.int32).reshape(1, count),
        dtype=np.int32,
    )
    selector = network.add_elementwise(
        best_indices, indices, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_selector = _reshape(
        network,
        _cast(network, selector, masks.dtype),
        (_BATCH, count, 1, 1),
    )
    selected_mask = network.add_elementwise(
        masks, mask_selector, trt.ElementWiseOperation.PROD
    ).get_output(0)
    selected_mask = network.add_reduce(
        selected_mask,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    token_selector = _reshape(
        network,
        _cast(network, selector, tokens.dtype),
        (_BATCH, count, 1),
    )
    selected_token = network.add_elementwise(
        tokens, token_selector, trt.ElementWiseOperation.PROD
    ).get_output(0)
    selected_token = network.add_reduce(
        selected_token,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=False,
    ).get_output(0)
    return _SelectedOutputs(selected_mask, best_score, selected_token)


def _appearing(network: Any, object_score_logits: Any):
    zero = _constant_for_dtype(
        network,
        np.zeros((1, 1), dtype=np.float32),
        (1, 1),
        object_score_logits.dtype,
    )
    return network.add_elementwise(
        object_score_logits,
        zero,
        _trt().ElementWiseOperation.GREATER,
    ).get_output(0)


def _visible_masks(network: Any, masks: Any, object_score_logits: Any):
    condition = _reshape(network, _appearing(network, object_score_logits), (_BATCH, 1, 1, 1))
    no_object = _constant_for_dtype(
        network,
        np.full((1, 1, 1, 1), -1024.0, dtype=np.float32),
        (1, 1, 1, 1),
        masks.dtype,
    )
    return network.add_select(condition, masks, no_object).get_output(0)


def _object_pointer(
    network: Any,
    token: Any,
    score: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    pointer = _three_layer_mlp(
        network,
        token,
        weights,
        "obj_ptr_proj",
        _HIDDEN,
        _HIDDEN,
        learned_dtype,
    )
    no_object = _constant(
        network,
        _weight(weights, "no_obj_ptr", (1, _HIDDEN)),
        (1, _HIDDEN),
    )
    return network.add_select(
        _appearing(network, score),
        _cast(network, pointer, _trt().float32),
        no_object,
    ).get_output(0)


def _prompt_head(
    network: Any,
    feature_0: Any,
    feature_1: Any,
    feature_2: Any,
    point_coords: Any,
    point_labels: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
) -> _HeadOutputs:
    trt = _trt()
    sparse = _point_prompt_embeddings(
        network,
        point_coords,
        point_labels,
        weights,
        learned_dtype,
    )
    no_memory = _constant(
        network,
        _weight(weights, "no_mem_embed", (1, 1, _HIDDEN)),
        (1, _HIDDEN, 1, 1),
    )
    feature_2 = _fp32_sum(network, feature_2, no_memory)
    decoder = _add_mask_decoder(
        network,
        feature_0,
        feature_1,
        feature_2,
        sparse,
        weights,
        learned_dtype,
    )
    single_mask = network.add_slice(
        decoder.masks,
        (0, 0, 0, 0),
        (_BATCH, 1, _LOW_RESOLUTION, _LOW_RESOLUTION),
        (1, 1, 1, 1),
    ).get_output(0)
    single_iou = network.add_slice(
        decoder.iou_scores,
        (0, 0),
        (_BATCH, 1),
        (1, 1),
    ).get_output(0)
    single_token = network.add_slice(
        decoder.mask_tokens,
        (0, 0, 0),
        (_BATCH, 1, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)
    single_token = _reshape(network, single_token, (_BATCH, _HIDDEN))
    multi_masks = network.add_slice(
        decoder.masks,
        (0, 1, 0, 0),
        (_BATCH, 3, _LOW_RESOLUTION, _LOW_RESOLUTION),
        (1, 1, 1, 1),
    ).get_output(0)
    multi_ious = network.add_slice(
        decoder.iou_scores,
        (0, 1),
        (_BATCH, 3),
        (1, 1),
    ).get_output(0)
    multi_tokens = network.add_slice(
        decoder.mask_tokens,
        (0, 1, 0),
        (_BATCH, 3, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)
    best = _best_candidate(network, multi_masks, multi_ious, multi_tokens, count=3)

    upper = _constant_for_dtype(
        network,
        np.full((1, 1, 1, 1), 0.05, dtype=np.float32),
        (1, 1, 1, 1),
        single_mask.dtype,
    )
    lower = _constant_for_dtype(
        network,
        np.full((1, 1, 1, 1), -0.05, dtype=np.float32),
        (1, 1, 1, 1),
        single_mask.dtype,
    )
    intersection = network.add_elementwise(
        single_mask, upper, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    union = network.add_elementwise(
        single_mask, lower, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    intersection = network.add_reduce(
        _cast(network, intersection, trt.float32),
        trt.ReduceOperation.SUM,
        (1 << 2) | (1 << 3),
        keep_dims=False,
    ).get_output(0)
    union = network.add_reduce(
        _cast(network, union, trt.float32),
        trt.ReduceOperation.SUM,
        (1 << 2) | (1 << 3),
        keep_dims=False,
    ).get_output(0)
    one = _constant(network, np.ones((1, 1), dtype=np.float32))
    zero = _constant(network, np.zeros((1, 1), dtype=np.float32))
    positive_union = network.add_elementwise(
        union, zero, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    denominator = network.add_elementwise(union, one, trt.ElementWiseOperation.MAX).get_output(0)
    ratio = network.add_elementwise(
        intersection, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    stability = network.add_select(positive_union, ratio, one).get_output(0)
    threshold = _constant(network, np.full((1, 1), 0.98, dtype=np.float32))
    greater = network.add_elementwise(
        stability, threshold, trt.ElementWiseOperation.GREATER
    ).get_output(0)
    equal = network.add_elementwise(
        stability, threshold, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    stable = network.add_elementwise(greater, equal, trt.ElementWiseOperation.OR).get_output(0)
    stable_mask = _reshape(network, stable, (_BATCH, 1, 1, 1))
    selected_mask = network.add_select(stable_mask, single_mask, best.mask).get_output(0)
    selected_iou = network.add_select(stable, single_iou, best.iou_score).get_output(0)
    selected_mask = _visible_masks(network, selected_mask, decoder.object_score_logits)
    pointer = _object_pointer(
        network,
        single_token,
        decoder.object_score_logits,
        weights,
        learned_dtype,
    )
    return _HeadOutputs(
        pred_masks=_cast(network, selected_mask, trt.float32),
        object_pointer=_cast(network, pointer, trt.float32),
        object_score_logits=_cast(network, decoder.object_score_logits, trt.float32),
        selected_iou=_cast(network, selected_iou, trt.float32),
    )


def _recurrent_head(
    network: Any,
    feature_0: Any,
    feature_1: Any,
    conditioned: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
) -> _HeadOutputs:
    trt = _trt()
    sparse = _empty_prompt_embeddings(network, weights, learned_dtype)
    decoder = _add_mask_decoder(
        network,
        feature_0,
        feature_1,
        conditioned,
        sparse,
        weights,
        learned_dtype,
    )
    masks = network.add_slice(
        decoder.masks,
        (0, 1, 0, 0),
        (_BATCH, 3, _LOW_RESOLUTION, _LOW_RESOLUTION),
        (1, 1, 1, 1),
    ).get_output(0)
    ious = network.add_slice(
        decoder.iou_scores,
        (0, 1),
        (_BATCH, 3),
        (1, 1),
    ).get_output(0)
    tokens = network.add_slice(
        decoder.mask_tokens,
        (0, 1, 0),
        (_BATCH, 3, _HIDDEN),
        (1, 1, 1),
    ).get_output(0)
    masks = _visible_masks(network, masks, decoder.object_score_logits)
    selected = _best_candidate(network, masks, ious, tokens, count=3)
    pointer = _object_pointer(
        network,
        selected.mask_token,
        decoder.object_score_logits,
        weights,
        learned_dtype,
    )
    return _HeadOutputs(
        pred_masks=_cast(network, selected.mask, trt.float32),
        object_pointer=_cast(network, pointer, trt.float32),
        object_score_logits=_cast(network, decoder.object_score_logits, trt.float32),
        # Preserve the reviewed wrapper ABI: recurrent publication retains all
        # three candidate IoUs even though the mask has already been selected.
        selected_iou=_cast(network, ious, trt.float32),
    )


@dataclass(frozen=True)
class _PreparedMemory:
    spatial_values: Any
    spatial_position: Any
    pointer_values: Any
    pointer_position: Any


@dataclass(frozen=True)
class _RopeConstants:
    cosine: Any
    sine: Any
    rotated_indices: Any
    rotated_sign: Any


def _prepare_spatial_memory(
    network: Any,
    memory_features: Any,
    memory_position: Any,
    memory_temporal_offsets: Any,
    weights: Mapping[str, np.ndarray],
):
    trt = _trt()
    temporal = _weight(weights, "maskmem_tpos_enc", (_MEMORY_FRAMES, 1, 1, _MEMORY_CHANNELS))
    temporal_table = _constant(
        network,
        temporal.reshape(_MEMORY_FRAMES, _MEMORY_CHANNELS),
        (_MEMORY_FRAMES, _MEMORY_CHANNELS),
    )
    last_slot = _constant(
        network,
        np.full((1, 1), _MEMORY_FRAMES - 1, dtype=np.int32),
        dtype=np.int32,
    )
    temporal_indices = network.add_elementwise(
        last_slot,
        memory_temporal_offsets,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    temporal_position = network.add_gather(
        temporal_table,
        temporal_indices,
        axis=0,
    ).get_output(0)
    temporal_position = _reshape(
        network,
        temporal_position,
        (_BATCH, -1, 1, 1, _MEMORY_CHANNELS),
    )
    position = _transpose(network, memory_position, (0, 1, 3, 4, 2))
    position = _fp32_sum(network, position, temporal_position)
    position = _reshape(network, position, (_BATCH, -1, _MEMORY_CHANNELS))
    values = _transpose(network, memory_features, (0, 1, 3, 4, 2))
    values = _reshape(network, values, (_BATCH, -1, _MEMORY_CHANNELS))
    return values, position


def _prepare_pointer_memory(
    network: Any,
    object_pointers: Any,
    temporal_offsets: Any,
    denominator: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    trt = _trt()
    offsets = _reshape(network, temporal_offsets, (_BATCH, -1, 1))
    denominator = _reshape(network, denominator, (1, 1, 1))
    normalized = network.add_elementwise(
        offsets,
        denominator,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)
    sine_width = _HIDDEN // 2
    indices = np.arange(sine_width, dtype=np.int32)
    frequencies = np.power(
        np.float32(10000.0),
        2.0 * (indices // 2).astype(np.float32) / float(sine_width),
    ).astype(np.float32)
    frequencies = _constant(
        network,
        frequencies.reshape(1, 1, sine_width),
        (1, 1, sine_width),
    )
    phases = network.add_elementwise(
        normalized,
        frequencies,
        trt.ElementWiseOperation.DIV,
    ).get_output(0)
    sine = network.add_unary(phases, trt.UnaryOperation.SIN).get_output(0)
    cosine = network.add_unary(phases, trt.UnaryOperation.COS).get_output(0)
    position = network.add_concatenation([sine, cosine])
    position.axis = 2
    position = _linear(
        network,
        position.get_output(0),
        weights,
        "obj_ptr_tpos_proj",
        _HIDDEN,
        _MEMORY_CHANNELS,
        learned_dtype,
    )
    position = _reshape(network, position, (_BATCH, -1, 1, _MEMORY_CHANNELS))
    repeated_position = network.add_concatenation([position] * _POINTER_TOKENS)
    repeated_position.axis = 2
    position = _reshape(
        network,
        repeated_position.get_output(0),
        (_BATCH, -1, _MEMORY_CHANNELS),
    )
    pointers = _reshape(
        network,
        object_pointers,
        (_BATCH, -1, _POINTER_TOKENS, _MEMORY_CHANNELS),
    )
    pointers = _reshape(network, pointers, (_BATCH, -1, _MEMORY_CHANNELS))
    return pointers, position


def _prepare_memory(
    network: Any,
    memory_features: Any,
    memory_position: Any,
    memory_temporal_offsets: Any,
    object_pointers: Any,
    pointer_temporal_offsets: Any,
    pointer_denominator: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
) -> _PreparedMemory:
    spatial_values, spatial_position = _prepare_spatial_memory(
        network,
        memory_features,
        memory_position,
        memory_temporal_offsets,
        weights,
    )
    pointer_values, pointer_position = _prepare_pointer_memory(
        network,
        object_pointers,
        pointer_temporal_offsets,
        pointer_denominator,
        weights,
        learned_dtype,
    )
    return _PreparedMemory(
        spatial_values=spatial_values,
        spatial_position=spatial_position,
        pointer_values=pointer_values,
        pointer_position=pointer_position,
    )


@lru_cache(maxsize=1)
def _axial_rope_arrays() -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.power(
        np.float32(10000.0),
        -np.arange(0, _HIDDEN, 4, dtype=np.float32) / float(_HIDDEN),
    )
    positions = np.arange(_TOKENS, dtype=np.int32)
    x_positions = (positions % _GRID).astype(np.float32)
    y_positions = (positions // _GRID).astype(np.float32)
    x_angles = np.outer(x_positions, frequencies)
    y_angles = np.outer(y_positions, frequencies)
    angles = np.repeat(np.concatenate((x_angles, y_angles), axis=1), 2, axis=1)
    return (
        np.ascontiguousarray(np.cos(angles), dtype=np.float32),
        np.ascontiguousarray(np.sin(angles), dtype=np.float32),
    )


def _rope_constants(network: Any) -> _RopeConstants:
    cosine, sine = _axial_rope_arrays()
    cosine_tensor = _constant(
        network,
        cosine.reshape(1, 1, _TOKENS, _HIDDEN),
        (1, 1, _TOKENS, _HIDDEN),
    )
    sine_tensor = _constant(
        network,
        sine.reshape(1, 1, _TOKENS, _HIDDEN),
        (1, 1, _TOKENS, _HIDDEN),
    )
    indices = np.arange(_HIDDEN, dtype=np.int32).reshape(-1, 2)[:, ::-1].reshape(-1)
    rotated_indices = _constant(network, indices, (_HIDDEN,), dtype=np.int32)
    signs = np.tile(np.array([-1.0, 1.0], dtype=np.float32), _HIDDEN // 2)
    rotated_sign = _constant(
        network,
        signs.reshape(1, 1, 1, _HIDDEN),
        (1, 1, 1, _HIDDEN),
    )
    return _RopeConstants(cosine_tensor, sine_tensor, rotated_indices, rotated_sign)


def _apply_rope(network: Any, tensor: Any, constants: _RopeConstants):
    trt = _trt()
    output_dtype = tensor.dtype
    tensor = _cast(network, tensor, trt.float32)
    rotated = network.add_gather(tensor, constants.rotated_indices, axis=3).get_output(0)
    rotated = network.add_elementwise(
        rotated,
        constants.rotated_sign,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    cosine = network.add_elementwise(
        tensor,
        constants.cosine,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    sine = network.add_elementwise(
        rotated,
        constants.sine,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    result = network.add_elementwise(cosine, sine, trt.ElementWiseOperation.SUM).get_output(0)
    return _cast(network, result, output_dtype)


def _recurrent_attention_context(network: Any, query: Any, key: Any, value: Any):
    context = _attention_core(network, query, key, value)
    return _reshape(network, context, (_BATCH, _TOKENS, _HIDDEN))


def _recurrent_self_attention(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    rope: _RopeConstants,
    learned_dtype: Any,
):
    query = _linear(network, tensor, weights, f"{prefix}.q_proj", _HIDDEN, _HIDDEN, learned_dtype)
    key = _linear(network, tensor, weights, f"{prefix}.k_proj", _HIDDEN, _HIDDEN, learned_dtype)
    value = _linear(network, tensor, weights, f"{prefix}.v_proj", _HIDDEN, _HIDDEN, learned_dtype)
    query = _reshape(network, query, (_BATCH, 1, _TOKENS, _HIDDEN))
    key = _reshape(network, key, (_BATCH, 1, _TOKENS, _HIDDEN))
    value = _reshape(network, value, (_BATCH, 1, _TOKENS, _HIDDEN))
    query = _apply_rope(network, query, rope)
    key = _apply_rope(network, key, rope)
    context = _recurrent_attention_context(network, query, key, value)
    return _linear(
        network,
        context,
        weights,
        f"{prefix}.out_proj",
        _HIDDEN,
        _HIDDEN,
        learned_dtype,
    )


def _recurrent_cross_attention(
    network: Any,
    tensor: Any,
    memory: _PreparedMemory,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    rope: _RopeConstants,
    learned_dtype: Any,
):
    query = _linear(network, tensor, weights, f"{prefix}.q_proj", _HIDDEN, _HIDDEN, learned_dtype)
    spatial_key_input = _fp32_sum(network, memory.spatial_values, memory.spatial_position)
    pointer_key_input = _fp32_sum(network, memory.pointer_values, memory.pointer_position)
    spatial_key = _linear(
        network,
        spatial_key_input,
        weights,
        f"{prefix}.k_proj",
        _MEMORY_CHANNELS,
        _HIDDEN,
        learned_dtype,
    )
    pointer_key = _linear(
        network,
        pointer_key_input,
        weights,
        f"{prefix}.k_proj",
        _MEMORY_CHANNELS,
        _HIDDEN,
        learned_dtype,
    )
    spatial_value = _linear(
        network,
        memory.spatial_values,
        weights,
        f"{prefix}.v_proj",
        _MEMORY_CHANNELS,
        _HIDDEN,
        learned_dtype,
    )
    pointer_value = _linear(
        network,
        memory.pointer_values,
        weights,
        f"{prefix}.v_proj",
        _MEMORY_CHANNELS,
        _HIDDEN,
        learned_dtype,
    )

    query = _reshape(network, query, (_BATCH, 1, _TOKENS, _HIDDEN))
    query = _apply_rope(network, query, rope)
    spatial_key = _reshape(network, spatial_key, (_BATCH, -1, _TOKENS, _HIDDEN))
    spatial_key = _apply_rope(network, spatial_key, rope)
    spatial_key = _reshape(network, spatial_key, (_BATCH, 1, -1, _HIDDEN))
    pointer_key = _reshape(network, pointer_key, (_BATCH, 1, -1, _HIDDEN))
    key = network.add_concatenation([spatial_key, pointer_key])
    key.axis = 2
    spatial_value = _reshape(network, spatial_value, (_BATCH, 1, -1, _HIDDEN))
    pointer_value = _reshape(network, pointer_value, (_BATCH, 1, -1, _HIDDEN))
    value = network.add_concatenation([spatial_value, pointer_value])
    value.axis = 2
    context = _recurrent_attention_context(
        network,
        query,
        key.get_output(0),
        value.get_output(0),
    )
    return _linear(
        network,
        context,
        weights,
        f"{prefix}.out_proj",
        _HIDDEN,
        _HIDDEN,
        learned_dtype,
    )


def _recurrent_feed_forward(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    learned_dtype: Any,
):
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.linear1",
        _HIDDEN,
        2048,
        learned_dtype,
    )
    tensor = _relu(network, tensor)
    return _linear(
        network,
        tensor,
        weights,
        f"{prefix}.linear2",
        2048,
        _HIDDEN,
        learned_dtype,
    )


def _features_to_tokens(network: Any, features: Any):
    features = _expand_batch2(network, features)
    features = _transpose(network, features, (0, 2, 3, 1))
    return _reshape(network, features, (_BATCH, _TOKENS, _HIDDEN))


def _tokens_to_features(network: Any, tokens: Any):
    features = _reshape(network, tokens, (_BATCH, _GRID, _GRID, _HIDDEN))
    return _transpose(network, features, (0, 3, 1, 2))


def _recurrent_conditioning(
    network: Any,
    current_features: Any,
    current_position: Any,
    memory_features: Any,
    memory_position: Any,
    memory_temporal_offsets: Any,
    object_pointers: Any,
    pointer_temporal_offsets: Any,
    pointer_denominator: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    trt = _trt()
    memory = _prepare_memory(
        network,
        memory_features,
        memory_position,
        memory_temporal_offsets,
        object_pointers,
        pointer_temporal_offsets,
        pointer_denominator,
        weights,
        learned_dtype,
    )
    rope = _rope_constants(network)
    output = _features_to_tokens(network, current_features)
    position = _features_to_tokens(network, current_position)
    scale = _constant(network, np.full((1, 1, 1), 0.1, dtype=np.float32))
    position = network.add_elementwise(
        position,
        scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    output = _fp32_sum(network, output, position)

    for layer_index in range(4):
        prefix = f"memory_attention.layers.{layer_index}"
        normalized = _layer_norm_last(network, output, weights, f"{prefix}.norm1", _HIDDEN)
        attention = _recurrent_self_attention(
            network,
            normalized,
            weights,
            f"{prefix}.self_attn",
            rope,
            learned_dtype,
        )
        output = _fp32_sum(network, output, attention)

        normalized = _layer_norm_last(network, output, weights, f"{prefix}.norm2", _HIDDEN)
        attention = _recurrent_cross_attention(
            network,
            normalized,
            memory,
            weights,
            f"{prefix}.cross_attn_image",
            rope,
            learned_dtype,
        )
        output = _fp32_sum(network, output, attention)

        normalized = _layer_norm_last(network, output, weights, f"{prefix}.norm3", _HIDDEN)
        feed_forward = _recurrent_feed_forward(
            network,
            normalized,
            weights,
            prefix,
            learned_dtype,
        )
        output = _fp32_sum(network, output, feed_forward)

    output = _layer_norm_last(network, output, weights, "memory_attention.norm", _HIDDEN)
    return _tokens_to_features(network, output)


def _resize_mask(network: Any, mask: Any, target: int):
    trt = _trt()
    resize = network.add_resize(mask)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    resize.shape = (_BATCH, 1, target, target)
    return resize.get_output(0)


def _prepare_memory_mask(network: Any, masks: Any, from_points: Any):
    trt = _trt()
    masks = _resize_mask(network, masks, _IMAGE_SIZE)
    soft = _exact_sigmoid(
        network,
        masks,
        instance_name="memory_encoder_mask_sigmoid",
    )
    zero = _constant(network, np.zeros((1, 1, 1, 1), dtype=np.float32))
    positive = network.add_elementwise(masks, zero, trt.ElementWiseOperation.GREATER).get_output(0)
    hard = _cast(network, positive, trt.float32)
    from_points = _reshape(network, from_points, (_BATCH, 1, 1, 1))
    flag_zero = _constant(network, np.zeros((1, 1, 1, 1), dtype=np.int32), dtype=np.int32)
    use_hard = network.add_elementwise(
        from_points,
        flag_zero,
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    selected = network.add_select(use_hard, hard, soft).get_output(0)
    scale = _constant(network, np.full((1, 1, 1, 1), 20.0, dtype=np.float32))
    bias = _constant(network, np.full((1, 1, 1, 1), -10.0, dtype=np.float32))
    selected = network.add_elementwise(selected, scale, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(selected, bias, trt.ElementWiseOperation.SUM).get_output(0)


def _memory_mask_downsampler(
    network: Any,
    mask: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    specifications = (
        (0, 1, 4),
        (3, 4, 16),
        (6, 16, 64),
        (9, 64, 256),
    )
    tensor = mask
    for convolution_index, in_channels, out_channels in specifications:
        tensor = _conv2d(
            network,
            tensor,
            weights,
            f"memory_encoder.mask_downsampler.encoder.{convolution_index}",
            in_channels,
            out_channels,
            (3, 3),
            learned_dtype,
            stride=(2, 2),
            padding=(1, 1),
        )
        tensor = _layer_norm_channels(
            network,
            tensor,
            weights,
            f"memory_encoder.mask_downsampler.encoder.{convolution_index + 1}",
            out_channels,
        )
        tensor = _cast(network, _gelu(network, tensor), learned_dtype)
    return _conv2d(
        network,
        tensor,
        weights,
        "memory_encoder.mask_downsampler.encoder.12",
        256,
        256,
        (1, 1),
        learned_dtype,
    )


def _memory_fuser_block(
    network: Any,
    tensor: Any,
    weights: Mapping[str, np.ndarray],
    index: int,
    learned_dtype: Any,
):
    trt = _trt()
    prefix = f"memory_encoder.fuser.layers.{index}"
    # CXBlock retains its incoming residual verbatim.  The first block sees a
    # BF16 residual; the FP32 gamma multiplication and residual sum promote its
    # output, so the second block must retain an FP32 residual.
    residual = tensor
    tensor = _conv2d(
        network,
        tensor,
        weights,
        f"{prefix}.dwconv",
        256,
        256,
        (7, 7),
        learned_dtype,
        padding=(3, 3),
        groups=256,
    )
    tensor = _layer_norm_channels(network, tensor, weights, f"{prefix}.norm", 256)
    tensor = _transpose(network, _cast(network, tensor, learned_dtype), (0, 2, 3, 1))
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.pwconv1",
        256,
        1024,
        learned_dtype,
    )
    tensor = _cast(network, _gelu(network, tensor), learned_dtype)
    tensor = _linear(
        network,
        tensor,
        weights,
        f"{prefix}.pwconv2",
        1024,
        256,
        learned_dtype,
    )
    gamma = _constant(
        network,
        _weight(weights, f"{prefix}.gamma", (256,)),
        (1, 1, 1, 256),
    )
    tensor = network.add_elementwise(
        _cast(network, tensor, trt.float32),
        gamma,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    tensor = _transpose(network, tensor, (0, 3, 1, 2))
    return _fp32_sum(network, residual, tensor)


@lru_cache(maxsize=1)
def _memory_position_encoding() -> np.ndarray:
    y = np.arange(1, _GRID + 1, dtype=np.float32)
    x = np.arange(1, _GRID + 1, dtype=np.float32)
    y = y / np.float32(_GRID + 1e-6) * np.float32(2.0 * np.pi)
    x = x / np.float32(_GRID + 1e-6) * np.float32(2.0 * np.pi)
    y_grid = np.broadcast_to(y[:, None], (_GRID, _GRID))
    x_grid = np.broadcast_to(x[None, :], (_GRID, _GRID))
    positional_features = _MEMORY_CHANNELS // 2
    indices = np.arange(positional_features, dtype=np.float32)
    dimensions = np.power(
        np.float32(10000.0),
        2.0 * np.floor(indices / 2.0) / positional_features,
    ).astype(np.float32)
    position_x = x_grid[..., None] / dimensions
    position_y = y_grid[..., None] / dimensions
    position_x = np.stack(
        (np.sin(position_x[..., 0::2]), np.cos(position_x[..., 1::2])),
        axis=-1,
    ).reshape(_GRID, _GRID, positional_features)
    position_y = np.stack(
        (np.sin(position_y[..., 0::2]), np.cos(position_y[..., 1::2])),
        axis=-1,
    ).reshape(_GRID, _GRID, positional_features)
    position = np.concatenate((position_y, position_x), axis=-1)
    position = np.broadcast_to(
        position[None],
        (_BATCH, _GRID, _GRID, _MEMORY_CHANNELS),
    )
    return np.ascontiguousarray(position.transpose(0, 3, 1, 2), dtype=np.float32)


def _memory_encoder(
    network: Any,
    feature_2: Any,
    pred_masks: Any,
    object_score_logits: Any,
    from_points: Any,
    weights: Mapping[str, np.ndarray],
    learned_dtype: Any,
):
    trt = _trt()
    mask = _prepare_memory_mask(network, pred_masks, from_points)
    mask = _memory_mask_downsampler(network, mask, weights, learned_dtype)
    feature_2 = _expand_batch2(network, feature_2)
    feature_2 = _conv2d(
        network,
        feature_2,
        weights,
        "memory_encoder.pix_feat_proj",
        256,
        256,
        (1, 1),
        learned_dtype,
    )
    tensor = _sum_as(network, feature_2, mask, learned_dtype)
    for index in range(2):
        tensor = _memory_fuser_block(network, tensor, weights, index, learned_dtype)
    memory = _conv2d(
        network,
        tensor,
        weights,
        "memory_encoder.out_proj",
        256,
        _MEMORY_CHANNELS,
        (1, 1),
        learned_dtype,
    )

    scores = _reshape(network, object_score_logits, (_BATCH, 1, 1, 1))
    zero = _constant(network, np.zeros((1, 1, 1, 1), dtype=np.float32))
    appearing = network.add_elementwise(scores, zero, trt.ElementWiseOperation.GREATER).get_output(
        0
    )
    appearing = _cast(network, appearing, trt.float32)
    one = _constant(network, np.ones((1, 1, 1, 1), dtype=np.float32))
    absent = network.add_elementwise(one, appearing, trt.ElementWiseOperation.SUB).get_output(0)
    no_object = _constant(
        network,
        _weight(weights, "no_obj_embed_spatial", (1, _MEMORY_CHANNELS)),
        (1, _MEMORY_CHANNELS, 1, 1),
    )
    no_object = network.add_elementwise(
        absent, no_object, trt.ElementWiseOperation.PROD
    ).get_output(0)
    memory = network.add_elementwise(
        _cast(network, memory, trt.float32),
        no_object,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    memory = _cast(network, memory, learned_dtype)
    position = _constant_for_dtype(
        network,
        _memory_position_encoding(),
        (_BATCH, _MEMORY_CHANNELS, _GRID, _GRID),
        learned_dtype,
    )
    return memory, position


def _learned_dtype_for_precision(trt: Any, precision: str):
    return trt.bfloat16 if _normalize_precision(precision) == "bf16" else trt.float32


def _build_prompt_plan(
    weights: Mapping[str, np.ndarray],
    bindings: NativeEngineBindings,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    trt, builder, network, config = _new_network(verbose=verbose)
    tensors = {binding.name: _input(network, binding) for binding in bindings.inputs}
    outputs = _prompt_head(
        network,
        tensors["tracker_feature_0"],
        tensors["tracker_feature_1"],
        tensors["tracker_feature_2"],
        tensors["point_coords"],
        tensors["point_labels"],
        weights,
        _learned_dtype_for_precision(trt, precision),
    )
    values = {
        "pred_masks": outputs.pred_masks,
        "object_pointer": outputs.object_pointer,
        "object_score_logits": outputs.object_score_logits,
        "selected_iou": outputs.selected_iou,
    }
    for binding in bindings.outputs:
        _mark(network, values[binding.name], binding)
    return _serialize(builder, network, config, bindings.section, verbose=verbose)


def _build_recurrent_plan(
    weights: Mapping[str, np.ndarray],
    bindings: NativeEngineBindings,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    trt, builder, network, config = _new_network(verbose=verbose)
    tensors = {binding.name: _input(network, binding) for binding in bindings.inputs}
    learned_dtype = _learned_dtype_for_precision(trt, precision)
    conditioned = _recurrent_conditioning(
        network,
        tensors["tracker_feature_2"],
        tensors["tracker_position_2"],
        tensors["memory_features"],
        tensors["memory_position"],
        tensors["memory_temporal_offsets"],
        tensors["object_pointers"],
        tensors["object_pointer_temporal_offsets"],
        tensors["object_pointer_time_denominator"],
        weights,
        learned_dtype,
    )
    outputs = _recurrent_head(
        network,
        tensors["tracker_feature_0"],
        tensors["tracker_feature_1"],
        conditioned,
        weights,
        learned_dtype,
    )
    values = {
        "pred_masks": outputs.pred_masks,
        "object_pointer": outputs.object_pointer,
        "object_score_logits": outputs.object_score_logits,
        "selected_iou": outputs.selected_iou,
    }
    for binding in bindings.outputs:
        _mark(network, values[binding.name], binding)
    _add_profiles(builder, config, bindings)
    return _serialize(builder, network, config, bindings.section, verbose=verbose)


def _build_memory_plan(
    weights: Mapping[str, np.ndarray],
    bindings: NativeEngineBindings,
    *,
    precision: str,
    verbose: bool,
) -> bytes:
    trt, builder, network, config = _new_network(verbose=verbose)
    tensors = {binding.name: _input(network, binding) for binding in bindings.inputs}
    memory, position = _memory_encoder(
        network,
        tensors["tracker_feature_2"],
        tensors["pred_masks"],
        tensors["object_score_logits"],
        tensors["is_mask_from_points"],
        weights,
        _learned_dtype_for_precision(trt, precision),
    )
    values = {"new_memory_features": memory, "new_memory_position": position}
    for binding in bindings.outputs:
        _mark(network, values[binding.name], binding)
    return _serialize(builder, network, config, bindings.section, verbose=verbose)


def build_tracker_engines(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> dict[str, bytes]:
    """Build all three direct plans from reviewed checkpoint arrays."""

    precision = _normalize_precision(precision)
    _validate_native_tracker_weights(weights)
    prompt, recurrent, memory = tracker_binding_specs(precision)
    return {
        prompt.section: _build_prompt_plan(
            weights,
            prompt,
            precision=precision,
            verbose=verbose,
        ),
        recurrent.section: _build_recurrent_plan(
            weights,
            recurrent,
            precision=precision,
            verbose=verbose,
        ),
        memory.section: _build_memory_plan(
            weights,
            memory,
            precision=precision,
            verbose=verbose,
        ),
    }


__all__ = [
    "MEMORY_ENCODER_SECTION",
    "NativeBinding",
    "NativeEngineBindings",
    "PROMPT_TRACKER_SECTION",
    "RECURRENT_TRACKER_SECTION",
    "build_tracker_engines",
    "tracker_binding_specs",
]
