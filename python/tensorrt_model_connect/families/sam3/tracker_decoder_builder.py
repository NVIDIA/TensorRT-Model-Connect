# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT graph helpers for the SAM3 video tracker mask head.

The public helpers in this module consume raw NumPy arrays from the official
SAM3 checkpoint.  They reconstruct the prompt mask encoder, two-way mask
decoder, multimask policy, object-pointer projection, and mask-input
initialization path with TensorRT network layers.

The tracker runtime has two fixed execution policies: one object and two
objects.  ``object_batch`` therefore accepts only 1 or 2.  The two-object
path may share the frame-level high-resolution features (batch one); the
helpers explicitly replicate those tensors before object-specific decoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


_HIDDEN_SIZE = 256
_IMAGE_GRID = 72
_IMAGE_TOKENS = _IMAGE_GRID * _IMAGE_GRID
_LOW_RESOLUTION = 288
_IMAGE_SIZE = 1008
_NUM_HEADS = 8
_NUM_OUTPUT_TOKENS = 6
_NUM_MASK_TOKENS = 4
_NUM_MULTIMASKS = 3
_TRANSFORMER_LAYER_NORM_EPS = 1e-5
_SPATIAL_LAYER_NORM_EPS = 1e-6
_WEIGHT_PREFIX = "tracker_model."

WeightMap = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class DecoderOutputs:
    """Mask decoder outputs after applying the three-mask tracker policy."""

    masks: object
    iou_scores: object
    mask_tokens: object
    object_score_logits: object


@dataclass(frozen=True)
class SelectedMaskOutputs:
    """One independently selected mask and token per tracked object."""

    mask: object
    iou_score: object
    mask_token: object
    object_score_logits: object


@dataclass(frozen=True)
class TrackerStepHeadOutputs:
    """Outputs consumed by the recurrent SAM3 session after one frame."""

    pred_masks: object
    object_pointer: object
    object_score_logits: object


@dataclass(frozen=True)
class TrackerInitHeadOutputs:
    """Mask-input initialization outputs before spatial-memory encoding."""

    pred_masks: object
    high_res_masks: object
    object_pointer: object
    object_score_logits: object


def _trt():
    from tensorrt_model_connect import trt_compat

    return trt_compat.get_trt()


def _graph_ops():
    from . import graph_ops

    return graph_ops


def _validate_object_batch(object_batch: int) -> None:
    if object_batch not in (1, 2):
        raise ValueError(f"SAM3 tracker object_batch must be 1 or 2, got {object_batch}")


def _weight_key(weight_prefix: str, suffix: str) -> str:
    return f"{weight_prefix}{suffix}"


def _weight(
    weights: WeightMap,
    suffix: str,
    shape: tuple[int, ...],
    *,
    weight_prefix: str,
) -> np.ndarray:
    key = _weight_key(weight_prefix, suffix)
    try:
        value = np.asarray(weights[key])
    except KeyError as error:
        raise KeyError(f"SAM3 tracker checkpoint is missing {key}") from error
    if tuple(value.shape) != shape:
        raise ValueError(f"SAM3 tracker weight {key} has shape {value.shape}, expected {shape}")
    return np.ascontiguousarray(value, dtype=np.float32)


def _constant(network, shape: tuple[int, ...], values, *, dtype=np.float32):
    return _graph_ops().add_constant(
        network,
        shape,
        np.asarray(values).reshape(shape),
        dtype=dtype,
    )


def _linear(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    input_width: int,
    output_width: int,
    *,
    weight_prefix: str,
):
    graph_ops = _graph_ops()
    checkpoint_weight = _weight(
        weights,
        f"{prefix}.weight",
        (output_width, input_width),
        weight_prefix=weight_prefix,
    )
    out = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        input_width,
        output_width,
        checkpoint_weight.T,
    )
    bias = _weight(
        weights,
        f"{prefix}.bias",
        (output_width,),
        weight_prefix=weight_prefix,
    )
    return graph_ops.add_bias_sum(network, out, output_width, bias)


def _layer_norm(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    width: int,
    *,
    epsilon: float = _TRANSFORMER_LAYER_NORM_EPS,
    weight_prefix: str,
):
    return _graph_ops().add_layer_norm_native(
        network,
        inp,
        width,
        _weight(
            weights,
            f"{prefix}.weight",
            (width,),
            weight_prefix=weight_prefix,
        ),
        _weight(
            weights,
            f"{prefix}.bias",
            (width,),
            weight_prefix=weight_prefix,
        ),
        epsilon,
    )


def _channels_first_layer_norm(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    channels: int,
    *,
    object_batch: int,
    height: int,
    width: int,
    weight_prefix: str,
):
    trt = _trt()
    to_channels_last = network.add_shuffle(inp)
    to_channels_last.first_transpose = trt.Permutation([0, 2, 3, 1])
    to_channels_last.reshape_dims = (object_batch, height, width, channels)
    normed = _layer_norm(
        network,
        to_channels_last.get_output(0),
        weights,
        prefix,
        channels,
        epsilon=_SPATIAL_LAYER_NORM_EPS,
        weight_prefix=weight_prefix,
    )
    to_channels_first = network.add_shuffle(normed)
    to_channels_first.first_transpose = trt.Permutation([0, 3, 1, 2])
    to_channels_first.reshape_dims = (object_batch, channels, height, width)
    return to_channels_first.get_output(0)


def _feed_forward(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    input_width: int,
    hidden_width: int,
    output_width: int,
    *,
    weight_prefix: str,
):
    trt = _trt()
    out = _linear(
        network,
        inp,
        weights,
        f"{prefix}.proj_in",
        input_width,
        hidden_width,
        weight_prefix=weight_prefix,
    )
    out = network.add_activation(out, trt.ActivationType.RELU).get_output(0)
    out = _linear(
        network,
        out,
        weights,
        f"{prefix}.layers.0",
        hidden_width,
        hidden_width,
        weight_prefix=weight_prefix,
    )
    out = network.add_activation(out, trt.ActivationType.RELU).get_output(0)
    return _linear(
        network,
        out,
        weights,
        f"{prefix}.proj_out",
        hidden_width,
        output_width,
        weight_prefix=weight_prefix,
    )


def _decoder_mlp(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    *,
    weight_prefix: str,
):
    trt = _trt()
    out = _linear(
        network,
        inp,
        weights,
        f"{prefix}.proj_in",
        _HIDDEN_SIZE,
        2048,
        weight_prefix=weight_prefix,
    )
    out = network.add_activation(out, trt.ActivationType.RELU).get_output(0)
    return _linear(
        network,
        out,
        weights,
        f"{prefix}.proj_out",
        2048,
        _HIDDEN_SIZE,
        weight_prefix=weight_prefix,
    )


def _reshape_batched_rows_to_heads(
    network,
    inp,
    *,
    object_batch: int,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
):
    layer = network.add_shuffle(inp)
    layer.reshape_dims = (object_batch, sequence_length, num_heads, head_dim)
    layer.second_transpose = _trt().Permutation([0, 2, 1, 3])
    return layer.get_output(0)


def _reshape_heads_to_batched_rows(
    network,
    inp,
    *,
    object_batch: int,
    sequence_length: int,
    width: int,
):
    layer = network.add_shuffle(inp)
    layer.first_transpose = _trt().Permutation([0, 2, 1, 3])
    layer.reshape_dims = (object_batch, sequence_length, width)
    return layer.get_output(0)


def _attention(
    network,
    query,
    key,
    value,
    weights: WeightMap,
    prefix: str,
    *,
    object_batch: int,
    query_length: int,
    key_length: int,
    internal_width: int,
    fp16_attention: bool,
    weight_prefix: str,
):
    trt = _trt()
    graph_ops = _graph_ops()
    head_dim = internal_width // _NUM_HEADS
    query_projected = _linear(
        network,
        query,
        weights,
        f"{prefix}.q_proj",
        _HIDDEN_SIZE,
        internal_width,
        weight_prefix=weight_prefix,
    )
    key_projected = _linear(
        network,
        key,
        weights,
        f"{prefix}.k_proj",
        _HIDDEN_SIZE,
        internal_width,
        weight_prefix=weight_prefix,
    )
    value_projected = _linear(
        network,
        value,
        weights,
        f"{prefix}.v_proj",
        _HIDDEN_SIZE,
        internal_width,
        weight_prefix=weight_prefix,
    )
    output_dtype = query_projected.dtype
    if fp16_attention:
        query_projected = network.add_cast(query_projected, trt.float16).get_output(0)
        key_projected = network.add_cast(key_projected, trt.float16).get_output(0)
        value_projected = network.add_cast(value_projected, trt.float16).get_output(0)
    query_heads = _reshape_batched_rows_to_heads(
        network,
        query_projected,
        object_batch=object_batch,
        sequence_length=query_length,
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
    )
    key_heads = _reshape_batched_rows_to_heads(
        network,
        key_projected,
        object_batch=object_batch,
        sequence_length=key_length,
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
    )
    value_heads = _reshape_batched_rows_to_heads(
        network,
        value_projected,
        object_batch=object_batch,
        sequence_length=key_length,
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
    )
    context = graph_ops.add_attention_core(network, query_heads, key_heads, value_heads)
    context = _reshape_heads_to_batched_rows(
        network,
        context,
        object_batch=object_batch,
        sequence_length=query_length,
        width=internal_width,
    )
    if context.dtype != output_dtype:
        context = network.add_cast(context, output_dtype).get_output(0)
    return _linear(
        network,
        context,
        weights,
        f"{prefix}.o_proj",
        internal_width,
        _HIDDEN_SIZE,
        weight_prefix=weight_prefix,
    )


def _match_object_batch(network, inp, object_batch: int):
    source_batch = int(inp.shape[0])
    if source_batch == object_batch:
        return inp
    if source_batch != 1:
        raise ValueError(
            "SAM3 tracker frame feature batch must be one or match object_batch; "
            f"got feature batch {source_batch} and object_batch {object_batch}"
        )
    concat = network.add_concatenation([inp] * object_batch)
    concat.axis = 0
    return concat.get_output(0)


def _nchw_to_rows(
    network,
    inp,
    *,
    object_batch: int,
    channels: int,
    height: int,
    width: int,
):
    layer = network.add_shuffle(inp)
    layer.first_transpose = _trt().Permutation([0, 2, 3, 1])
    layer.reshape_dims = (object_batch, height * width, channels)
    return layer.get_output(0)


def _rows_to_nchw(
    network,
    inp,
    *,
    object_batch: int,
    channels: int,
    height: int,
    width: int,
):
    layer = network.add_shuffle(inp)
    layer.reshape_dims = (object_batch, height, width, channels)
    layer.second_transpose = _trt().Permutation([0, 3, 1, 2])
    return layer.get_output(0)


def _add_deconvolution(
    network,
    inp,
    weights: WeightMap,
    prefix: str,
    input_channels: int,
    output_channels: int,
    *,
    weight_prefix: str,
):
    trt = _trt()
    weight = _weight(
        weights,
        f"{prefix}.weight",
        (input_channels, output_channels, 2, 2),
        weight_prefix=weight_prefix,
    )
    bias = _weight(
        weights,
        f"{prefix}.bias",
        (output_channels,),
        weight_prefix=weight_prefix,
    )
    layer = network.add_deconvolution_nd(
        inp,
        num_output_maps=output_channels,
        kernel_shape=(2, 2),
        kernel=trt.Weights(weight),
        bias=trt.Weights(bias),
    )
    layer.stride_nd = (2, 2)
    return layer.get_output(0)


def _add_bilinear_resize(
    network,
    inp,
    *,
    object_batch: int,
    channels: int,
    target_height: int,
    target_width: int,
):
    trt = _trt()
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    resize.shape = (object_batch, channels, target_height, target_width)
    return resize.get_output(0)


def make_image_position_embedding(
    weights: WeightMap,
    *,
    weight_prefix: str = _WEIGHT_PREFIX,
) -> np.ndarray:
    """Return the checkpoint's fixed image-wide embedding as ``[1,256,72,72]``."""

    projection = _weight(
        weights,
        "shared_image_embedding.positional_embedding",
        (2, _HIDDEN_SIZE // 2),
        weight_prefix=weight_prefix,
    )
    coordinates = np.arange(_IMAGE_GRID, dtype=np.float32) + 0.5
    coordinates /= float(_IMAGE_GRID)
    x_coordinates, y_coordinates = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid = np.stack((x_coordinates, y_coordinates), axis=-1)
    angles = ((2.0 * grid - 1.0) @ projection) * (2.0 * np.pi)
    embedding = np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)
    return np.ascontiguousarray(embedding.transpose(2, 0, 1)[None], dtype=np.float32)


def make_linear_antialias_matrix(input_size: int, output_size: int) -> np.ndarray:
    """Build the separable linear antialias operator for a fixed downsample."""

    if input_size <= 0 or output_size <= 0 or output_size > input_size:
        raise ValueError(
            "SAM3 linear antialias matrix requires 0 < output_size <= input_size, "
            f"got {input_size} -> {output_size}"
        )
    scale = float(input_size) / float(output_size)
    source = np.arange(input_size, dtype=np.float64)[None, :]
    centers = ((np.arange(output_size, dtype=np.float64) + 0.5) * scale - 0.5)[:, None]
    matrix = np.maximum(0.0, 1.0 - np.abs(source - centers) / scale)
    matrix /= matrix.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def add_separable_antialias_downsample(
    network,
    inp,
    *,
    object_batch: int,
    input_size: int,
    output_size: int,
):
    """Apply a fixed square linear-antialias downsample using two matrix products."""

    trt = _trt()
    matrix = make_linear_antialias_matrix(input_size, output_size).T
    matrix_tensor = _constant(
        network,
        (1, 1, input_size, output_size),
        matrix.reshape(1, 1, input_size, output_size),
    )
    width_reduced = network.add_matrix_multiply(
        inp,
        trt.MatrixOperation.NONE,
        matrix_tensor,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    transpose = network.add_shuffle(width_reduced)
    transpose.first_transpose = trt.Permutation([0, 1, 3, 2])
    transpose.reshape_dims = (object_batch, 1, output_size, input_size)
    height_reduced = network.add_matrix_multiply(
        transpose.get_output(0),
        trt.MatrixOperation.NONE,
        matrix_tensor,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    restore = network.add_shuffle(height_reduced)
    restore.first_transpose = trt.Permutation([0, 1, 3, 2])
    restore.reshape_dims = (object_batch, 1, output_size, output_size)
    return restore.get_output(0)


def add_empty_prompt_embeddings(
    network,
    weights: WeightMap,
    *,
    object_batch: int,
    weight_prefix: str = _WEIGHT_PREFIX,
):
    """Create the two padded empty-point embeddings used by tracker propagation."""

    _validate_object_batch(object_batch)
    not_a_point = _weight(
        weights,
        "prompt_encoder.not_a_point_embed.weight",
        (1, _HIDDEN_SIZE),
        weight_prefix=weight_prefix,
    )
    embeddings = np.tile(not_a_point.reshape(1, 1, _HIDDEN_SIZE), (object_batch, 2, 1))
    return _constant(network, embeddings.shape, embeddings)


def add_no_mask_dense_embeddings(
    network,
    weights: WeightMap,
    *,
    object_batch: int,
    weight_prefix: str = _WEIGHT_PREFIX,
):
    """Create the dense no-mask prompt used by ordinary recurrent steps."""

    _validate_object_batch(object_batch)
    no_mask = _weight(
        weights,
        "prompt_encoder.no_mask_embed.weight",
        (1, _HIDDEN_SIZE),
        weight_prefix=weight_prefix,
    )
    dense = np.tile(
        no_mask.reshape(1, _HIDDEN_SIZE, 1, 1),
        (object_batch, 1, _IMAGE_GRID, _IMAGE_GRID),
    )
    return _constant(network, dense.shape, dense)


def add_mask_prompt_encoder(
    network,
    input_masks,
    weights: WeightMap,
    *,
    object_batch: int,
    weight_prefix: str = _WEIGHT_PREFIX,
):
    """Encode ``[B,1,288,288]`` prompt masks into ``[B,256,72,72]``."""

    _validate_object_batch(object_batch)
    graph_ops = _graph_ops()
    hidden = graph_ops.add_conv2d(
        network,
        input_masks,
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv1.weight",
            (4, 1, 2, 2),
            weight_prefix=weight_prefix,
        ),
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv1.bias",
            (4,),
            weight_prefix=weight_prefix,
        ),
        4,
        (2, 2),
        stride=(2, 2),
    )
    hidden = _channels_first_layer_norm(
        network,
        hidden,
        weights,
        "prompt_encoder.mask_embed.layer_norm1",
        4,
        object_batch=object_batch,
        height=144,
        width=144,
        weight_prefix=weight_prefix,
    )
    hidden = graph_ops.add_gelu_erf(network, hidden)
    hidden = graph_ops.add_conv2d(
        network,
        hidden,
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv2.weight",
            (16, 4, 2, 2),
            weight_prefix=weight_prefix,
        ),
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv2.bias",
            (16,),
            weight_prefix=weight_prefix,
        ),
        16,
        (2, 2),
        stride=(2, 2),
    )
    hidden = _channels_first_layer_norm(
        network,
        hidden,
        weights,
        "prompt_encoder.mask_embed.layer_norm2",
        16,
        object_batch=object_batch,
        height=_IMAGE_GRID,
        width=_IMAGE_GRID,
        weight_prefix=weight_prefix,
    )
    hidden = graph_ops.add_gelu_erf(network, hidden)
    return graph_ops.add_conv2d(
        network,
        hidden,
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv3.weight",
            (_HIDDEN_SIZE, 16, 1, 1),
            weight_prefix=weight_prefix,
        ),
        _weight(
            weights,
            "prompt_encoder.mask_embed.conv3.bias",
            (_HIDDEN_SIZE,),
            weight_prefix=weight_prefix,
        ),
        _HIDDEN_SIZE,
        (1, 1),
    )


def _two_way_transformer(
    network,
    point_embeddings,
    image_embeddings,
    image_position_embeddings,
    weights: WeightMap,
    *,
    object_batch: int,
    fp16_attention: bool,
    weight_prefix: str,
):
    trt = _trt()
    queries = point_embeddings
    keys = image_embeddings
    for layer_index in range(2):
        prefix = f"mask_decoder.transformer.layers.{layer_index}"
        if layer_index == 0:
            queries = _attention(
                network,
                queries,
                queries,
                queries,
                weights,
                f"{prefix}.self_attn",
                object_batch=object_batch,
                query_length=8,
                key_length=8,
                internal_width=_HIDDEN_SIZE,
                fp16_attention=fp16_attention,
                weight_prefix=weight_prefix,
            )
        else:
            self_attention_input = network.add_elementwise(
                queries,
                point_embeddings,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
            self_attention = _attention(
                network,
                self_attention_input,
                self_attention_input,
                queries,
                weights,
                f"{prefix}.self_attn",
                object_batch=object_batch,
                query_length=8,
                key_length=8,
                internal_width=_HIDDEN_SIZE,
                fp16_attention=fp16_attention,
                weight_prefix=weight_prefix,
            )
            queries = network.add_elementwise(
                queries,
                self_attention,
                trt.ElementWiseOperation.SUM,
            ).get_output(0)
        queries = _layer_norm(
            network,
            queries,
            weights,
            f"{prefix}.layer_norm1",
            _HIDDEN_SIZE,
            weight_prefix=weight_prefix,
        )

        token_query = network.add_elementwise(
            queries,
            point_embeddings,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        image_key = network.add_elementwise(
            keys,
            image_position_embeddings,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        cross_attention = _attention(
            network,
            token_query,
            image_key,
            keys,
            weights,
            f"{prefix}.cross_attn_token_to_image",
            object_batch=object_batch,
            query_length=8,
            key_length=_IMAGE_TOKENS,
            internal_width=_HIDDEN_SIZE // 2,
            fp16_attention=fp16_attention,
            weight_prefix=weight_prefix,
        )
        queries = network.add_elementwise(
            queries,
            cross_attention,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        queries = _layer_norm(
            network,
            queries,
            weights,
            f"{prefix}.layer_norm2",
            _HIDDEN_SIZE,
            weight_prefix=weight_prefix,
        )

        mlp_out = _decoder_mlp(
            network,
            queries,
            weights,
            f"{prefix}.mlp",
            weight_prefix=weight_prefix,
        )
        queries = network.add_elementwise(
            queries,
            mlp_out,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        queries = _layer_norm(
            network,
            queries,
            weights,
            f"{prefix}.layer_norm3",
            _HIDDEN_SIZE,
            weight_prefix=weight_prefix,
        )

        token_key = network.add_elementwise(
            queries,
            point_embeddings,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        image_query = network.add_elementwise(
            keys,
            image_position_embeddings,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        image_attention = _attention(
            network,
            image_query,
            token_key,
            queries,
            weights,
            f"{prefix}.cross_attn_image_to_token",
            object_batch=object_batch,
            query_length=_IMAGE_TOKENS,
            key_length=8,
            internal_width=_HIDDEN_SIZE // 2,
            fp16_attention=fp16_attention,
            weight_prefix=weight_prefix,
        )
        keys = network.add_elementwise(
            keys,
            image_attention,
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        keys = _layer_norm(
            network,
            keys,
            weights,
            f"{prefix}.layer_norm4",
            _HIDDEN_SIZE,
            weight_prefix=weight_prefix,
        )

    final_query = network.add_elementwise(
        queries,
        point_embeddings,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    final_key = network.add_elementwise(
        keys,
        image_position_embeddings,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    final_attention = _attention(
        network,
        final_query,
        final_key,
        keys,
        weights,
        "mask_decoder.transformer.final_attn_token_to_image",
        object_batch=object_batch,
        query_length=8,
        key_length=_IMAGE_TOKENS,
        internal_width=_HIDDEN_SIZE // 2,
        fp16_attention=fp16_attention,
        weight_prefix=weight_prefix,
    )
    queries = network.add_elementwise(
        queries,
        final_attention,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    queries = _layer_norm(
        network,
        queries,
        weights,
        "mask_decoder.transformer.layer_norm_final_attn",
        _HIDDEN_SIZE,
        weight_prefix=weight_prefix,
    )
    return queries, keys


def add_mask_decoder(
    network,
    feature_0,
    feature_1,
    image_features,
    dense_prompt_embeddings,
    weights: WeightMap,
    *,
    object_batch: int,
    sparse_prompt_embeddings=None,
    fp16_attention: bool = False,
    weight_prefix: str = _WEIGHT_PREFIX,
) -> DecoderOutputs:
    """Build the SAM3 mask decoder and return its three tracker candidates.

    ``feature_0`` and ``feature_1`` are the frame-level, pre-projected
    high-resolution maps.  They may have batch one for the two-object path.
    ``image_features`` and the prompt tensors must have ``object_batch``.
    """

    _validate_object_batch(object_batch)
    trt = _trt()
    graph_ops = _graph_ops()
    feature_0 = _match_object_batch(network, feature_0, object_batch)
    feature_1 = _match_object_batch(network, feature_1, object_batch)
    image_features = _match_object_batch(network, image_features, object_batch)
    dense_prompt_embeddings = _match_object_batch(network, dense_prompt_embeddings, object_batch)
    if sparse_prompt_embeddings is None:
        sparse_prompt_embeddings = add_empty_prompt_embeddings(
            network,
            weights,
            object_batch=object_batch,
            weight_prefix=weight_prefix,
        )

    output_tokens = np.concatenate(
        (
            _weight(
                weights,
                "mask_decoder.obj_score_token.weight",
                (1, _HIDDEN_SIZE),
                weight_prefix=weight_prefix,
            ),
            _weight(
                weights,
                "mask_decoder.iou_token.weight",
                (1, _HIDDEN_SIZE),
                weight_prefix=weight_prefix,
            ),
            _weight(
                weights,
                "mask_decoder.mask_tokens.weight",
                (_NUM_MASK_TOKENS, _HIDDEN_SIZE),
                weight_prefix=weight_prefix,
            ),
        ),
        axis=0,
    )
    output_tokens = np.tile(output_tokens[None], (object_batch, 1, 1))
    output_token_tensor = _constant(network, output_tokens.shape, output_tokens)
    point_concat = network.add_concatenation([output_token_tensor, sparse_prompt_embeddings])
    point_concat.axis = 1
    point_embeddings = point_concat.get_output(0)

    image_features = network.add_elementwise(
        image_features,
        dense_prompt_embeddings,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    image_rows = _nchw_to_rows(
        network,
        image_features,
        object_batch=object_batch,
        channels=_HIDDEN_SIZE,
        height=_IMAGE_GRID,
        width=_IMAGE_GRID,
    )
    image_position = make_image_position_embedding(weights, weight_prefix=weight_prefix)
    image_position = np.tile(image_position, (object_batch, 1, 1, 1))
    image_position_tensor = _constant(
        network,
        image_position.shape,
        image_position,
    )
    image_position_rows = _nchw_to_rows(
        network,
        image_position_tensor,
        object_batch=object_batch,
        channels=_HIDDEN_SIZE,
        height=_IMAGE_GRID,
        width=_IMAGE_GRID,
    )
    point_outputs, image_outputs = _two_way_transformer(
        network,
        point_embeddings,
        image_rows,
        image_position_rows,
        weights,
        object_batch=object_batch,
        fp16_attention=fp16_attention,
        weight_prefix=weight_prefix,
    )

    iou_token = network.add_slice(
        point_outputs,
        (0, 1, 0),
        (object_batch, 1, _HIDDEN_SIZE),
        (1, 1, 1),
    ).get_output(0)
    mask_tokens = network.add_slice(
        point_outputs,
        (0, 2, 0),
        (object_batch, _NUM_MASK_TOKENS, _HIDDEN_SIZE),
        (1, 1, 1),
    ).get_output(0)
    object_score_token = network.add_slice(
        point_outputs,
        (0, 0, 0),
        (object_batch, 1, _HIDDEN_SIZE),
        (1, 1, 1),
    ).get_output(0)

    upscaled = _rows_to_nchw(
        network,
        image_outputs,
        object_batch=object_batch,
        channels=_HIDDEN_SIZE,
        height=_IMAGE_GRID,
        width=_IMAGE_GRID,
    )
    upscaled = _add_deconvolution(
        network,
        upscaled,
        weights,
        "mask_decoder.upscale_conv1",
        _HIDDEN_SIZE,
        64,
        weight_prefix=weight_prefix,
    )
    upscaled = network.add_elementwise(
        upscaled,
        feature_1,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    upscaled = _channels_first_layer_norm(
        network,
        upscaled,
        weights,
        "mask_decoder.upscale_layer_norm",
        64,
        object_batch=object_batch,
        height=144,
        width=144,
        weight_prefix=weight_prefix,
    )
    upscaled = graph_ops.add_gelu_erf(network, upscaled)
    upscaled = _add_deconvolution(
        network,
        upscaled,
        weights,
        "mask_decoder.upscale_conv2",
        64,
        32,
        weight_prefix=weight_prefix,
    )
    upscaled = network.add_elementwise(
        upscaled,
        feature_0,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    upscaled = graph_ops.add_gelu_erf(network, upscaled)

    hypernetwork_outputs = []
    for mask_index in range(_NUM_MASK_TOKENS):
        token = network.add_slice(
            mask_tokens,
            (0, mask_index, 0),
            (object_batch, 1, _HIDDEN_SIZE),
            (1, 1, 1),
        ).get_output(0)
        hypernetwork_outputs.append(
            _feed_forward(
                network,
                token,
                weights,
                f"mask_decoder.output_hypernetworks_mlps.{mask_index}",
                _HIDDEN_SIZE,
                _HIDDEN_SIZE,
                32,
                weight_prefix=weight_prefix,
            )
        )
    hypernetwork_concat = network.add_concatenation(hypernetwork_outputs)
    hypernetwork_concat.axis = 1
    hypernetwork = hypernetwork_concat.get_output(0)
    flattened_upscale = network.add_shuffle(upscaled)
    flattened_upscale.reshape_dims = (
        object_batch,
        32,
        _LOW_RESOLUTION * _LOW_RESOLUTION,
    )
    masks = network.add_matrix_multiply(
        hypernetwork,
        trt.MatrixOperation.NONE,
        flattened_upscale.get_output(0),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    mask_shape = network.add_shuffle(masks)
    mask_shape.reshape_dims = (
        object_batch,
        _NUM_MASK_TOKENS,
        _LOW_RESOLUTION,
        _LOW_RESOLUTION,
    )

    iou_scores = _feed_forward(
        network,
        iou_token,
        weights,
        "mask_decoder.iou_prediction_head",
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        _NUM_MASK_TOKENS,
        weight_prefix=weight_prefix,
    )
    iou_scores = network.add_activation(iou_scores, trt.ActivationType.SIGMOID).get_output(0)
    iou_scores_shape = network.add_shuffle(iou_scores)
    iou_scores_shape.reshape_dims = (object_batch, _NUM_MASK_TOKENS)
    object_score = _feed_forward(
        network,
        object_score_token,
        weights,
        "mask_decoder.pred_obj_score_head",
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        1,
        weight_prefix=weight_prefix,
    )
    object_score_shape = network.add_shuffle(object_score)
    object_score_shape.reshape_dims = (object_batch, 1)

    multimasks = network.add_slice(
        mask_shape.get_output(0),
        (0, 1, 0, 0),
        (object_batch, _NUM_MULTIMASKS, _LOW_RESOLUTION, _LOW_RESOLUTION),
        (1, 1, 1, 1),
    ).get_output(0)
    multiiou = network.add_slice(
        iou_scores_shape.get_output(0),
        (0, 1),
        (object_batch, _NUM_MULTIMASKS),
        (1, 1),
    ).get_output(0)
    multitokens = network.add_slice(
        mask_tokens,
        (0, 1, 0),
        (object_batch, _NUM_MULTIMASKS, _HIDDEN_SIZE),
        (1, 1, 1),
    ).get_output(0)
    return DecoderOutputs(
        masks=multimasks,
        iou_scores=multiiou,
        mask_tokens=multitokens,
        object_score_logits=object_score_shape.get_output(0),
    )


def add_multimask_selection(
    network,
    decoder_outputs: DecoderOutputs,
    *,
    object_batch: int,
) -> SelectedMaskOutputs:
    """Apply object visibility and select each object's highest-IoU mask."""

    _validate_object_batch(object_batch)
    trt = _trt()
    zero_score = _constant(network, (1, 1), np.zeros((1, 1), dtype=np.float32))
    object_appearing = network.add_elementwise(
        decoder_outputs.object_score_logits,
        zero_score,
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    appearing_for_masks = network.add_shuffle(object_appearing)
    appearing_for_masks.reshape_dims = (object_batch, 1, 1, 1)
    no_object_masks = _constant(
        network,
        (1, 1, 1, 1),
        np.full((1, 1, 1, 1), -1024.0, dtype=np.float32),
    )
    visible_masks = network.add_select(
        appearing_for_masks.get_output(0),
        decoder_outputs.masks,
        no_object_masks,
    ).get_output(0)

    best = network.add_topk(
        decoder_outputs.iou_scores,
        trt.TopKOperation.MAX,
        1,
        1 << 1,
    )
    best_indices = best.get_output(1)
    candidates = _constant(
        network,
        (1, _NUM_MULTIMASKS),
        np.arange(_NUM_MULTIMASKS, dtype=np.int32).reshape(1, -1),
        dtype=np.int32,
    )
    selector = network.add_elementwise(
        best_indices,
        candidates,
        trt.ElementWiseOperation.EQUAL,
    ).get_output(0)
    selector_float = network.add_cast(selector, visible_masks.dtype).get_output(0)

    mask_selector = network.add_shuffle(selector_float)
    mask_selector.reshape_dims = (object_batch, _NUM_MULTIMASKS, 1, 1)
    selected_masks = network.add_elementwise(
        visible_masks,
        mask_selector.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    selected_masks = network.add_reduce(
        selected_masks,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=True,
    ).get_output(0)

    token_selector = network.add_shuffle(selector_float)
    token_selector.reshape_dims = (object_batch, _NUM_MULTIMASKS, 1)
    selected_token = network.add_elementwise(
        decoder_outputs.mask_tokens,
        token_selector.get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    selected_token = network.add_reduce(
        selected_token,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=False,
    ).get_output(0)

    selected_iou = network.add_elementwise(
        decoder_outputs.iou_scores,
        selector_float,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    selected_iou = network.add_reduce(
        selected_iou,
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=True,
    ).get_output(0)
    return SelectedMaskOutputs(
        mask=selected_masks,
        iou_score=selected_iou,
        mask_token=selected_token,
        object_score_logits=decoder_outputs.object_score_logits,
    )


def add_object_pointer_projection(
    network,
    mask_token,
    object_score_logits,
    weights: WeightMap,
    *,
    object_batch: int,
    weight_prefix: str = _WEIGHT_PREFIX,
):
    """Project selected SAM tokens and apply the learned no-object pointer."""

    _validate_object_batch(object_batch)
    trt = _trt()
    pointer = _feed_forward(
        network,
        mask_token,
        weights,
        "object_pointer_proj",
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        weight_prefix=weight_prefix,
    )
    zero = _constant(network, (1, 1), np.zeros((1, 1), dtype=np.float32))
    appearing = network.add_elementwise(
        object_score_logits,
        zero,
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    no_object_pointer = _weight(
        weights,
        "no_object_pointer",
        (1, _HIDDEN_SIZE),
        weight_prefix=weight_prefix,
    )
    no_object_pointer = _constant(
        network,
        (1, _HIDDEN_SIZE),
        no_object_pointer,
    )
    return network.add_select(appearing, pointer, no_object_pointer).get_output(0)


def add_tracker_step_head(
    network,
    feature_0,
    feature_1,
    conditioned_features,
    weights: WeightMap,
    *,
    object_batch: int,
    fp16_attention: bool = False,
    weight_prefix: str = _WEIGHT_PREFIX,
) -> TrackerStepHeadOutputs:
    """Build the recurrent frame's no-prompt mask decoder and pointer head."""

    dense_prompt = add_no_mask_dense_embeddings(
        network,
        weights,
        object_batch=object_batch,
        weight_prefix=weight_prefix,
    )
    decoder_outputs = add_mask_decoder(
        network,
        feature_0,
        feature_1,
        conditioned_features,
        dense_prompt,
        weights,
        object_batch=object_batch,
        fp16_attention=fp16_attention,
        weight_prefix=weight_prefix,
    )
    selected = add_multimask_selection(network, decoder_outputs, object_batch=object_batch)
    pointer = add_object_pointer_projection(
        network,
        selected.mask_token,
        selected.object_score_logits,
        weights,
        object_batch=object_batch,
        weight_prefix=weight_prefix,
    )
    return TrackerStepHeadOutputs(
        pred_masks=selected.mask,
        object_pointer=pointer,
        object_score_logits=selected.object_score_logits,
    )


def add_tracker_init_head(
    network,
    feature_0,
    feature_1,
    image_features,
    detector_mask,
    weights: WeightMap,
    *,
    weight_prefix: str = _WEIGHT_PREFIX,
) -> TrackerInitHeadOutputs:
    """Build the official mask-input initialization and pointer path."""

    trt = _trt()
    graph_ops = _graph_ops()
    object_batch = 1
    resized_detector_mask = _add_bilinear_resize(
        network,
        detector_mask,
        object_batch=object_batch,
        channels=1,
        target_height=_IMAGE_SIZE,
        target_width=_IMAGE_SIZE,
    )
    scale = _constant(network, (1, 1, 1, 1), np.array([20.0], dtype=np.float32))
    bias = _constant(network, (1, 1, 1, 1), np.array([-10.0], dtype=np.float32))
    high_res_mask = network.add_elementwise(
        resized_detector_mask,
        scale,
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    high_res_mask = network.add_elementwise(
        high_res_mask,
        bias,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    low_res_mask = add_separable_antialias_downsample(
        network,
        high_res_mask,
        object_batch=object_batch,
        input_size=_IMAGE_SIZE,
        output_size=_LOW_RESOLUTION,
    )

    downsampled_prompt = graph_ops.add_conv2d(
        network,
        resized_detector_mask,
        _weight(
            weights,
            "mask_downsample.weight",
            (1, 1, 4, 4),
            weight_prefix=weight_prefix,
        ),
        _weight(
            weights,
            "mask_downsample.bias",
            (1,),
            weight_prefix=weight_prefix,
        ),
        1,
        (4, 4),
        stride=(4, 4),
    )
    downsampled_prompt = _add_bilinear_resize(
        network,
        downsampled_prompt,
        object_batch=object_batch,
        channels=1,
        target_height=_LOW_RESOLUTION,
        target_width=_LOW_RESOLUTION,
    )
    dense_prompt = add_mask_prompt_encoder(
        network,
        downsampled_prompt,
        weights,
        object_batch=object_batch,
        weight_prefix=weight_prefix,
    )
    decoder_outputs = add_mask_decoder(
        network,
        feature_0,
        feature_1,
        image_features,
        dense_prompt,
        weights,
        object_batch=object_batch,
        weight_prefix=weight_prefix,
    )
    selected = add_multimask_selection(network, decoder_outputs, object_batch=1)
    decoder_pointer = add_object_pointer_projection(
        network,
        selected.mask_token,
        selected.object_score_logits,
        weights,
        object_batch=1,
        weight_prefix=weight_prefix,
    )

    mask_max = network.add_reduce(
        detector_mask,
        trt.ReduceOperation.MAX,
        (1 << 1) | (1 << 2) | (1 << 3),
        keep_dims=False,
    ).get_output(0)
    mask_max_shape = network.add_shuffle(mask_max)
    mask_max_shape.reshape_dims = (1, 1)
    zero = _constant(network, (1, 1), np.zeros((1, 1), dtype=np.float32))
    object_appearing = network.add_elementwise(
        mask_max_shape.get_output(0),
        zero,
        trt.ElementWiseOperation.GREATER,
    ).get_output(0)
    no_object_pointer = _constant(
        network,
        (1, _HIDDEN_SIZE),
        _weight(
            weights,
            "no_object_pointer",
            (1, _HIDDEN_SIZE),
            weight_prefix=weight_prefix,
        ),
    )
    object_pointer = network.add_select(
        object_appearing,
        decoder_pointer,
        no_object_pointer,
    ).get_output(0)
    appearing_float = network.add_cast(object_appearing, detector_mask.dtype).get_output(0)
    object_score = network.add_elementwise(
        appearing_float,
        _constant(network, (1, 1), np.array([20.0], dtype=np.float32)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    object_score = network.add_elementwise(
        object_score,
        _constant(network, (1, 1), np.array([-10.0], dtype=np.float32)),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return TrackerInitHeadOutputs(
        pred_masks=low_res_mask,
        high_res_masks=high_res_mask,
        object_pointer=object_pointer,
        object_score_logits=object_score,
    )
