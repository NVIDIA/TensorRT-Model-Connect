# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT vision encoder for Phi-4-multimodal's Dynamic-HD image path."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops
from .utils import const_in_work_dtype, create_builder_context


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


_CROP_SIZE = 448
_PATCH_SIZE = 14
_GRID_SIZE = _CROP_SIZE // _PATCH_SIZE
_NUM_PATCHES = _GRID_SIZE * _GRID_SIZE
_EMBED_DIM = 1152
_MLP_DIM = 4304
_NUM_HEADS = 16
_HEAD_DIM = _EMBED_DIM // _NUM_HEADS
_NUM_EXECUTED_LAYERS = 26  # hidden_states[-2] from the 27-layer SigLIP tower
_POOLED_GRID = _GRID_SIZE // 2
_POOLED_PER_ENTRY = _POOLED_GRID * _POOLED_GRID
_TEXT_HIDDEN = 3072
# Dynamic-HD: the global view plus up to 36 crops (Phi4MMImageProcessor
# dynamic_hd). Image tokens: crop rows with separators, the global separator
# and the 16 x 17 global view.
MAX_CROPS = 36
MAX_ENTRIES = 1 + MAX_CROPS
MIN_IMAGE_TOKENS = 1 * (1 + 1) + 1 + _POOLED_GRID * (_POOLED_GRID + 1)
MAX_IMAGE_TOKENS = max(
    (_POOLED_GRID * rows) * (_POOLED_GRID * cols + 1)
    for rows in range(1, MAX_CROPS + 1)
    for cols in range(1, MAX_CROPS // rows + 1)
) + 1 + _POOLED_GRID * (_POOLED_GRID + 1)
_OPT_ENTRIES = 2
_OPT_IMAGE_TOKENS = 545  # one full 448x448 crop
# Rows the image-token gather reads: [sub_GN, glb_GN, pooled features...].
_FIRST_FEATURE_ROW = 2
_MASKED_KEY = -1.0e4


def _require(weights: WeightDict, key: str) -> np.ndarray:
    value = weights.get(key)
    if value is None:
        raise RuntimeError(f"Missing Phi-4 vision weight: {key}")
    return np.asarray(value)


def navit_bucket_table() -> np.ndarray:
    """SiglipVisionEmbeddings' bucketized patch coordinates by valid patch count.

    Ported from the checkpoint's vision_siglip_navit.py (Apache-2.0); see
    families/phi4_multimodal/NOTICE.

    Row ``n`` holds ``bucketize(arange(0, 1 - 1e-6, 1 / n), arange(1/32, 1, 1/32),
    right=True)`` for ``n`` valid patches, evaluated in the checkpoint's float32
    arithmetic; row 0 (no valid patches) stays zero.
    """
    boundaries = (np.arange(1, _GRID_SIZE, dtype=np.float64) / _GRID_SIZE).astype(np.float32)
    table = np.zeros((_GRID_SIZE + 1, _GRID_SIZE), dtype=np.int32)
    for count in range(1, _GRID_SIZE + 1):
        step = np.float32(1.0) / np.float32(count)
        coords = (np.arange(count, dtype=np.float64) * np.float64(step)).astype(np.float32)
        table[count, :count] = np.searchsorted(boundaries, coords, side="right")
    return table


def _cast(network, tensor, dtype):
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _linear(network, inp, weight, bias, work_np_dtype):
    weight = np.asarray(weight, dtype=np.float32)
    out_features, in_features = weight.shape
    result = graph_ops.add_matmul_rhs_constant(
        network, inp, in_features, out_features, weight.T,
        dtype=work_np_dtype,
        fp32_accumulation=work_np_dtype == np.float16)
    if bias is not None:
        result = graph_ops.add_bias_sum(
            network, result, out_features, np.asarray(bias),
            dtype=work_np_dtype)
    return result


def _gelu_tanh(network, inp):
    """SigLIP's gelu_pytorch_tanh, evaluated in FP32 like PyTorch's half kernels."""
    output_dtype = inp.dtype
    value = _cast(network, inp, trt.float32)
    value = network.add_activation(value, trt.ActivationType.GELU_TANH).get_output(0)
    return _cast(network, value, output_dtype)


def _build_encoder_layer(
    network,
    hidden,
    weights,
    layer_idx,
    attention_mask,
    work_np_dtype,
):
    prefix = f"img_processor.encoder.layers.{layer_idx}"
    normed = graph_ops.add_layer_norm_native(
        network, hidden, _EMBED_DIM,
        _require(weights, f"{prefix}.layer_norm1.weight"),
        _require(weights, f"{prefix}.layer_norm1.bias"),
        1.0e-6, dtype=work_np_dtype, fp32_compute=True)

    q = _linear(
        network, normed,
        _require(weights, f"{prefix}.self_attn.q_proj.weight"),
        _require(weights, f"{prefix}.self_attn.q_proj.bias"),
        work_np_dtype)
    k = _linear(
        network, normed,
        _require(weights, f"{prefix}.self_attn.k_proj.weight"),
        _require(weights, f"{prefix}.self_attn.k_proj.bias"),
        work_np_dtype)
    v = _linear(
        network, normed,
        _require(weights, f"{prefix}.self_attn.v_proj.weight"),
        _require(weights, f"{prefix}.self_attn.v_proj.bias"),
        work_np_dtype)

    def as_heads(tensor):
        layer = network.add_shuffle(tensor)
        # 0 keeps the dynamic vision batch (global view + crops).
        layer.reshape_dims = (0, _NUM_PATCHES, _NUM_HEADS, _HEAD_DIM)
        layer.second_transpose = trt.Permutation([0, 2, 1, 3])
        return layer.get_output(0)

    context = graph_ops.add_siglip_attention_core(
        network, as_heads(q), as_heads(k), as_heads(v),
        mask=attention_mask,
        scale=_HEAD_DIM ** -0.5,
    )
    context_rows = network.add_shuffle(context)
    context_rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    context_rows.reshape_dims = (0, _NUM_PATCHES, _EMBED_DIM)
    attention_out = _linear(
        network, context_rows.get_output(0),
        _require(weights, f"{prefix}.self_attn.out_proj.weight"),
        _require(weights, f"{prefix}.self_attn.out_proj.bias"),
        work_np_dtype)
    hidden = network.add_elementwise(
        hidden, attention_out, trt.ElementWiseOperation.SUM).get_output(0)

    normed = graph_ops.add_layer_norm_native(
        network, hidden, _EMBED_DIM,
        _require(weights, f"{prefix}.layer_norm2.weight"),
        _require(weights, f"{prefix}.layer_norm2.bias"),
        1.0e-6, dtype=work_np_dtype, fp32_compute=True)
    mlp = _linear(
        network, normed,
        _require(weights, f"{prefix}.mlp.fc1.weight"),
        _require(weights, f"{prefix}.mlp.fc1.bias"),
        work_np_dtype)
    mlp = _gelu_tanh(network, mlp)
    mlp = _linear(
        network, mlp,
        _require(weights, f"{prefix}.mlp.fc2.weight"),
        _require(weights, f"{prefix}.mlp.fc2.bias"),
        work_np_dtype)
    return network.add_elementwise(
        hidden, mlp, trt.ElementWiseOperation.SUM).get_output(0)


def _pool_and_gather_image_tokens(network, hidden, weights, token_rows,
                                  work_np_dtype, work_trt_dtype):
    """AvgPool2d(2) every entry to 16 x 16, then gather the sub_glb image tokens.

    ``token_rows`` index ``[sub_GN, glb_GN, pooled features of every entry]``;
    the runtime derives them from the image's crop grid and padding mask.
    """
    image_grid = network.add_shuffle(hidden)
    image_grid.reshape_dims = (0, _GRID_SIZE, _GRID_SIZE, _EMBED_DIM)
    image_grid.second_transpose = trt.Permutation([0, 3, 1, 2])
    pool = network.add_pooling_nd(
        image_grid.get_output(0), trt.PoolingType.AVERAGE, (2, 2))
    pool.stride_nd = (2, 2)
    pooled = network.add_shuffle(pool.get_output(0))
    pooled.first_transpose = trt.Permutation([0, 2, 3, 1])
    pooled.reshape_dims = (-1, _EMBED_DIM)

    separators = const_in_work_dtype(
        network, (2, _EMBED_DIM),
        np.concatenate([
            _require(weights, "sub_GN").reshape(1, _EMBED_DIM),
            _require(weights, "glb_GN").reshape(1, _EMBED_DIM),
        ]),
        work_np_dtype, work_trt_dtype)
    table = network.add_concatenation([separators, pooled.get_output(0)])
    table.axis = 0
    return network.add_gather(table.get_output(0), token_rows, 0).get_output(0)


def _navit_positions_and_key_mask(network, crop_valid, work_np_dtype,
                                  work_trt_dtype):
    """Per-entry NaViT position ids and the additive key mask of padded patches.

    ``crop_valid`` is ``[entries, 2]`` int32 valid patch rows/cols; padded
    patches keep position 0, as in SiglipVisionEmbeddings.
    """
    def const(values):
        values = np.ascontiguousarray(values, dtype=np.int32)
        return graph_ops.add_constant(network, values.shape, values, dtype=np.int32)

    def gather(data, indices, axis):
        return network.add_gather(data, indices, axis).get_output(0)

    def rows(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (-1, _NUM_PATCHES)
        return layer.get_output(0)

    def binary(lhs, rhs, operation):
        return network.add_elementwise(lhs, rhs, operation).get_output(0)

    patch = np.arange(_NUM_PATCHES)
    buckets = const(navit_bucket_table())
    ids, valid = [], None
    for axis, coordinate in enumerate((patch // _GRID_SIZE, patch % _GRID_SIZE)):
        count = gather(crop_valid, const(np.array([axis])), 1)  # [entries, 1]
        bucket = gather(gather(buckets, count, 0), const(coordinate), 2)  # [entries, 1, 1024]
        ids.append(rows(bucket))
        inside = binary(const(coordinate.reshape(1, -1)), count, trt.ElementWiseOperation.LESS)
        valid = inside if valid is None else binary(valid, inside, trt.ElementWiseOperation.AND)
    positions = binary(
        binary(ids[0], const(np.array([[_GRID_SIZE]])), trt.ElementWiseOperation.PROD),
        ids[1], trt.ElementWiseOperation.SUM)
    positions = binary(
        positions, network.add_cast(valid, trt.int32).get_output(0),
        trt.ElementWiseOperation.PROD)

    masked = const_in_work_dtype(
        network, (1, 1), np.full((1, 1), _MASKED_KEY), work_np_dtype, work_trt_dtype)
    zero = const_in_work_dtype(
        network, (1, 1), np.zeros((1, 1)), work_np_dtype, work_trt_dtype)
    key_mask = network.add_shuffle(network.add_select(valid, zero, masked).get_output(0))
    key_mask.reshape_dims = (-1, 1, 1, _NUM_PATCHES)
    return positions, key_mask.get_output(0)


def build_phi4mm_vision_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build the Phi-4 Dynamic-HD vision engine.

    Inputs are the global view followed by the image's 448px crops
    ``pixel_values[entries, 3, 448, 448]`` (2..37 entries), their valid patch
    rows/cols ``crop_valid[entries, 2]`` and the image-token rows
    ``image_token_rows[tokens]``; the output is ``image_features[tokens, 3072]``.
    """
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Phi-4 vision precision must be fp32 or fp16, got {precision}")
    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32

    context = create_builder_context(
        verbose=verbose, workspace_bytes=8 << 30, disable_tf32=True)
    builder, network, config = context.builder, context.network, context.config
    pixel_values = network.add_input(
        "pixel_values", trt.float32, (-1, 3, _CROP_SIZE, _CROP_SIZE))
    crop_valid = network.add_input("crop_valid", trt.int32, (-1, 2))
    token_rows = network.add_input("image_token_rows", trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values", (2, 3, _CROP_SIZE, _CROP_SIZE),
        (_OPT_ENTRIES, 3, _CROP_SIZE, _CROP_SIZE), (MAX_ENTRIES, 3, _CROP_SIZE, _CROP_SIZE))
    profile.set_shape("crop_valid", (2, 2), (_OPT_ENTRIES, 2), (MAX_ENTRIES, 2))
    profile.set_shape(
        "image_token_rows", (MIN_IMAGE_TOKENS,), (_OPT_IMAGE_TOKENS,), (MAX_IMAGE_TOKENS,))
    config.add_optimization_profile(profile)

    patch_input = _cast(network, pixel_values, work_trt_dtype)
    patch_np_dtype = work_np_dtype
    if precision == "fp16":
        patch_input = _cast(network, patch_input, trt.float32)
        patch_np_dtype = np.float32
    patch_weight = np.asarray(
        _require(weights, "img_processor.embeddings.patch_embedding.weight"),
        dtype=patch_np_dtype)
    patch_bias = np.asarray(
        _require(weights, "img_processor.embeddings.patch_embedding.bias"),
        dtype=patch_np_dtype)
    patch = network.add_convolution_nd(
        patch_input, _EMBED_DIM, (_PATCH_SIZE, _PATCH_SIZE),
        trt.Weights(np.ascontiguousarray(patch_weight)),
        trt.Weights(np.ascontiguousarray(patch_bias)))
    patch.stride_nd = (_PATCH_SIZE, _PATCH_SIZE)
    rows = network.add_shuffle(patch.get_output(0))
    rows.first_transpose = trt.Permutation([0, 2, 3, 1])
    rows.reshape_dims = (0, _NUM_PATCHES, _EMBED_DIM)
    patch_rows = _cast(network, rows.get_output(0), work_trt_dtype)

    positions, attention_mask = _navit_positions_and_key_mask(
        network, crop_valid, work_np_dtype, work_trt_dtype)
    position_table = np.asarray(
        _require(weights, "img_processor.embeddings.position_embedding.weight"),
        dtype=np.float32)
    if position_table.shape != (_NUM_PATCHES, _EMBED_DIM):
        raise ValueError(
            f"Unexpected Phi-4 SigLIP position embedding shape: {position_table.shape}")
    position = network.add_gather(
        const_in_work_dtype(network, position_table.shape, position_table,
                            work_np_dtype, work_trt_dtype),
        positions, 0).get_output(0)
    hidden = network.add_elementwise(
        patch_rows, position, trt.ElementWiseOperation.SUM).get_output(0)
    for layer_idx in range(_NUM_EXECUTED_LAYERS):
        hidden = _build_encoder_layer(
            network, hidden, weights, layer_idx, attention_mask,
            work_np_dtype)

    hidden = _pool_and_gather_image_tokens(
        network, hidden, weights, token_rows, work_np_dtype, work_trt_dtype)
    hidden = _linear(
        network, hidden,
        _require(weights, "img_projection.0.weight"),
        _require(weights, "img_projection.0.bias"),
        work_np_dtype)
    hidden = graph_ops.add_gelu_erf(network, hidden, dtype=work_np_dtype)
    hidden = _linear(
        network, hidden,
        _require(weights, "img_projection.2.weight"),
        _require(weights, "img_projection.2.bias"),
        work_np_dtype)
    hidden = _cast(network, hidden, trt.float32)
    if tuple(hidden.shape)[1:] != (_TEXT_HIDDEN,):
        raise RuntimeError(
            f"Unexpected Phi-4 vision output shape: {tuple(hidden.shape)}")
    hidden.name = "image_features"
    network.mark_output(hidden)

    if verbose:
        print(
            "[trtmc build] Building Phi-4 Dynamic-HD vision engine "
            f"({_NUM_EXECUTED_LAYERS} SigLIP layers, up to {MAX_ENTRIES} entries / "
            f"{MAX_IMAGE_TOKENS} image tokens, precision={precision}) ...",
            file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT Phi-4 Dynamic-HD vision engine build failed")
    return bytes(plan)
