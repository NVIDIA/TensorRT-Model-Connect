# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT vision encoder for Phi-4-multimodal's Dynamic-HD image path."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .utils import const_in_work_dtype, create_builder_context

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


_CROP_SIZE = 448
_PATCH_SIZE = 14
_GRID_SIZE = _CROP_SIZE // _PATCH_SIZE
_NUM_CROPS = 3  # global crop followed by the two 2x1 Dynamic-HD tiles
_EMBED_DIM = 1152
_MLP_DIM = 4304
_NUM_HEADS = 16
_HEAD_DIM = _EMBED_DIM // _NUM_HEADS
_NUM_EXECUTED_LAYERS = 26  # hidden_states[-2] from the 27-layer SigLIP tower
_POOLED_GRID = _GRID_SIZE // 2
_SECOND_CROP_VALID_PATCH_COLS = 22
_SUB_IMAGE_VALID_COLS = 27
_NUM_IMAGE_TOKENS = 721


def _require(weights: WeightDict, key: str) -> np.ndarray:
    value = weights.get(key)
    if value is None:
        raise RuntimeError(f"Missing Phi-4 vision weight: {key}")
    return np.asarray(value)


def _position_ids(valid_cols: int) -> np.ndarray:
    """Match SiglipVisionEmbeddings' NaViT position remapping."""
    ids = np.zeros((_GRID_SIZE, _GRID_SIZE), dtype=np.int64)
    row_ids = np.floor(
        np.arange(_GRID_SIZE, dtype=np.float64) * _GRID_SIZE / _GRID_SIZE
    ).astype(np.int64)
    col_ids = np.floor(
        np.arange(valid_cols, dtype=np.float64) * _GRID_SIZE / valid_cols
    ).astype(np.int64)
    ids[:, :valid_cols] = row_ids[:, None] * _GRID_SIZE + col_ids[None, :]
    return ids.reshape(-1)


def _static_position_embeddings(position_weight: np.ndarray) -> np.ndarray:
    position_weight = np.asarray(position_weight, dtype=np.float32)
    if position_weight.shape != (_GRID_SIZE * _GRID_SIZE, _EMBED_DIM):
        raise ValueError(
            "Unexpected Phi-4 SigLIP position embedding shape: "
            f"{position_weight.shape}")
    position_ids = np.stack([
        _position_ids(_GRID_SIZE),
        _position_ids(_GRID_SIZE),
        _position_ids(_SECOND_CROP_VALID_PATCH_COLS),
    ])
    return position_weight[position_ids]


def _static_attention_mask() -> np.ndarray:
    """Additive key mask for the padded columns in the second HD tile."""
    valid = np.ones((_NUM_CROPS, _GRID_SIZE, _GRID_SIZE), dtype=bool)
    valid[2, :, _SECOND_CROP_VALID_PATCH_COLS:] = False
    key_mask = valid.reshape(_NUM_CROPS, -1)
    mask = np.zeros(
        (_NUM_CROPS, 1, _GRID_SIZE * _GRID_SIZE, _GRID_SIZE * _GRID_SIZE),
        dtype=np.float32,
    )
    mask[~key_mask[:, None, None, :].repeat(_GRID_SIZE * _GRID_SIZE, axis=2)] = -1.0e4
    return mask


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
        layer.reshape_dims = (_NUM_CROPS, _GRID_SIZE * _GRID_SIZE,
                              _NUM_HEADS, _HEAD_DIM)
        layer.second_transpose = trt.Permutation([0, 2, 1, 3])
        return layer.get_output(0)

    context = graph_ops.add_siglip_attention_core(
        network, as_heads(q), as_heads(k), as_heads(v),
        mask=attention_mask,
        scale=_HEAD_DIM ** -0.5,
    )
    context_rows = network.add_shuffle(context)
    context_rows.first_transpose = trt.Permutation([0, 2, 1, 3])
    context_rows.reshape_dims = (
        _NUM_CROPS, _GRID_SIZE * _GRID_SIZE, _EMBED_DIM)
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
    mlp = graph_ops.add_gelu_erf(network, mlp, dtype=work_np_dtype)
    mlp = _linear(
        network, mlp,
        _require(weights, f"{prefix}.mlp.fc2.weight"),
        _require(weights, f"{prefix}.mlp.fc2.bias"),
        work_np_dtype)
    return network.add_elementwise(
        hidden, mlp, trt.ElementWiseOperation.SUM).get_output(0)


def _pool_and_apply_hd_layout(network, hidden, weights, work_np_dtype,
                              work_trt_dtype):
    image_grid = network.add_shuffle(hidden)
    image_grid.reshape_dims = (
        _NUM_CROPS, _GRID_SIZE, _GRID_SIZE, _EMBED_DIM)
    image_grid.second_transpose = trt.Permutation([0, 3, 1, 2])
    pool = network.add_pooling_nd(
        image_grid.get_output(0), trt.PoolingType.AVERAGE, (2, 2))
    pool.stride_nd = (2, 2)
    pooled = network.add_shuffle(pool.get_output(0))
    pooled.first_transpose = trt.Permutation([0, 2, 3, 1])
    pooled.reshape_dims = (
        _NUM_CROPS, _POOLED_GRID, _POOLED_GRID, _EMBED_DIM)

    global_crop = network.add_slice(
        pooled.get_output(0), start=(0, 0, 0, 0),
        shape=(1, _POOLED_GRID, _POOLED_GRID, _EMBED_DIM),
        stride=(1, 1, 1, 1)).get_output(0)
    sub_crops = network.add_slice(
        pooled.get_output(0), start=(1, 0, 0, 0),
        shape=(2, _POOLED_GRID, _POOLED_GRID, _EMBED_DIM),
        stride=(1, 1, 1, 1)).get_output(0)

    global_rows = network.add_shuffle(global_crop)
    global_rows.reshape_dims = (
        _POOLED_GRID, _POOLED_GRID, _EMBED_DIM)
    sub_grid = network.add_shuffle(sub_crops)
    sub_grid.reshape_dims = (1, 2, _POOLED_GRID, _POOLED_GRID, _EMBED_DIM)
    sub_grid.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
    sub_rows_grid = network.add_shuffle(sub_grid.get_output(0))
    sub_rows_grid.reshape_dims = (
        _POOLED_GRID, 2 * _POOLED_GRID, _EMBED_DIM)
    valid_sub = network.add_slice(
        sub_rows_grid.get_output(0), start=(0, 0, 0),
        shape=(_POOLED_GRID, _SUB_IMAGE_VALID_COLS, _EMBED_DIM),
        stride=(1, 1, 1)).get_output(0)

    sub_separator = const_in_work_dtype(
        network, (_POOLED_GRID, 1, _EMBED_DIM),
        np.broadcast_to(
            _require(weights, "sub_GN").reshape(1, 1, _EMBED_DIM),
            (_POOLED_GRID, 1, _EMBED_DIM)),
        work_np_dtype, work_trt_dtype)
    global_separator = const_in_work_dtype(
        network, (1, _EMBED_DIM),
        _require(weights, "glb_GN").reshape(1, _EMBED_DIM),
        work_np_dtype, work_trt_dtype)

    sub_with_separators = network.add_concatenation([valid_sub, sub_separator])
    sub_with_separators.axis = 1
    sub_rows = network.add_shuffle(sub_with_separators.get_output(0))
    sub_rows.reshape_dims = (
        _POOLED_GRID * (_SUB_IMAGE_VALID_COLS + 1), _EMBED_DIM)
    global_with_separators = network.add_concatenation(
        [global_rows.get_output(0), sub_separator])
    global_with_separators.axis = 1
    global_flat = network.add_shuffle(global_with_separators.get_output(0))
    global_flat.reshape_dims = (
        _POOLED_GRID * (_POOLED_GRID + 1), _EMBED_DIM)

    hd = network.add_concatenation([
        sub_rows.get_output(0), global_separator, global_flat.get_output(0)])
    hd.axis = 0
    return hd.get_output(0)


def build_phi4mm_vision_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build the canonical Phi-4 Dynamic-HD vision engine.

    The engine is fixed to the 2x1 crop topology used by the checked-in E2E
    image. Its input is the concatenated global/left/right crop tensor
    ``[9, 448, 448]`` and its output is ``[721, 3072]``.
    """
    if precision not in {"fp32", "fp16"}:
        raise ValueError(f"Phi-4 vision precision must be fp32 or fp16, got {precision}")
    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32

    context = create_builder_context(
        verbose=verbose, workspace_bytes=8 << 30, disable_tf32=True)
    builder, network, config = context.builder, context.network, context.config
    pixel_values = network.add_input(
        "pixel_values", trt.float32, (9, _CROP_SIZE, _CROP_SIZE))
    work_input = _cast(network, pixel_values, work_trt_dtype)
    batched = network.add_shuffle(work_input)
    batched.reshape_dims = (_NUM_CROPS, 3, _CROP_SIZE, _CROP_SIZE)

    patch_input = batched.get_output(0)
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
    rows.reshape_dims = (
        _NUM_CROPS, _GRID_SIZE * _GRID_SIZE, _EMBED_DIM)
    patch_rows = _cast(network, rows.get_output(0), work_trt_dtype)

    position = const_in_work_dtype(
        network, (_NUM_CROPS, _GRID_SIZE * _GRID_SIZE, _EMBED_DIM),
        _static_position_embeddings(_require(
            weights, "img_processor.embeddings.position_embedding.weight")),
        work_np_dtype, work_trt_dtype)
    hidden = network.add_elementwise(
        patch_rows, position, trt.ElementWiseOperation.SUM).get_output(0)
    attention_mask = const_in_work_dtype(
        network,
        (_NUM_CROPS, 1, _GRID_SIZE * _GRID_SIZE, _GRID_SIZE * _GRID_SIZE),
        _static_attention_mask(), work_np_dtype, work_trt_dtype)
    for layer_idx in range(_NUM_EXECUTED_LAYERS):
        hidden = _build_encoder_layer(
            network, hidden, weights, layer_idx, attention_mask,
            work_np_dtype)

    hidden = _pool_and_apply_hd_layout(
        network, hidden, weights, work_np_dtype, work_trt_dtype)
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
    if tuple(hidden.shape) != (_NUM_IMAGE_TOKENS, 3072):
        raise RuntimeError(
            f"Unexpected Phi-4 vision output shape: {tuple(hidden.shape)}")
    hidden.name = "image_features"
    network.mark_output(hidden)

    if verbose:
        print(
            "[trtmc build] Building Phi-4 Dynamic-HD vision engine "
            f"({_NUM_EXECUTED_LAYERS} SigLIP layers, precision={precision}) ...",
            file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT Phi-4 Dynamic-HD vision engine build failed")
    return bytes(plan)
