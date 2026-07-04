# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 vision encoder TensorRT builder.

Builds the SAM3 image backbone plus the first three FPN neck outputs used by
the image PCS path.  The DETR/mask core engine consumes these named tensors
together with text prompt features.
"""

from __future__ import annotations

import sys

import numpy as np

from .checkpoint_mapper import WeightDict


def _trt():
    from tensorrt_model_connect import trt_compat

    return trt_compat.get_trt()


def _graph_ops():
    from . import graph_ops

    return graph_ops


def _tile_position_embeddings(
    position_embeddings: np.ndarray,
    *,
    pretrain_grid: int,
    grid_size: int,
    hidden_size: int,
) -> np.ndarray:
    pos = np.asarray(position_embeddings, dtype=np.float32).reshape(
        1, pretrain_grid, pretrain_grid, hidden_size)
    if pretrain_grid == grid_size:
        return np.ascontiguousarray(pos.reshape(grid_size * grid_size, hidden_size))
    repeat_h = grid_size // pretrain_grid + 1
    repeat_w = grid_size // pretrain_grid + 1
    tiled = np.tile(pos.transpose(0, 3, 1, 2), (1, 1, repeat_h, repeat_w))
    tiled = tiled[:, :, :grid_size, :grid_size]
    return np.ascontiguousarray(
        tiled.transpose(0, 2, 3, 1).reshape(grid_size * grid_size, hidden_size))


def _window_indices(grid_size: int, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if grid_size % window_size != 0:
        raise ValueError(
            f"SAM3 vision builder requires grid_size divisible by window_size, "
            f"got grid={grid_size}, window={window_size}")
    indices: list[int] = []
    for wy in range(0, grid_size, window_size):
        for wx in range(0, grid_size, window_size):
            for y in range(window_size):
                for x in range(window_size):
                    indices.append((wy + y) * grid_size + (wx + x))
    forward = np.asarray(indices, dtype=np.int32)
    inverse = np.empty_like(forward)
    inverse[forward] = np.arange(forward.size, dtype=np.int32)
    return forward, inverse


def _sam3_rope_table(end_x: int, end_y: int, head_dim: int, rope_theta: float,
                     scale: float) -> tuple[np.ndarray, np.ndarray]:
    if head_dim % 4 != 0:
        raise ValueError("SAM3 vision RoPE head_dim must be divisible by 4")
    freqs = 1.0 / (
        float(rope_theta) ** (np.arange(0, head_dim, 4, dtype=np.float32) / head_dim))
    flat = np.arange(end_x * end_y, dtype=np.int64)
    x_pos = (flat % end_x).astype(np.float32) * scale
    y_pos = (flat // end_x).astype(np.float32) * scale
    freqs_x = np.outer(x_pos, freqs)
    freqs_y = np.outer(y_pos, freqs)
    inv_freq = np.concatenate([freqs_x, freqs_y], axis=-1)
    inv_freq = np.repeat(inv_freq, 2, axis=-1).astype(np.float32)
    return np.cos(inv_freq).astype(np.float32), np.sin(inv_freq).astype(np.float32)


def _sam3_position_encoding(height: int, width: int, channels: int) -> np.ndarray:
    num_pos_feats = channels // 2
    y_embed = np.cumsum(np.ones((height, width), dtype=np.float32), axis=0)
    x_embed = np.cumsum(np.ones((height, width), dtype=np.float32), axis=1)
    scale = 2.0 * np.pi
    eps = 1e-6
    y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, -1:] + eps) * scale
    dim_t = np.arange(num_pos_feats, dtype=np.float32)
    dim_t = 10000.0 ** (2 * np.floor(dim_t / 2) / num_pos_feats)
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = np.stack((np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])),
                     axis=3).reshape(height, width, -1)
    pos_y = np.stack((np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])),
                     axis=3).reshape(height, width, -1)
    pos = np.concatenate((pos_y, pos_x), axis=2)
    return np.ascontiguousarray(pos.transpose(2, 0, 1)[None, :, :, :].astype(np.float32))


def _add_attention_with_rope(
    network,
    hidden,
    weights: WeightDict,
    prefix: str,
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    num_windows: int | None = None,
    dtype=np.float32,
):
    trt = _trt()
    graph_ops = _graph_ops()
    q = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size,
        weights[f"{prefix}.attention.q_proj.weight"])
    q = graph_ops.add_bias_sum(
        network, q, hidden_size, weights[f"{prefix}.attention.q_proj.bias"])
    k = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size,
        weights[f"{prefix}.attention.k_proj.weight"])
    k = graph_ops.add_bias_sum(
        network, k, hidden_size, weights[f"{prefix}.attention.k_proj.bias"])
    v = graph_ops.add_matmul_rhs_constant(
        network, hidden, hidden_size, hidden_size,
        weights[f"{prefix}.attention.v_proj.weight"])
    v = graph_ops.add_bias_sum(
        network, v, hidden_size, weights[f"{prefix}.attention.v_proj.bias"])

    cos = graph_ops.add_constant(
        network, (1, seq_len, head_dim // 2),
        cos_table[:, 0::2].reshape(1, seq_len, -1), dtype=dtype)
    sin = graph_ops.add_constant(
        network, (1, seq_len, head_dim // 2),
        sin_table[:, 0::2].reshape(1, seq_len, -1), dtype=dtype)
    q = graph_ops.add_apply_rope_native_sequence(
        network, q, num_heads, head_dim, cos, sin,
        rotary_embedding_dim=head_dim, interleaved=True, sequence_length=seq_len)
    k = graph_ops.add_apply_rope_native_sequence(
        network, k, num_heads, head_dim, cos, sin,
        rotary_embedding_dim=head_dim, interleaved=True, sequence_length=seq_len)

    if num_windows is None:
        context = graph_ops.add_attention_from_rows(
            network, q, k, v, num_heads=num_heads, head_dim=head_dim,
            q_seq=seq_len, kv_seq=seq_len)
    else:
        win_seq = seq_len // num_windows
        q_win = network.add_shuffle(q)
        q_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        q_win.second_transpose = trt.Permutation([0, 2, 1, 3])
        k_win = network.add_shuffle(k)
        k_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        k_win.second_transpose = trt.Permutation([0, 2, 1, 3])
        v_win = network.add_shuffle(v)
        v_win.reshape_dims = (num_windows, win_seq, num_heads, head_dim)
        v_win.second_transpose = trt.Permutation([0, 2, 1, 3])
        ctx = graph_ops.add_attention_core(
            network, q_win.get_output(0), k_win.get_output(0), v_win.get_output(0))
        flat = network.add_shuffle(ctx)
        flat.first_transpose = trt.Permutation([0, 2, 1, 3])
        flat.reshape_dims = (seq_len, hidden_size)
        context = flat.get_output(0)

    out = graph_ops.add_matmul_rhs_constant(
        network, context, hidden_size, hidden_size,
        weights[f"{prefix}.attention.o_proj.weight"])
    return graph_ops.add_bias_sum(
        network, out, hidden_size, weights[f"{prefix}.attention.o_proj.bias"])


def _add_deconv2d(network, inp, weight: np.ndarray, bias: np.ndarray,
                  out_channels: int, dtype=np.float32):
    trt = _trt()
    layer = network.add_deconvolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=(2, 2),
        kernel=trt.Weights(np.ascontiguousarray(weight, dtype=dtype)),
        bias=trt.Weights(np.ascontiguousarray(bias, dtype=dtype)),
    )
    layer.stride_nd = (2, 2)
    return layer.get_output(0)


def _add_fpn_level(network, hidden_spatial, weights: WeightDict, level: int,
                   hidden_size: int, fpn_hidden_size: int,
                   dtype=np.float32):
    trt = _trt()
    graph_ops = _graph_ops()
    x = hidden_spatial
    if level == 0:
        x = _add_deconv2d(
            network, x, weights["vision.fpn.0.deconv0.weight"],
            weights["vision.fpn.0.deconv0.bias"], hidden_size // 2,
            dtype=dtype)
        x = graph_ops.add_gelu_erf(network, x)
        x = _add_deconv2d(
            network, x, weights["vision.fpn.0.deconv1.weight"],
            weights["vision.fpn.0.deconv1.bias"], hidden_size // 4,
            dtype=dtype)
    elif level == 1:
        x = _add_deconv2d(
            network, x, weights["vision.fpn.1.deconv0.weight"],
            weights["vision.fpn.1.deconv0.bias"], hidden_size // 2,
            dtype=dtype)

    prefix = f"vision.fpn.{level}"
    x = graph_ops.add_conv2d(
        network, x, weights[f"{prefix}.proj1.weight"],
        weights[f"{prefix}.proj1.bias"], fpn_hidden_size, (1, 1),
        dtype=dtype)
    x = graph_ops.add_conv2d(
        network, x, weights[f"{prefix}.proj2.weight"],
        weights[f"{prefix}.proj2.bias"], fpn_hidden_size, (3, 3),
        padding=(1, 1), dtype=dtype)
    cast = network.add_cast(x, trt.float32)
    out = cast.get_output(0)
    out.name = f"sam3_fpn_hidden_{level}"
    network.mark_output(out)
    return out


def _add_sam3_vision_activation(network, inp, hidden_act: str):
    graph_ops = _graph_ops()
    normalized = str(hidden_act).lower()
    if normalized == "gelu":
        return graph_ops.add_gelu_erf(network, inp)
    if normalized in {"gelu_new", "gelu_pytorch_tanh"}:
        return graph_ops.add_gelu_new(network, inp)
    return graph_ops.add_activation(network, inp, hidden_act)


def build_sam3_vision_encoder_engine(
    weights: WeightDict,
    *,
    image_size: int,
    patch_size: int,
    pretrain_image_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_heads: int,
    window_size: int,
    global_attn_indexes: list[int],
    fpn_hidden_size: int,
    rope_theta: float,
    eps: float,
    precision: str = "fp32",
    hidden_act: str = "gelu",
    verbose: bool = False,
) -> bytes:
    """Build the SAM3 ViT+FPN vision plan with TensorRT APIs."""
    trt = _trt()
    graph_ops = _graph_ops()
    grid_size = image_size // patch_size
    pretrain_grid = pretrain_image_size // patch_size
    seq_len = grid_size * grid_size
    head_dim = hidden_size // num_heads
    window_order, inverse_window_order = _window_indices(grid_size, window_size)
    window_seq = window_size * window_size
    num_windows = seq_len // window_seq
    global_layers = set(global_attn_indexes)
    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    pixel_values = network.add_input(
        "pixel_values", trt.float32, (1, 3, image_size, image_size))
    pixel_values_work = pixel_values
    if work_trt_dtype != trt.float32:
        pixel_values_work = network.add_cast(
            pixel_values, work_trt_dtype).get_output(0)
    patch_bias = weights.get(
        "vision.patch_embed.bias", np.zeros(hidden_size, dtype=np.float32))
    patch = graph_ops.add_conv2d(
        network,
        pixel_values_work,
        weights["vision.patch_embed.weight"],
        patch_bias,
        hidden_size,
        (patch_size, patch_size),
        stride=(patch_size, patch_size),
        dtype=work_np_dtype,
    )
    to_rows = network.add_shuffle(patch)
    to_rows.first_transpose = trt.Permutation([0, 2, 3, 1])
    to_rows.reshape_dims = (seq_len, hidden_size)

    pos = _tile_position_embeddings(
        weights["vision.position_embeddings"],
        pretrain_grid=pretrain_grid,
        grid_size=grid_size,
        hidden_size=hidden_size,
    )
    pos_t = graph_ops.add_constant(
        network, (seq_len, hidden_size), pos, dtype=work_np_dtype)
    hidden = network.add_elementwise(
        to_rows.get_output(0), pos_t, trt.ElementWiseOperation.SUM).get_output(0)
    hidden = graph_ops.add_layer_norm_native(
        network,
        hidden,
        hidden_size,
        weights["vision.pre_layer_norm.weight"],
        weights["vision.pre_layer_norm.bias"],
        eps,
    )

    gather_window = graph_ops.add_constant(
        network, (seq_len,), window_order, dtype=np.int32)
    gather_inverse = graph_ops.add_constant(
        network, (seq_len,), inverse_window_order, dtype=np.int32)
    window_cos, window_sin = _sam3_rope_table(
        window_size, window_size, head_dim, rope_theta, scale=1.0)
    window_cos = np.tile(window_cos, (num_windows, 1))
    window_sin = np.tile(window_sin, (num_windows, 1))
    global_cos, global_sin = _sam3_rope_table(
        grid_size, grid_size, head_dim, rope_theta,
        scale=float(window_size) / float(grid_size))

    for layer_idx in range(num_layers):
        prefix = f"vision.layers.{layer_idx}"
        normed = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.layer_norm1.weight"],
            weights[f"{prefix}.layer_norm1.bias"],
            eps,
        )
        if layer_idx in global_layers:
            attn = _add_attention_with_rope(
                network,
                normed,
                weights,
                prefix,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                cos_table=global_cos,
                sin_table=global_sin,
                dtype=work_np_dtype,
            )
        else:
            ordered = network.add_gather(normed, gather_window, axis=0).get_output(0)
            attn_ordered = _add_attention_with_rope(
                network,
                ordered,
                weights,
                prefix,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                cos_table=window_cos,
                sin_table=window_sin,
                num_windows=num_windows,
                dtype=work_np_dtype,
            )
            attn = network.add_gather(attn_ordered, gather_inverse, axis=0).get_output(0)

        hidden = network.add_elementwise(hidden, attn, trt.ElementWiseOperation.SUM).get_output(0)
        normed2 = graph_ops.add_layer_norm_native(
            network,
            hidden,
            hidden_size,
            weights[f"{prefix}.layer_norm2.weight"],
            weights[f"{prefix}.layer_norm2.bias"],
            eps,
        )
        mlp = graph_ops.add_matmul_rhs_constant(
            network, normed2, hidden_size, intermediate_size,
            weights[f"{prefix}.mlp.fc1.weight"])
        mlp = graph_ops.add_bias_sum(
            network, mlp, intermediate_size, weights[f"{prefix}.mlp.fc1.bias"])
        mlp = _add_sam3_vision_activation(network, mlp, hidden_act)
        mlp = graph_ops.add_matmul_rhs_constant(
            network, mlp, intermediate_size, hidden_size,
            weights[f"{prefix}.mlp.fc2.weight"])
        mlp = graph_ops.add_bias_sum(
            network, mlp, hidden_size, weights[f"{prefix}.mlp.fc2.bias"])
        hidden = network.add_elementwise(hidden, mlp, trt.ElementWiseOperation.SUM).get_output(0)

    spatial = network.add_shuffle(hidden)
    spatial.reshape_dims = (1, grid_size, grid_size, hidden_size)
    spatial_nchw = network.add_shuffle(spatial.get_output(0))
    spatial_nchw.first_transpose = trt.Permutation([0, 3, 1, 2])
    hidden_spatial = spatial_nchw.get_output(0)

    for level, scale in enumerate((4, 2, 1)):
        _add_fpn_level(
            network, hidden_spatial, weights, level, hidden_size,
            fpn_hidden_size, dtype=work_np_dtype)
        pos_np = _sam3_position_encoding(
            grid_size * scale, grid_size * scale, fpn_hidden_size)
        pos = graph_ops.add_constant(
            network, pos_np.shape, pos_np, dtype=work_np_dtype)
        pos = network.add_cast(pos, trt.float32).get_output(0)
        pos.name = f"sam3_fpn_position_{level}"
        network.mark_output(pos)

    if verbose:
        print(
            f"[sam3-vision-builder] Building TRT engine "
            f"(image={image_size}, hidden={hidden_size}, layers={num_layers}, "
            f"grid={grid_size}, fpn={fpn_hidden_size}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialization failed for SAM3 vision encoder")
    return bytes(plan)
