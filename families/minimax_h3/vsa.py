# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastH3 VSA geometry and TensorRT graph helpers.

The public VSA student is trained with segment-pure 64-token tiles.  Prefix
segments are text and audio; video rows are tiled in T/H/W order using a
``(4, 4, 4)`` window.  Geometry is kept here, inside the MiniMax-H3 family,
because it is part of that checkpoint's mathematical contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import MiniMaxH3Config


VSA_TILE_SHAPE = (4, 4, 4)
VSA_TILE_SIZE = math.prod(VSA_TILE_SHAPE)
VSA_SPARSITY = 0.9
# 1344x768 decodes from a 37x48x84 latent.  The DiT's (1,2,2) patching
# therefore packs video rows in a 37x24x42 T/H/W grid.
VSA_VIDEO_SHAPE = (37, 24, 42)
VSA_BYOK_KERNEL_NAME = "minimax_h3.vsa_sm100a"


@dataclass(frozen=True)
class VsaGeometry:
    tile_shape: tuple[int, int, int]
    tile_size: int
    num_prefix_tiles: int
    num_video_tiles: int
    topk_video_tiles: int
    gather_indices: np.ndarray
    untile_indices: np.ndarray
    variable_block_sizes: np.ndarray

    @property
    def num_tiles(self) -> int:
        return self.num_prefix_tiles + self.num_video_tiles

    @property
    def padded_rows(self) -> int:
        return self.num_tiles * self.tile_size


@dataclass(frozen=True)
class VsaGraphContext:
    geometry: VsaGeometry
    variable_block_sizes: object


def _segment_tiles(start: int, rows: int, *, sentinel: int) -> tuple[list[int], list[int]]:
    indices: list[int] = []
    sizes: list[int] = []
    for offset in range(0, rows, VSA_TILE_SIZE):
        valid = min(VSA_TILE_SIZE, rows - offset)
        sizes.append(valid)
        indices.extend(range(start + offset, start + offset + valid))
        indices.extend([sentinel] * (VSA_TILE_SIZE - valid))
    return indices, sizes


def _video_tiles(start: int, shape: tuple[int, int, int], *, sentinel: int):
    t_rows, h_rows, w_rows = shape
    tile_t, tile_h, tile_w = VSA_TILE_SHAPE
    indices: list[int] = []
    sizes: list[int] = []
    for tile_t_index in range(math.ceil(t_rows / tile_t)):
        for tile_h_index in range(math.ceil(h_rows / tile_h)):
            for tile_w_index in range(math.ceil(w_rows / tile_w)):
                live: list[int] = []
                for t in range(tile_t_index * tile_t, min((tile_t_index + 1) * tile_t, t_rows)):
                    for h in range(
                        tile_h_index * tile_h,
                        min((tile_h_index + 1) * tile_h, h_rows),
                    ):
                        for w in range(
                            tile_w_index * tile_w,
                            min((tile_w_index + 1) * tile_w, w_rows),
                        ):
                            live.append(start + (t * h_rows + h) * w_rows + w)
                sizes.append(len(live))
                indices.extend(live)
                indices.extend([sentinel] * (VSA_TILE_SIZE - len(live)))
    return indices, sizes


def make_vsa_geometry(profile: MiniMaxH3Config) -> VsaGeometry:
    """Return the exact fixed-max FastH3 tile transport geometry.

    Text is represented by its engine maximum here. At runtime its per-tile
    valid sizes are replaced with values derived from the actual prompt
    length; the fixed layout keeps audio/video offsets and the plugin ABI
    stable while zero-sized text tiles remain masked by the sparse kernel.
    """

    if profile.video_rows != math.prod(VSA_VIDEO_SHAPE):
        raise ValueError("FastH3 VSA requires video rows shaped as 37x24x42")
    sentinel = profile.sequence_length
    text_indices, text_sizes = _segment_tiles(0, profile.text_rows, sentinel=sentinel)
    audio_indices, audio_sizes = _segment_tiles(
        profile.text_rows,
        profile.audio_rows,
        sentinel=sentinel,
    )
    video_indices, video_sizes = _video_tiles(
        profile.text_rows + profile.audio_rows,
        VSA_VIDEO_SHAPE,
        sentinel=sentinel,
    )
    gather = np.asarray(text_indices + audio_indices + video_indices, dtype=np.int32)
    sizes = np.asarray(text_sizes + audio_sizes + video_sizes, dtype=np.int32)
    live_slots = np.flatnonzero(gather != sentinel)
    live_sources = gather[live_slots]
    untile = np.empty(profile.sequence_length, dtype=np.int32)
    untile[live_sources] = live_slots

    prefix_tiles = len(text_sizes) + len(audio_sizes)
    video_tiles = len(video_sizes)
    geometry = VsaGeometry(
        tile_shape=VSA_TILE_SHAPE,
        tile_size=VSA_TILE_SIZE,
        num_prefix_tiles=prefix_tiles,
        num_video_tiles=video_tiles,
        topk_video_tiles=max(1, math.ceil((1.0 - VSA_SPARSITY) * video_tiles)),
        gather_indices=gather,
        untile_indices=untile,
        variable_block_sizes=sizes,
    )
    if geometry.num_tiles % 2:
        raise ValueError("FastH3 sm100a requires an even tile count")
    if geometry.variable_block_sizes.sum() != profile.sequence_length:
        raise ValueError("FastH3 VSA tile geometry does not cover the packed sequence")
    return geometry


def _shape_dim(network, tensor, axis: int):
    shape = network.add_shape(tensor).get_output(0)
    return network.add_slice(shape, (axis,), (1,), (1,)).get_output(0)


def _shape_vector(network, parts):
    from . import graph_ops as op

    tensors = [
        op.constant(network, np.asarray([part], dtype=np.int64), dtype=np.int64)
        if isinstance(part, int)
        else part
        for part in parts
    ]
    if len(tensors) == 1:
        return tensors[0]
    layer = network.add_concatenation(tensors)
    layer.axis = 0
    return layer.get_output(0)


def _dynamic_slice(network, tensor, starts, sizes):
    layer = network.add_slice(
        tensor,
        tuple(0 for _ in starts),
        tuple(1 if not isinstance(size, int) else size for size in sizes),
        tuple(1 for _ in starts),
    )
    layer.set_input(1, _shape_vector(network, starts))
    layer.set_input(2, _shape_vector(network, sizes))
    return layer.get_output(0)


def pad_text_rows(network, tensor, max_text_rows: int):
    """Append zero rows to a dynamic text tensor up to the engine maximum."""

    import tensorrt as trt

    from . import graph_ops as op

    rows = _shape_dim(network, tensor, 0)
    maximum = op.constant(network, np.asarray([max_text_rows], dtype=np.int64), dtype=np.int64)
    padding_rows = network.add_elementwise(
        maximum,
        rows,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    width = int(tensor.shape[1])
    zeros = op.constant(network, np.zeros((max_text_rows, width), dtype=np.float32))
    zeros = op.cast(network, zeros, tensor.dtype)
    padding = _dynamic_slice(network, zeros, (0, 0), (padding_rows, width))
    concat = network.add_concatenation((tensor, padding))
    concat.axis = 0
    return concat.get_output(0)


def pad_packed_text_segment(network, tensor, profile: MiniMaxH3Config):
    """Pad only the leading text segment of ``[text | audio | video]``."""

    import tensorrt as trt

    from . import graph_ops as op

    rank = len(tuple(tensor.shape))
    total_rows = _shape_dim(network, tensor, 0)
    media_rows = profile.audio_rows + profile.video_rows
    if rank == 1:
        media_start = network.add_elementwise(
            total_rows,
            op.constant(network, np.asarray([media_rows], dtype=np.int64), dtype=np.int64),
            trt.ElementWiseOperation.SUB,
        ).get_output(0)
        media = _dynamic_slice(network, tensor, (media_start,), (media_rows,))
    elif rank == 2:
        media = op.slice_rows_from_end(network, tensor, offset=media_rows, rows=media_rows)
    else:
        raise ValueError("FastH3 VSA packed padding supports rank-1 or rank-2 tensors")
    text_rows = network.add_elementwise(
        total_rows,
        op.constant(network, np.asarray([media_rows], dtype=np.int64), dtype=np.int64),
        trt.ElementWiseOperation.SUB,
    ).get_output(0)
    width = int(tensor.shape[1]) if rank == 2 else None
    if rank == 1:
        text = _dynamic_slice(network, tensor, (0,), (text_rows,))
        maximum = op.constant(
            network,
            np.asarray([profile.text_rows], dtype=np.int64),
            dtype=np.int64,
        )
        padding_rows = network.add_elementwise(
            maximum,
            text_rows,
            trt.ElementWiseOperation.SUB,
        ).get_output(0)
        zeros = op.constant(network, np.zeros((profile.text_rows,), dtype=np.int32), dtype=np.int32)
        zeros = op.cast(network, zeros, tensor.dtype)
        padding = _dynamic_slice(network, zeros, (0,), (padding_rows,))
    elif rank == 2:
        text = _dynamic_slice(network, tensor, (0, 0), (text_rows, width))
        maximum = op.constant(
            network,
            np.asarray([profile.text_rows], dtype=np.int64),
            dtype=np.int64,
        )
        padding_rows = network.add_elementwise(
            maximum,
            text_rows,
            trt.ElementWiseOperation.SUB,
        ).get_output(0)
        zeros = op.constant(network, np.zeros((profile.text_rows, width), dtype=np.float32))
        zeros = op.cast(network, zeros, tensor.dtype)
        padding = _dynamic_slice(network, zeros, (0, 0), (padding_rows, width))
    concat = network.add_concatenation((text, padding, media))
    concat.axis = 0
    return concat.get_output(0)


def _runtime_variable_block_sizes(
    network,
    text,
    geometry: VsaGeometry,
    profile: MiniMaxH3Config,
):
    import tensorrt as trt

    from . import graph_ops as op

    text_rows = op.cast(network, _shape_dim(network, text, 0), trt.int32)
    zero = op.constant(network, np.asarray([0], dtype=np.int32), dtype=np.int32)
    tile = op.constant(
        network,
        np.asarray([geometry.tile_size], dtype=np.int32),
        dtype=np.int32,
    )
    text_sizes = []
    text_tiles = math.ceil(profile.text_rows / geometry.tile_size)
    for index in range(text_tiles):
        offset = op.constant(
            network,
            np.asarray([index * geometry.tile_size], dtype=np.int32),
            dtype=np.int32,
        )
        remaining = network.add_elementwise(
            text_rows,
            offset,
            trt.ElementWiseOperation.SUB,
        ).get_output(0)
        remaining = network.add_elementwise(
            remaining,
            zero,
            trt.ElementWiseOperation.MAX,
        ).get_output(0)
        text_sizes.append(
            network.add_elementwise(
                remaining,
                tile,
                trt.ElementWiseOperation.MIN,
            ).get_output(0)
        )
    text_layer = network.add_concatenation(text_sizes)
    text_layer.axis = 0
    fixed = op.constant(
        network,
        geometry.variable_block_sizes[text_tiles:],
        dtype=np.int32,
    )
    combined = network.add_concatenation((text_layer.get_output(0), fixed))
    combined.axis = 0
    return combined.get_output(0)


def prepare_vsa_graph(network, text, profile: MiniMaxH3Config) -> VsaGraphContext:
    geometry = make_vsa_geometry(profile)
    return VsaGraphContext(
        geometry=geometry,
        variable_block_sizes=_runtime_variable_block_sizes(network, text, geometry, profile),
    )


def _tile_heads(network, tensor, geometry: VsaGeometry, profile: MiniMaxH3Config):
    import tensorrt as trt

    from . import graph_ops as op

    shaped = network.add_shuffle(tensor)
    shaped.reshape_dims = (profile.sequence_length, profile.num_heads, profile.head_dim)
    zero = op.constant(
        network, np.zeros((1, profile.num_heads, profile.head_dim), dtype=np.float32)
    )
    zero = op.cast(network, zero, tensor.dtype)
    extended = network.add_concatenation((shaped.get_output(0), zero))
    extended.axis = 0
    indices = op.constant(network, geometry.gather_indices, dtype=np.int32)
    gathered = network.add_gather(extended.get_output(0), indices, 0).get_output(0)
    transpose = network.add_shuffle(gathered)
    transpose.first_transpose = trt.Permutation([1, 0, 2])
    transpose.reshape_dims = (1, profile.num_heads, geometry.padded_rows, profile.head_dim)
    return transpose.get_output(0)


def vsa_workspace_bytes(geometry: VsaGeometry, profile: MiniMaxH3Config) -> int:
    """Workspace for FP32 pooling/scores plus the sparse block map."""

    pool_bytes = profile.num_heads * geometry.num_tiles * profile.head_dim * 4
    score_bytes = profile.num_heads * geometry.num_tiles * geometry.num_tiles * 4
    count_bytes = profile.num_heads * geometry.num_tiles * 4
    return 3 * pool_bytes + 2 * score_bytes + count_bytes


def sparse_attention(
    network,
    q,
    k,
    v,
    gate,
    context: VsaGraphContext,
    profile: MiniMaxH3Config,
    *,
    name: str,
):
    """Add the complete trained VSA operation through one BYOK boundary."""

    from tensorrt_model_connect.byok import add_kernel

    from . import graph_ops as op

    geometry = context.geometry
    q_tiled = _tile_heads(network, q, geometry, profile)
    k_tiled = _tile_heads(network, k, geometry, profile)
    v_tiled = _tile_heads(network, v, geometry, profile)
    gate_tiled = _tile_heads(network, gate, geometry, profile)
    (combined,) = add_kernel(
        network,
        kernel_name=VSA_BYOK_KERNEL_NAME,
        inputs=[q_tiled, k_tiled, v_tiled, gate_tiled, context.variable_block_sizes],
        output_specs=[{"dims": "same_as_input_0", "dtype": "bfloat16"}],
        workspace_bytes=vsa_workspace_bytes(geometry, profile),
    )
    combined.name = f"{name}.output"
    untile = op.constant(network, geometry.untile_indices, dtype=np.int32)
    packed = network.add_gather(combined, untile, 2).get_output(0)
    return op.heads_to_rows(
        network,
        packed,
        profile.sequence_length,
        profile.attention_size,
    )
