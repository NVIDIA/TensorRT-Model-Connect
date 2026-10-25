# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniRef's 128-token mixed-reference transport and TVM-FFI graph boundary.

Dense segments stay segment-pure. Every reference video and the final target
video has a separate sparse-key budget. This is distinct from the old T2V
student's 64-token, single-region attention recipe.
"""

from dataclasses import dataclass
import math

import numpy as np


TILE_SHAPE = (4, 4, 8)
TILE_SIZE = 128
SPARSITY = 0.9
KERNEL_NAME = "minimax_h3.omniref_vsa128"
MAX_REGION_TILES = 1024
MAX_TILES = 8192
MAX_REGIONS = 64


@dataclass(frozen=True)
class ReferenceVsaLayout:
    source_rows: int
    prefix_tiles: int
    logical_tiles: int
    gather_indices: np.ndarray
    untile_indices: np.ndarray
    block_sizes: np.ndarray
    # Each row is [start_tile, end_tile, retained_tiles], in packed video order.
    regions: np.ndarray

    @property
    def padded_rows(self):
        return int(self.block_sizes.size) * TILE_SIZE


def make_layout(packed_segments):
    """Accept dense row counts and raw latent video grids in packed order.

    Video grids use the checkpoint's (1, 2, 2) patch size. Empty dense segments
    contribute no tiles. The final video region is the generated target. An
    odd logical tile count receives one zero-valid partner for the native
    block-sparse kernel; that partner never enters the learned key budget.
    """
    if not isinstance(packed_segments, tuple) or not packed_segments:
        raise ValueError("OmniRef VSA requires an explicit ordered segment tuple")
    dense, videos = [], []
    cursor = 0
    for segment in packed_segments:
        if type(segment) is int:
            if segment < 0:
                raise ValueError("OmniRef dense segment lengths must be nonnegative")
            if cursor + segment >= np.iinfo(np.int32).max:
                raise ValueError("OmniRef packed rows exceed the native index range")
            if len(dense) + sum(map(len, videos)) + math.ceil(segment / TILE_SIZE) > MAX_TILES:
                raise ValueError("OmniRef geometry exceeds the VSA tile profile")
            for offset in range(0, segment, TILE_SIZE):
                dense.append(
                    list(range(cursor + offset, cursor + min(offset + TILE_SIZE, segment)))
                )
            cursor += segment
        elif isinstance(segment, tuple) and len(segment) == 3:
            if any(type(n) is not int or n <= 0 for n in segment):
                raise ValueError("OmniRef video latent dimensions must be positive integers")
            if segment[1] % 2 or segment[2] % 2:
                raise ValueError("OmniRef video regions require the released (1,2,2) patching")
            grid = (segment[0], segment[1] // 2, segment[2] // 2)
            tile_count = math.prod(math.ceil(n / tile) for n, tile in zip(grid, TILE_SHAPE))
            if tile_count > MAX_REGION_TILES:
                raise ValueError("OmniRef video region exceeds the VSA routing profile")
            if len(videos) >= MAX_REGIONS:
                raise ValueError("OmniRef geometry exceeds the VSA region profile")
            if len(dense) + sum(map(len, videos)) + tile_count > MAX_TILES:
                raise ValueError("OmniRef geometry exceeds the VSA tile profile")
            if cursor + math.prod(grid) >= np.iinfo(np.int32).max:
                raise ValueError("OmniRef packed rows exceed the native index range")
            region = []
            for t in range(0, grid[0], TILE_SHAPE[0]):
                for h in range(0, grid[1], TILE_SHAPE[1]):
                    for w in range(0, grid[2], TILE_SHAPE[2]):
                        region.append(
                            [
                                cursor + (i * grid[1] + j) * grid[2] + k
                                for i in range(t, min(t + TILE_SHAPE[0], grid[0]))
                                for j in range(h, min(h + TILE_SHAPE[1], grid[1]))
                                for k in range(w, min(w + TILE_SHAPE[2], grid[2]))
                            ]
                        )
            if len(region) > MAX_REGION_TILES:
                raise ValueError("OmniRef video region exceeds the VSA routing profile")
            videos.append(region)
            cursor += math.prod(grid)
        else:
            raise ValueError("OmniRef segments must be integer dense rows or latent triples")
        if cursor >= np.iinfo(np.int32).max:
            raise ValueError("OmniRef packed rows exceed the native index range")
    if not videos:
        raise ValueError("OmniRef VSA requires a target video region")
    tiles = list(dense)
    spans = []
    for region in videos:
        start = len(tiles)
        tiles.extend(region)
        # Match the released compute_topk expression, including its rounding.
        keep = max(1, min(math.ceil((1.0 - SPARSITY) * len(region)), len(region)))
        spans.append((start, len(tiles), keep))
    logical = len(tiles)
    if logical % 2:
        tiles.append([])
    gather = np.full((len(tiles), TILE_SIZE), cursor, dtype=np.int32)
    sizes = np.asarray([len(tile) for tile in tiles], dtype=np.int32)
    for i, tile in enumerate(tiles):
        gather[i, : len(tile)] = tile
    gather = gather.reshape(-1)
    slots = np.flatnonzero(gather != cursor)
    if slots.size != cursor or np.unique(gather[slots]).size != cursor:
        raise ValueError("OmniRef tiles must cover each original row exactly once")
    untile = np.empty(cursor, dtype=np.int32)
    untile[gather[slots]] = slots
    for array in (gather, sizes, untile):
        array.flags.writeable = False
    regions = np.asarray(spans, dtype=np.int32)
    regions.flags.writeable = False
    return ReferenceVsaLayout(cursor, len(dense), logical, gather, untile, sizes, regions)


def workspace_bytes(heads, max_tiles):
    if (
        type(heads) is not int
        or not 0 < heads <= 256
        or type(max_tiles) is not int
        or not 0 < max_tiles <= MAX_TILES
    ):
        raise ValueError("OmniRef VSA workspace requires positive head and tile bounds")
    pools = heads * max_tiles * 128 * 4
    scores = heads * max_tiles * max_tiles * 4
    counts = heads * max_tiles * 4
    return 3 * pools + 2 * scores + counts


def add_tiled_attention(
    network, operands, block_sizes, regions, *, plugin_library, heads, max_tiles
):
    """Add the complete sparse attention/routing/compression BYOK operation.

    Operands are already tiled BF16 [1, heads, padded_rows, 128]. The family
    runtime supplies validated geometry. The enclosing builder bounds their
    dynamic shapes and reserves this operation's maximum workspace.
    """
    import tensorrt as trt
    from tensorrt_model_connect.byok import add_kernel

    if not isinstance(operands, dict) or set(operands) != {"q", "k", "v", "gate"}:
        raise ValueError("OmniRef VSA requires Q, K, V and the learned compression gate")
    for value in operands.values():
        shape = tuple(value.shape)
        if (
            len(shape) != 4
            or shape[:2] != (1, heads)
            or shape[3] != 128
            or value.dtype != trt.bfloat16
        ):
            raise ValueError("OmniRef VSA operands must be tiled BF16 head tensors")
    if block_sizes.dtype != trt.int32 or len(tuple(block_sizes.shape)) != 1:
        raise ValueError("OmniRef block sizes must be an INT32 vector")
    if regions.dtype != trt.int32 or len(tuple(regions.shape)) != 2 or regions.shape[1] != 3:
        raise ValueError("OmniRef region descriptors must have shape [regions,3] and dtype INT32")
    (result,) = add_kernel(
        network,
        plugin_library=plugin_library,
        kernel_name=KERNEL_NAME,
        inputs=[operands[n] for n in ("q", "k", "v", "gate")] + [block_sizes, regions],
        output_specs=[{"dims": "same_as_input_0", "dtype": "bfloat16"}],
        workspace_bytes=workspace_bytes(heads, max_tiles),
    )
    return result


def add_packed_attention(
    network,
    operands,
    gather_indices,
    untile_indices,
    block_sizes,
    regions,
    *,
    plugin_library,
    heads,
    max_tiles,
):
    """Transport live packed rows into VSA tiles and invert that permutation."""
    import tensorrt as trt
    from . import graph_ops as op

    tiled = {}
    for name, value in operands.items():
        if tuple(value.shape)[-1] != heads * 128 or value.dtype != trt.bfloat16:
            raise ValueError("OmniRef packed attention requires complete BF16 head rows")
        shaped = network.add_shuffle(value)
        shaped.reshape_dims = (-1, heads, 128)
        zero = op.cast(network, op.constant(network, np.zeros((1, heads, 128))), trt.bfloat16)
        extended = network.add_concatenation([shaped.get_output(0), zero])
        extended.axis = 0
        gathered = network.add_gather(extended.get_output(0), gather_indices, 0).get_output(0)
        transpose = network.add_shuffle(gathered)
        transpose.first_transpose = trt.Permutation([1, 0, 2])
        transpose.reshape_dims = (1, heads, -1, 128)
        tiled[name] = transpose.get_output(0)
    combined = add_tiled_attention(
        network,
        tiled,
        block_sizes,
        regions,
        plugin_library=plugin_library,
        heads=heads,
        max_tiles=max_tiles,
    )
    packed = network.add_gather(combined, untile_indices, 2).get_output(0)
    return op.heads_to_rows(network, packed, -1, heads * 128)
