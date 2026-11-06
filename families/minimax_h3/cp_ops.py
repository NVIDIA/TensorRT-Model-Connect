# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ulysses context-parallel graph vocabulary for the MiniMax-H3 DiT.

Layout. The packed sequence is ``text | audio | video``. Only the video rows
(``video_rows``, fixed and divisible by the CP size) are sharded; the
``text + audio`` rows (dynamic, <1k) stay replicated on every rank as the
"head" segment. Everything outside attention is row-wise, so it runs on the
local rows unchanged.

Attention (per layer), with ``N`` ranks, ``H`` heads, ``h = H / N``:

* video q/k/v ``[V/N, H*D]`` -> ``ALL_TO_ALL`` -> ``[1, h, V, D]`` (sequence to heads)
* head-segment q/k/v are replicated, so this rank's head block is selected out
  of them with an ``ALL_TO_ALL`` on identical data (pure data movement, exact)
* attention over ``[head | video]`` for ``h`` heads
* video context ``ALL_TO_ALL`` back to ``[V/N, H*D]``; head-segment context
  ``ALL_GATHER`` over heads, restoring ``[head_rows, H*D]`` on every rank

No reduction is split, so the result is bit-identical to single-device
attention. Every graph is rank-agnostic: one serialized plan serves all ranks,
and the communicator supplies the rank at run time.

Selecting rank-local data with an all-to-all of replicated data, rather than a
scaled ``REDUCE_SCATTER``, keeps it bit-exact for every CP size (summing ``N``
equal values is not exact in floating point once a partial sum such as ``3a``
appears) and avoids carrying the rank as an engine input.
"""

from __future__ import annotations

import os

import numpy as np

from . import graph_ops as op
from ._trt import trt
from .config import MiniMaxH3Config


def add_collective(network, tensor, operation, cp_size: int):
    """Add a DIST_COLLECTIVE layer.

    Debug option: MINIMAX_H3_CP_WIRE_DTYPE=float32 carries 2-byte payloads in
    FP32 on the wire (not needed once contexts are materialized, see
    ``_materialize``).
    """

    original = tensor.dtype
    widen = original in (trt.bfloat16, trt.float16) and (
        os.environ.get("MINIMAX_H3_CP_WIRE_DTYPE", "native") == "float32"
    )
    if widen:
        tensor = op.cast(network, tensor, trt.float32)
    layer = network.add_dist_collective(tensor, operation, trt.ReduceOperation.NONE, -1, [])
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add MiniMax-H3 {operation} collective")
    layer.num_ranks = int(cp_size)
    result = layer.get_output(0)
    return op.cast(network, result, original) if widen else result


def collectives_per_attention() -> int:
    """q/k/v head-block select (3) + q/k/v seq->head (3) + ctx head->seq + head all-gather."""

    return 8


def expected_collectives(profile: MiniMaxH3Config) -> int:
    """Collective layers in a CP DiT plan (0 for a single device)."""

    if profile.context_parallel_size == 1:
        return 0
    input_shards = 4  # video latent, position ids, adaln indices, timestep indices
    output_gather = 1  # video velocity
    return input_shards + output_gather + profile.num_layers * collectives_per_attention()


def _slice(network, tensor, starts, sizes):
    """Slice with each start/size either a Python int or an int64 ``[1]`` tensor."""

    ndim = len(starts)
    static_starts = tuple(s if isinstance(s, (int, np.integer)) else 0 for s in starts)
    static_sizes = tuple(s if isinstance(s, (int, np.integer)) else 1 for s in sizes)
    layer = network.add_slice(tensor, static_starts, static_sizes, (1,) * ndim)
    if any(not isinstance(s, (int, np.integer)) for s in starts):
        layer.set_input(1, op._shape_vector(network, starts))
    layer.set_input(2, op._shape_vector(network, sizes))
    return layer.get_output(0)


def _dim_minus(network, tensor, axis: int, amount: int):
    total = op._shape_dim(network, tensor, axis)
    value = op.constant(network, np.asarray([amount], dtype=np.int64), dtype=np.int64)
    return network.add_elementwise(total, value, trt.ElementWiseOperation.SUB).get_output(0)


def head_rows(network, tensor, video_rows: int):
    """Rows before the trailing ``video_rows`` of a packed tensor (dynamic count)."""

    tail = tuple(int(d) for d in tensor.shape[1:])
    count = _dim_minus(network, tensor, 0, video_rows)
    return _slice(network, tensor, (0,) * (1 + len(tail)), (count, *tail))


def trailing_rows(network, tensor, rows: int):
    """The last ``rows`` rows of a packed tensor of any rank >= 1."""

    tail = tuple(int(d) for d in tensor.shape[1:])
    start = _dim_minus(network, tensor, 0, rows)
    return _slice(network, tensor, (start, *([0] * len(tail))), (rows, *tail))


def shard_replicated_rows(network, tensor, cp_size: int):
    """Give each rank its ``rows / N`` block of identical full-row input.

    ``tensor`` is ``[rows, ...]`` with a static row count, replicated on all ranks.
    An all-to-all sends block ``j`` to rank ``j``; every source holds the same
    data, so block 0 of the result is this rank's block, bit-exactly.
    """

    rows = int(tensor.shape[0])
    if rows % cp_size:
        raise ValueError(f"cannot shard {rows} rows across {cp_size} ranks")
    tail = tuple(int(d) for d in tensor.shape[1:])
    local = rows // cp_size
    blocks = network.add_shuffle(tensor)
    blocks.reshape_dims = (cp_size, local, *tail)
    exchanged = add_collective(network, blocks.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, cp_size)
    mine = network.add_slice(
        exchanged, (0,) * (2 + len(tail)), (1, local, *tail), (1,) * (2 + len(tail))
    ).get_output(0)
    flat = network.add_shuffle(mine)
    flat.reshape_dims = (local, *tail)
    return flat.get_output(0)


def shard_replicated_rows_int(network, tensor, cp_size: int):
    """``shard_replicated_rows`` for int32 index tensors (small ints are exact in FP32)."""

    as_float = op.cast(network, tensor, trt.float32)
    sharded = shard_replicated_rows(network, as_float, cp_size)
    return op.cast(network, sharded, trt.int32)


def local_packed_rows(network, tensor, profile: MiniMaxH3Config, *, integer: bool = False):
    """Per-row packed tensor ``[text|audio|video, ...]`` -> ``[text|audio|video/N, ...]``."""

    video = trailing_rows(network, tensor, profile.video_rows)
    shard = shard_replicated_rows_int if integer else shard_replicated_rows
    local_video = shard(network, video, profile.context_parallel_size)
    concat = network.add_concatenation((head_rows(network, tensor, profile.video_rows), local_video))
    concat.axis = 0
    return concat.get_output(0)


def _seq_to_heads(network, rows, profile: MiniMaxH3Config):
    """``[V/N, H*D]`` rows -> ``[1, h, V, D]`` (full video sequence, ``h`` heads)."""

    n, h, d = profile.context_parallel_size, profile.local_heads, profile.head_dim
    local = profile.local_video_rows
    routed = network.add_shuffle(rows)
    routed.reshape_dims = (local, n, h, d)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = add_collective(network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, n)
    full = network.add_shuffle(exchanged)
    full.first_transpose = trt.Permutation([2, 0, 1, 3])
    full.reshape_dims = (1, h, profile.video_rows, d)
    return full.get_output(0)


def _heads_to_seq(network, context, profile: MiniMaxH3Config):
    """``[1, h, V, D]`` context -> ``[V/N, H*D]`` local video rows (all heads)."""

    n, h, d = profile.context_parallel_size, profile.local_heads, profile.head_dim
    local = profile.local_video_rows
    routed = network.add_shuffle(context)
    routed.reshape_dims = (h, n, local, d)
    routed.second_transpose = trt.Permutation([1, 0, 2, 3])
    exchanged = add_collective(network, routed.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, n)
    rows = network.add_shuffle(exchanged)
    rows.first_transpose = trt.Permutation([2, 0, 1, 3])
    rows.reshape_dims = (local, profile.attention_size)
    return rows.get_output(0)


def _my_head_block(network, rows, profile: MiniMaxH3Config):
    """Replicated ``[T, H*D]`` rows -> this rank's heads as ``[1, h, T, D]``."""

    n, h, d = profile.context_parallel_size, profile.local_heads, profile.head_dim
    blocks = network.add_shuffle(rows)
    blocks.reshape_dims = (-1, n, h * d)
    blocks.second_transpose = trt.Permutation([1, 0, 2])
    exchanged = add_collective(network, blocks.get_output(0), trt.CollectiveOperation.ALL_TO_ALL, n)
    mine = op.dynamic_slice(network, exchanged, (0, 0, 0), (1, None, h * d))
    heads = network.add_shuffle(mine)
    heads.reshape_dims = (1, -1, h, d)
    heads.second_transpose = trt.Permutation([0, 2, 1, 3])
    return heads.get_output(0)


def _materialize(network, context, head_dim: int):
    """Fusion barrier: multiply ``[1, h, rows, D]`` by an exact identity ``[D, D]``.

    TRT-RTX 1.7.1 (sm_120) fuses an IAttention output straight into the
    shuffle/collective chain that follows it and then returns wrong data
    (relL2 ~1.1; correct, relL2 ~2e-3, once the context is materialized, e.g.
    as a network output). The identity GEMM keeps the values bit-exact and costs
    ~2*rows*h*D*D FLOPs.
    """

    eye = op.weight_constant(network, np.eye(head_dim, dtype=np.float32).reshape(1, 1, head_dim, head_dim))
    eye = op.cast(network, eye, context.dtype)
    return network.add_matrix_multiply(
        context, trt.MatrixOperation.NONE, eye, trt.MatrixOperation.NONE
    ).get_output(0)


def ulysses_attention(network, q, k, v, *, profile: MiniMaxH3Config, name: str):
    """Context-parallel attention over local ``[head|video/N]`` q/k/v rows.

    Returns the attention output as local ``[head|video/N, H*D]`` rows.

    The head-segment and video queries run as two attentions over the same full
    K/V, so each context goes straight from its IAttention output into its
    return collective. Slicing one shared attention output with a dynamic start
    and feeding the views to collectives returns wrong data on TRT-RTX 1.7.1
    (sm_120), whereas a collective that consumes the attention output directly
    is exact.
    """

    n, h, d = profile.context_parallel_size, profile.local_heads, profile.head_dim
    video_rows = profile.video_rows
    local = profile.local_video_rows

    head_blocks, video_blocks = [], []
    for tensor in (q, k, v):
        head = head_rows(network, tensor, local)
        video = op.slice_rows_from_end(network, tensor, offset=local, rows=local)
        head_blocks.append(_my_head_block(network, head, profile))
        video_blocks.append(_seq_to_heads(network, video, profile))

    full_kv = []
    for head_block, video_block in zip(head_blocks[1:], video_blocks[1:]):
        merged = network.add_concatenation((head_block, video_block))
        merged.axis = 2
        full_kv.append(merged.get_output(0))

    video_context = op.attention_core(
        network, video_blocks[0], *full_kv, head_dim=d, name=f"{name}.video"
    )
    head_context = op.attention_core(
        network, head_blocks[0], *full_kv, head_dim=d, name=f"{name}.head"
    )

    barrier = os.environ.get("MINIMAX_H3_CP_BARRIER", "gemm")
    if barrier == "gemm":
        video_context = _materialize(network, video_context, d)
        head_context = _materialize(network, head_context, d)

    if os.environ.get("MINIMAX_H3_CP_DEBUG_OUTPUTS"):  # test hook: expose contexts
        for tag, t in (("dbg_ctx_video", video_context), ("dbg_ctx_head", head_context)):
            c = network.add_identity(op.cast(network, t, trt.float32)).get_output(0)
            c.name = tag
            network.mark_output(c)

    video_out = _heads_to_seq(network, video_context, profile)

    gathered_in = network.add_shuffle(head_context)
    gathered_in.reshape_dims = (h, -1, d)
    gathered = add_collective(
        network, gathered_in.get_output(0), trt.CollectiveOperation.ALL_GATHER, n
    )
    head_out = network.add_shuffle(gathered)
    head_out.first_transpose = trt.Permutation([1, 0, 2])
    head_out.reshape_dims = (-1, profile.attention_size)

    rows = network.add_concatenation((head_out.get_output(0), video_out))
    rows.axis = 0
    return rows.get_output(0)
