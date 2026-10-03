# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 model-owned context-parallel build primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SUPPORTED_CONTEXT_PARALLEL_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class ParallelConfig:
    """LTX-2.5 runs single-device or context parallel (video token shards); TP is not supported."""

    cp_size: int = 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def mode(self) -> str:
        return "context_parallel" if self.cp_enabled else "single"

    @property
    def world_size(self) -> int:
        return self.cp_size

    def validate(self) -> None:
        if self.cp_size not in SUPPORTED_CONTEXT_PARALLEL_SIZES:
            raise ValueError("LTX-2.5 context_parallel_size must be one of 1, 2, 4, 8")


def validate_context_parallel_layout(parallel: ParallelConfig, *, video_tokens: int, video_heads: int,
                                     audio_heads: int) -> None:
    """Reject layouts whose video tokens or attention heads cannot be split evenly."""
    parallel.validate()
    if not parallel.cp_enabled:
        return
    cp = parallel.cp_size
    if video_tokens % cp:
        raise ValueError(f"LTX-2.5 context parallel needs the video token count ({video_tokens}) "
                         f"divisible by context_parallel_size ({cp})")
    for name, heads in (("video", video_heads), ("audio", audio_heads)):
        if heads % cp:
            raise ValueError(f"LTX-2.5 context parallel needs {name} heads ({heads}) divisible by "
                             f"context_parallel_size ({cp})")


def add_collective(network, tensor, operation, cp_size: int, *, reduce_operation=None):
    """One TensorRT distributed collective spanning the whole CP world."""
    import tensorrt as trt

    if reduce_operation is None:
        reduce_operation = trt.ReduceOperation.NONE
    layer = network.add_dist_collective(tensor, operation, reduce_operation, -1, [])
    if layer is None:
        raise RuntimeError(f"TensorRT failed to add the LTX-2.5 {operation} collective")
    layer.num_ranks = int(cp_size)
    return layer.get_output(0)


def rank_selector_values(cp_size: int) -> np.ndarray:
    """Replicated ``[CP, 1]`` values whose SUM reduce-scatter yields each rank's index.

    Every rank contributes ``r / CP`` at row ``r``; summing CP identical copies hands
    rank ``r`` exactly ``r`` (CP is a power of two, so this is exact in fp32).
    """
    return (np.arange(cp_size, dtype=np.float32) / np.float32(cp_size)).reshape(cp_size, 1)


def local_row_indices(g, *, cp: int, local_rows: int):
    """int32 ``[local_rows]`` indices of the contiguous token shard owned by this rank."""
    import tensorrt as trt

    selector = g.const(rank_selector_values(cp), trt.float32)
    rank_f = add_collective(g.net, selector, trt.CollectiveOperation.REDUCE_SCATTER, cp,
                            reduce_operation=trt.ReduceOperation.SUM)
    rank_i = g.reshape(g.cast(rank_f, trt.int32), (1,))
    start = g.mul(rank_i, g.const(np.array([local_rows], np.int32), trt.int32))
    return g.add(g.const(np.arange(local_rows, dtype=np.int32), trt.int32), start)
