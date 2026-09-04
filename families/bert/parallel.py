# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import tensorrt as trt


@dataclass(frozen=True)
class ParallelConfig:
    tp_size: int = 1
    rank: int = -1

    @property
    def enabled(self) -> bool:
        return self.tp_size > 1

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("BERT tensor_parallel_size must be 1, 2, 4, or 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("BERT tensor-parallel rank is outside tensor_parallel_size")


def add_all_reduce_sum(network, tensor, tp_size: int):
    if tp_size == 1:
        return tensor
    layer = network.add_dist_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create the BERT all-reduce")
    layer.num_ranks = tp_size
    return layer.get_output(0)
