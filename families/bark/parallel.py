# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark tensor-parallel build primitives."""

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
            raise ValueError("Bark tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("Bark tensor-parallel rank is outside the requested world")


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    config = value or ParallelConfig()
    config.validate()
    return config


def add_all_reduce_sum(network, tensor, tp_size: int):
    if int(tp_size) <= 1:
        return tensor
    layer = network.add_dist_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create Bark ALL_REDUCE")
    layer.num_ranks = int(tp_size)
    return layer.get_output(0)
