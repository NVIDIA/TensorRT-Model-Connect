# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixArt tensor-parallel build primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
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
            raise ValueError("PixArt tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("PixArt rank is outside the requested world")


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    config = value or ParallelConfig()
    config.validate()
    return config


def validate_dit_tp(
    *, dim: int, num_heads: int, ffn_dim: int, parallel: ParallelConfig, feature: str
) -> None:
    parallel.validate()
    if parallel.rank < 0:
        raise ValueError(f"{feature} requires a concrete rank")
    for name, value in (("dim", dim), ("num_heads", num_heads), ("ffn_dim", ffn_dim)):
        if value % parallel.tp_size:
            raise ValueError(f"{feature} {name} must be divisible by tensor_parallel_size")


def _slice_first_dim(value: np.ndarray, rank: int, size: int) -> np.ndarray:
    if value.shape[0] % size:
        raise ValueError("PixArt tensor first dimension is not TP divisible")
    return np.ascontiguousarray(np.array_split(value, size, axis=0)[rank])


def _slice_last_dim(value: np.ndarray, rank: int, size: int) -> np.ndarray:
    if value.shape[-1] % size:
        raise ValueError("PixArt tensor last dimension is not TP divisible")
    return np.ascontiguousarray(np.array_split(value, size, axis=-1)[rank])


def add_all_reduce_sum(network, tensor, tp_size: int):
    if int(tp_size) <= 1:
        return tensor
    layer = network.add_dist_collective(
        tensor, trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1, []
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create PixArt ALL_REDUCE")
    layer.num_ranks = int(tp_size)
    return layer.get_output(0)
