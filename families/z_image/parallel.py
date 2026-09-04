# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image tensor-parallel build primitives."""

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
            raise ValueError("Z-Image tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("Z-Image rank is outside the requested world")


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
    if any(value % parallel.tp_size for value in (dim, num_heads, ffn_dim)):
        raise ValueError(f"{feature} dimensions are not TP divisible")


def _slice_first_dim(value: np.ndarray, rank: int, size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(value, size, axis=0)[rank])


def _slice_last_dim(value: np.ndarray, rank: int, size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(value, size, axis=-1)[rank])


def add_all_reduce_sum(network, tensor, tp_size: int):
    if tp_size <= 1:
        return tensor
    layer = network.add_dist_collective(
        tensor, trt.CollectiveOperation.ALL_REDUCE, trt.ReduceOperation.SUM, -1, []
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create Z-Image ALL_REDUCE")
    layer.num_ranks = tp_size
    return layer.get_output(0)


def add_dynamic_batch_profile(
    builder,
    config,
    *,
    input_names: list[str],
    max_batch: int,
    opt_batch: int,
    static_shape: dict[str, tuple[int, ...]],
) -> None:
    if max_batch < 1 or not 1 <= opt_batch <= max_batch:
        raise ValueError("Z-Image dynamic batch profile is invalid")
    profile = builder.create_optimization_profile()
    for name in input_names:
        tail = tuple(static_shape[name])
        profile.set_shape(name, min=(1, *tail), opt=(opt_batch, *tail), max=(max_batch, *tail))
    config.add_optimization_profile(profile)
