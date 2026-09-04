# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan model-owned distributed build primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import tensorrt as trt


@dataclass(frozen=True)
class ParallelConfig:
    tp_size: int = 1
    cp_size: int = 1
    rank: int = -1

    @property
    def enabled(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp_size > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1

    @property
    def distributed(self) -> bool:
        return self.enabled or self.cp_enabled

    @property
    def world_size(self) -> int:
        return self.cp_size if self.cp_enabled else self.tp_size

    @property
    def mode(self) -> str:
        if self.enabled:
            return "tensor_parallel"
        if self.cp_enabled:
            return "context_parallel"
        return "single"

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("Wan tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.cp_size not in {1, 2, 4, 8}:
            raise ValueError("Wan context_parallel_size must be one of 1, 2, 4, 8")
        if self.enabled and self.cp_enabled:
            raise ValueError("Wan tensor and context parallelism cannot be enabled together")
        if self.rank < -1 or self.rank >= self.world_size:
            raise ValueError("Wan rank is outside the requested world")


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    config = value or ParallelConfig()
    config.validate()
    return config


def validate_dit_tp(
    *, dim: int, num_heads: int, ffn_dim: int, parallel: ParallelConfig, feature: str
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError(f"{feature} requires a concrete rank")
    if any(value % parallel.tp_size for value in (dim, num_heads, ffn_dim)):
        raise ValueError(f"{feature} dimensions must be divisible by tensor_parallel_size")


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
        raise RuntimeError("TensorRT failed to create Wan ALL_REDUCE")
    layer.num_ranks = tp_size
    return layer.get_output(0)
