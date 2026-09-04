# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ALBERT-owned tensor-parallel build primitives."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


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
            raise ValueError("ALBERT tensor_parallel_size must be one of 1, 2, 4, 8")
        if self.rank < -1 or self.rank >= self.tp_size:
            raise ValueError("ALBERT tensor-parallel rank is outside the requested world")


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    config = value or ParallelConfig()
    config.validate()
    return config


def _slice_last_dim(array: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(array, tp_size, axis=-1)[rank])


def _slice_first_dim(array: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(array, tp_size, axis=0)[rank])


def shard_standard_decoder_weights(
    model_config: "ModelConfig",
    weights: "WeightDict",
    parallel: ParallelConfig,
) -> "WeightDict":
    parallel.validate()
    if not parallel.enabled:
        return weights
    if parallel.rank < 0:
        raise ValueError("ALBERT tensor-parallel build requires a concrete rank")
    if model_config.num_attention_heads % parallel.tp_size:
        raise ValueError(
            "ALBERT num_attention_heads must be divisible by tensor_parallel_size"
        )
    if model_config.num_key_value_heads % parallel.tp_size:
        raise ValueError(
            "ALBERT num_key_value_heads must be divisible by tensor_parallel_size"
        )
    mlp_size = int(weights.get("_mlp_size", model_config.intermediate_size))
    if mlp_size % parallel.tp_size:
        raise ValueError(
            "ALBERT intermediate size must be divisible by tensor_parallel_size"
        )

    rank = parallel.rank
    tp_size = parallel.tp_size
    sharded = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            sharded[key] = value
        elif key.endswith((".w_q", ".w_k", ".w_v", ".q_bias", ".k_bias", ".v_bias")):
            sharded[key] = _slice_last_dim(value, rank, tp_size)
        elif key.endswith((".w_o", ".w_fc2")):
            sharded[key] = _slice_first_dim(value, rank, tp_size)
        elif key.endswith((".w_fc1", ".fc1_bias")):
            sharded[key] = _slice_last_dim(value, rank, tp_size)
        else:
            sharded[key] = value

    sharded["_attention_size"] = int(weights["_attention_size"]) // tp_size
    sharded["_kv_attention_size"] = int(weights["_kv_attention_size"]) // tp_size
    sharded["_mlp_size"] = int(weights["_mlp_size"]) // tp_size
    sharded["_tensor_parallel_size"] = tp_size
    sharded["_tensor_parallel_rank"] = rank
    return sharded


def add_all_reduce_sum(network, tensor, tp_size: int):
    tp_size = int(tp_size)
    if tp_size <= 1:
        return tensor
    layer = network.add_dist_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to create ALBERT ALL_REDUCE")
    layer.num_ranks = tp_size
    return layer.get_output(0)
