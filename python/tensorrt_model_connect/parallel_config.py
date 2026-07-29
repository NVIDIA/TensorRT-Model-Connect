# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel build metadata and weight sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


_TP_MODES = frozenset({"tensor_parallel"})
_SP_MODES = frozenset({"sp_ulysses", "sp_ring", "sp_allgather_kv"})
_ALL_MODES = frozenset({"single"}) | _TP_MODES | _SP_MODES


@dataclass(frozen=True)
class ParallelConfig:
    mode: str = "single"
    tp_size: int = 1
    cp_size: int = 1
    rank: int = -1
    require_mpirun: bool = True

    @property
    def enabled(self) -> bool:
        if self.tp_size > 1 and self.mode in _TP_MODES:
            return True
        if self.cp_size > 1 and self.mode in _SP_MODES:
            return True
        return False

    @property
    def world_size(self) -> int:
        """Total ranks needed for this parallel layout (tp * cp)."""
        return max(1, int(self.tp_size)) * max(1, int(self.cp_size))

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.mode not in _ALL_MODES:
            raise ValueError(f"Unsupported parallel.mode={self.mode!r}")
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("parallel.tp_size must be one of 1, 2, 4, 8")
        if self.cp_size <= 0 or self.cp_size > 8:
            raise ValueError(
                "parallel.cp_size must be a positive integer <= 8")
        if self.cp_size not in {1, 2, 4, 8}:
            raise ValueError("parallel.cp_size must be one of 1, 2, 4, 8")
        if self.cp_size > 1 and self.mode not in _SP_MODES:
            raise ValueError(
                f"parallel.cp_size={self.cp_size} requires a sequence-parallel mode "
                f"(one of {sorted(_SP_MODES)}); got mode={self.mode!r}")
        if self.rank < -1:
            raise ValueError("parallel.rank must be -1 or a non-negative rank")
        world = self.world_size
        if self.rank >= world:
            raise ValueError(
                f"parallel.rank={self.rank} must be smaller than world_size={world}")
        if self.mode == "single" and self.tp_size != 1:
            raise ValueError("parallel.mode=single requires parallel.tp_size=1")
        if self.mode == "single" and self.cp_size != 1:
            raise ValueError("parallel.mode=single requires parallel.cp_size=1")
        if self.mode in _SP_MODES and self.cp_size == 1:
            raise ValueError(
                f"parallel.mode={self.mode!r} requires parallel.cp_size > 1")

    def to_config_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mode": self.mode,
            "tp_size": self.tp_size,
            "cp_size": self.cp_size,
            "rank": self.rank,
            "require_mpirun": self.require_mpirun,
        }

    def to_bundle_config_fields(self) -> dict[str, object]:
        if not self.enabled:
            return {}
        return {
            "parallelism": self.to_config_dict(),
            "tensor_parallel_mode": self.mode,
            "tensor_parallel_size": self.tp_size,
            "tensor_parallel_require_mpirun": int(self.require_mpirun),
        }


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    cfg = value or ParallelConfig()
    cfg.validate()
    return cfg


def rank_engine_section(rank: int) -> str:
    return f"engine_plan_tp_rank{int(rank)}"


def rank_denoiser_section(rank: int) -> str:
    return f"denoiser_plan_tp_rank{int(rank)}"


def require_tensorrt_11_for_tensor_parallel(
    parallel: ParallelConfig,
    *,
    feature: str = "Tensor-parallel builds",
) -> None:
    """Raise unless an enabled tensor-parallel request is running on TRT 11.0+."""
    parallel.validate()
    if not parallel.enabled:
        return

    from . import trt_compat

    version = trt_compat.tensorrt_version()
    match = re.search(r"(\d+)", version or "")
    major = int(match.group(1)) if match else 0
    if major < 11:
        found = version or "unavailable"
        raise RuntimeError(f"{feature} requires TensorRT 11.0+; found {found}")


def validate_standard_decoder_tp(
    model_config: "ModelConfig",
    weights: "WeightDict",
    parallel: ParallelConfig,
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Tensor-parallel engine build requires a concrete rank")
    tp = parallel.tp_size
    if model_config.num_attention_heads % tp != 0:
        raise ValueError(
            "Standard decoder tensor parallel requires num_attention_heads divisible by tp_size "
            f"({model_config.num_attention_heads} vs {tp})")
    if model_config.num_key_value_heads % tp != 0:
        raise ValueError(
            "Standard decoder tensor parallel requires num_key_value_heads divisible by tp_size "
            f"({model_config.num_key_value_heads} vs {tp})")
    mlp_size = int(weights.get("_mlp_size", model_config.intermediate_size))
    if mlp_size % tp != 0:
        raise ValueError(
            f"Standard decoder tensor parallel requires intermediate size divisible by tp_size "
            f"({mlp_size} vs {tp})")


def validate_dit_tp(
    *,
    dim: int,
    num_heads: int,
    ffn_dim: int,
    parallel: ParallelConfig,
    feature: str = "DiT tensor parallel",
) -> None:
    """Validate common DiT tensor-parallel dimensions."""
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError(f"{feature} engine build requires a concrete rank")
    tp = parallel.tp_size
    if dim % tp != 0:
        raise ValueError(
            f"{feature} requires hidden dim divisible by tp_size ({dim} vs {tp})")
    if num_heads % tp != 0:
        raise ValueError(
            f"{feature} requires num_attention_heads divisible by tp_size "
            f"({num_heads} vs {tp})")
    if ffn_dim % tp != 0:
        raise ValueError(
            f"{feature} requires FFN size divisible by tp_size ({ffn_dim} vs {tp})")


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    parts = np.array_split(arr, tp_size, axis=-1)
    return np.ascontiguousarray(parts[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    parts = np.array_split(arr, tp_size, axis=0)
    return np.ascontiguousarray(parts[rank])


def shard_standard_decoder_weights(
    model_config: "ModelConfig",
    weights: "WeightDict",
    parallel: ParallelConfig,
) -> "WeightDict":
    """Return rank-local standard-decoder weights.

    The current TP policy keeps embeddings, norms, and the LM head replicated.
    Attention and MLP inner dimensions are sharded. Row-parallel projections
    are followed by TRT distributed all-reduce in the builder.
    """
    validate_standard_decoder_tp(model_config, weights, parallel)
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((".w_q", ".w_k", ".w_v", ".q_bias", ".k_bias", ".v_bias")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, rank, tp)
        elif key.endswith((".w_gate", ".w_up", ".w_fc1", ".fc1_bias")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".w_fc2"):
            out[key] = _slice_first_dim(value, rank, tp)
        elif key.endswith((".q_norm", ".k_norm")) and value.size > model_config.head_dim:
            out[key] = _slice_first_dim(value.reshape(-1, model_config.head_dim), rank, tp).reshape(-1)
        else:
            out[key] = value

    out["_attention_size"] = int(weights["_attention_size"]) // tp
    out["_kv_attention_size"] = int(weights["_kv_attention_size"]) // tp
    out["_mlp_size"] = int(weights["_mlp_size"]) // tp
    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


_TRT11_REQUIRED_MSG = (
    "Distributed collectives require TensorRT 11.0+ Python bindings "
    "with INetworkDefinition.add_dist_collective"
)


def _resolve_collective_api(network):
    """Return (trt_proxy, add_collective) or raise if TRT 11+ API is missing."""
    from tensorrt_model_connect import trt_compat

    add_collective = getattr(network, "add_dist_collective", None)
    if add_collective is None:
        raise RuntimeError(_TRT11_REQUIRED_MSG + " (requires TRT 11+)")
    return trt_compat.get_trt(), add_collective


def _finalize_collective_layer(layer, op_name: str, ranks: int):
    if layer is None:
        raise RuntimeError(
            f"TensorRT failed to add {op_name} distributed collective"
            " (requires TRT 11+)")
    if not hasattr(layer, "num_ranks"):
        raise RuntimeError(
            f"{op_name} requires TensorRT 11.0+ Python bindings "
            f"with IDistCollectiveLayer.num_ranks (requires TRT 11+)")
    layer.num_ranks = int(ranks)
    return layer.get_output(0)


def _check_parallel_size(parallel_size: int, op_name: str) -> int:
    parallel_size = int(parallel_size)
    if parallel_size < 0:
        raise RuntimeError(
            f"{op_name}: parallel_size must be >= 0, got {parallel_size}"
            " (requires TRT 11+)")
    return parallel_size


def add_all_reduce_sum(network, tensor, tp_size: int):
    """Insert a TRT 11.0+ all-reduce SUM collective for tensor-parallel joins."""
    tp_size = _check_parallel_size(tp_size, "add_all_reduce_sum")
    if tp_size <= 1:
        return tensor

    trt, add_collective = _resolve_collective_api(network)
    layer = add_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    return _finalize_collective_layer(layer, "ALL_REDUCE", tp_size)


def add_all_gather(network, tensor, cp_size: int, gather_axis: int = -1):
    """All-gather across ``cp_size`` ranks along ``gather_axis``.

    Output shape matches the input except the gather axis grows by ``cp_size``
    (each rank contributes its local shard). A pass-through is returned when
    ``cp_size <= 1``.
    """
    cp_size = _check_parallel_size(cp_size, "add_all_gather")
    if cp_size <= 1:
        return tensor

    trt, add_collective = _resolve_collective_api(network)
    gather_axis = int(gather_axis)
    layer = add_collective(
        tensor,
        trt.CollectiveOperation.ALL_GATHER,
        trt.ReduceOperation.NONE,
        gather_axis,
        [],
    )
    return _finalize_collective_layer(layer, "ALL_GATHER", cp_size)


def add_all_to_all(
    network,
    tensor,
    cp_size: int,
    scatter_axis: int,
    gather_axis: int,
):
    """All-to-all redistribution across ``cp_size`` ranks.

    Input tensor is sharded along ``gather_axis`` (each rank holds 1/cp_size of
    that axis) and the output is sharded along ``scatter_axis`` instead. The
    declared output shape grows ``scatter_axis`` by ``cp_size`` and shrinks
    ``gather_axis`` by ``cp_size``.

    The underlying TRT 11 API expects the ``axis`` argument to encode both
    scatter and gather axes; we pass them as a 2-element list. If the runtime
    only accepts a scalar we fall back to passing the scatter axis and rely on
    the ``sizes`` argument to encode the gather axis.
    """
    cp_size = _check_parallel_size(cp_size, "add_all_to_all")
    if cp_size <= 1:
        return tensor

    trt, add_collective = _resolve_collective_api(network)
    scatter_axis = int(scatter_axis)
    gather_axis = int(gather_axis)
    if scatter_axis == gather_axis:
        raise RuntimeError(
            "add_all_to_all: scatter_axis and gather_axis must differ "
            f"(both = {scatter_axis}) (requires TRT 11+)")

    # Try the documented 2-axis form first; fall back to scalar+sizes if the
    # binding rejects the list form.
    layer = None
    try:
        layer = add_collective(
            tensor,
            trt.CollectiveOperation.ALL_TO_ALL,
            trt.ReduceOperation.NONE,
            [scatter_axis, gather_axis],
            [],
        )
    except TypeError:
        layer = None
    if layer is None:
        layer = add_collective(
            tensor,
            trt.CollectiveOperation.ALL_TO_ALL,
            trt.ReduceOperation.NONE,
            scatter_axis,
            [gather_axis],
        )
    return _finalize_collective_layer(layer, "ALL_TO_ALL", cp_size)


def add_reduce_scatter_sum(network, tensor, cp_size: int, scatter_axis: int = -1):
    """Reduce-scatter SUM across ``cp_size`` ranks along ``scatter_axis``.

    Output declares ``scatter_axis`` shrunk by ``cp_size`` after a SUM reduction
    across all ranks. Pass-through when ``cp_size <= 1``.
    """
    cp_size = _check_parallel_size(cp_size, "add_reduce_scatter_sum")
    if cp_size <= 1:
        return tensor

    trt, add_collective = _resolve_collective_api(network)
    scatter_axis = int(scatter_axis)
    layer = add_collective(
        tensor,
        trt.CollectiveOperation.REDUCE_SCATTER,
        trt.ReduceOperation.SUM,
        scatter_axis,
        [],
    )
    return _finalize_collective_layer(layer, "REDUCE_SCATTER", cp_size)


