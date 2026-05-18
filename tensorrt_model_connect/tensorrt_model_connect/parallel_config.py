"""Tensor-parallel build metadata and weight sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from .runtime_config import ConfigBundle


@dataclass(frozen=True)
class ParallelConfig:
    mode: str = "single"
    tp_size: int = 1
    rank: int = -1
    require_mpirun: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode == "tensor_parallel" and self.tp_size > 1

    def for_rank(self, rank: int) -> "ParallelConfig":
        return replace(self, rank=rank)

    def validate(self) -> None:
        if self.mode not in {"single", "tensor_parallel"}:
            raise ValueError(f"Unsupported parallel.mode={self.mode!r}")
        if self.tp_size not in {1, 2, 4, 8}:
            raise ValueError("parallel.tp_size must be one of 1, 2, 4, 8")
        if self.rank < -1:
            raise ValueError("parallel.rank must be -1 or a non-negative rank")
        if self.rank >= self.tp_size:
            raise ValueError(
                f"parallel.rank={self.rank} must be smaller than tp_size={self.tp_size}")
        if self.mode == "single" and self.tp_size != 1:
            raise ValueError("parallel.mode=single requires parallel.tp_size=1")

    def to_config_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mode": self.mode,
            "tp_size": self.tp_size,
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


def parallel_config_from_bundle(bundle: "ConfigBundle | None") -> ParallelConfig:
    if bundle is None:
        return ParallelConfig()
    cfg = ParallelConfig(
        mode=str(bundle.get("parallel", "mode")),
        tp_size=int(bundle.get("parallel", "tp_size")),
        rank=int(bundle.get("parallel", "rank")),
        require_mpirun=bool(bundle.get("parallel", "require_mpirun")),
    )
    cfg.validate()
    return cfg


def normalize_parallel_config(value: ParallelConfig | None) -> ParallelConfig:
    cfg = value or ParallelConfig()
    cfg.validate()
    return cfg


def rank_engine_section(rank: int) -> str:
    return f"engine_plan_tp_rank{int(rank)}"


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
            "Qwen tensor parallel requires num_attention_heads divisible by tp_size "
            f"({model_config.num_attention_heads} vs {tp})")
    if model_config.num_key_value_heads % tp != 0:
        raise ValueError(
            "Qwen tensor parallel requires num_key_value_heads divisible by tp_size "
            f"({model_config.num_key_value_heads} vs {tp})")
    mlp_size = int(weights.get("_mlp_size", model_config.intermediate_size))
    if mlp_size % tp != 0:
        raise ValueError(
            f"Qwen tensor parallel requires intermediate size divisible by tp_size "
            f"({mlp_size} vs {tp})")


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
    """Return rank-local Qwen/standard-decoder weights.

    The current TP policy keeps embeddings, norms, and the LM head replicated.
    Attention and MLP inner dimensions are sharded. Row-parallel projections
    are followed by TRT distributed all-reduce in the builder.
    """
    from .checkpoint_mapper import WeightDict

    validate_standard_decoder_tp(model_config, weights, parallel)
    if not parallel.enabled:
        return weights

    rank = parallel.rank
    tp = parallel.tp_size
    out = WeightDict()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((".w_q", ".w_k", ".w_v", ".q_bias", ".k_bias", ".v_bias")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, rank, tp)
        elif key.endswith((".w_gate", ".w_up", ".w_fc1")):
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


def add_all_reduce_sum(network, tensor, tp_size: int):
    """Insert a TRT 11.0+ all-reduce SUM collective for tensor-parallel joins."""
    from tensorrt_model_connect import trt_compat

    tp_size = int(tp_size)
    if tp_size <= 1:
        return tensor

    trt = trt_compat.get_trt()
    add_collective = getattr(network, "add_dist_collective", None)
    if add_collective is None:
        raise RuntimeError(
            "Tensor-parallel Qwen builds require TensorRT 11.0+ Python bindings "
            "with INetworkDefinition.add_dist_collective")
    layer = add_collective(
        tensor,
        trt.CollectiveOperation.ALL_REDUCE,
        trt.ReduceOperation.SUM,
        -1,
        [],
    )
    if layer is None:
        raise RuntimeError("TensorRT failed to add ALL_REDUCE distributed collective")
    if not hasattr(layer, "num_ranks"):
        raise RuntimeError(
            "Tensor-parallel Qwen builds require TensorRT 11.0+ Python bindings "
            "with IDistCollectiveLayer.num_ranks")
    layer.num_ranks = tp_size
    return layer.get_output(0)
