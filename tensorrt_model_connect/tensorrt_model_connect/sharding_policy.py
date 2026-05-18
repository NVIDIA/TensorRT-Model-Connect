"""Distributed sharding policy objects.

Policies translate a distributed plan into local builder decisions: weight
slices, local tensor dimensions, and required collectives. The implemented
policy in this branch is the tensor-parallel standard-decoder policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .distributed_plan import CollectivePlan, RegionPlan
from .parallel_config import ParallelConfig

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from .model_recipe import ModelRecipe


def _slice_last_dim(arr: np.ndarray, rank: int, size: int) -> np.ndarray:
    parts = np.array_split(arr, size, axis=-1)
    return np.ascontiguousarray(parts[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, size: int) -> np.ndarray:
    parts = np.array_split(arr, size, axis=0)
    return np.ascontiguousarray(parts[rank])


def _add_all_reduce_sum(network, tensor, group_size: int):
    """Insert a TRT 11.0+ all-reduce SUM collective for distributed joins."""

    from tensorrt_model_connect import trt_compat

    group_size = int(group_size)
    if group_size <= 1:
        return tensor

    trt = trt_compat.get_trt()
    add_collective = getattr(network, "add_dist_collective", None)
    if add_collective is None:
        raise RuntimeError(
            "Distributed decoder builds require TensorRT 11.0+ Python bindings "
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
            "Distributed decoder builds require TensorRT 11.0+ Python bindings "
            "with IDistCollectiveLayer.num_ranks")
    layer.num_ranks = group_size
    return layer.get_output(0)


@dataclass(frozen=True)
class StandardDecoderShardingPolicy:
    """Rank-local sharding policy for standard decoder engines."""

    config: "ModelConfig"
    weights: "WeightDict"
    parallel: ParallelConfig
    recipe: "ModelRecipe | None" = None

    @property
    def distributed(self) -> bool:
        return self.parallel.enabled

    @property
    def tp_size(self) -> int:
        return self.parallel.tp_size if self.distributed else 1

    @property
    def rank(self) -> int:
        return self.parallel.rank

    def validate(self) -> None:
        self.parallel.validate()
        if not self.distributed:
            return
        if self.rank < 0:
            raise ValueError("Distributed engine build requires a concrete rank")
        if self.config.num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "Decoder tensor parallel requires num_attention_heads divisible by tp_size "
                f"({self.config.num_attention_heads} vs {self.tp_size})")
        if self.config.num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                "Decoder tensor parallel requires num_key_value_heads divisible by tp_size "
                f"({self.config.num_key_value_heads} vs {self.tp_size})")
        mlp_size = int(self.weights.get("_mlp_size", self.config.intermediate_size))
        if mlp_size % self.tp_size != 0:
            raise ValueError(
                "Decoder tensor parallel requires intermediate size divisible by tp_size "
                f"({mlp_size} vs {self.tp_size})")

    def shard_weights(self) -> "WeightDict":
        self.validate()
        if not self.distributed:
            return self.weights

        out = type(self.weights)()
        for key, value in self.weights.items():
            if not isinstance(value, np.ndarray):
                out[key] = value
                continue
            if key.endswith((".w_q", ".w_k", ".w_v", ".q_bias", ".k_bias", ".v_bias")):
                out[key] = _slice_last_dim(value, self.rank, self.tp_size)
            elif key.endswith((".w_o", ".w_down")):
                out[key] = _slice_first_dim(value, self.rank, self.tp_size)
            elif key.endswith((".w_gate", ".w_up", ".w_fc1")):
                out[key] = _slice_last_dim(value, self.rank, self.tp_size)
            elif key.endswith(".w_fc2"):
                out[key] = _slice_first_dim(value, self.rank, self.tp_size)
            elif key.endswith((".q_norm", ".k_norm")) and value.size > self.config.head_dim:
                out[key] = _slice_first_dim(
                    value.reshape(-1, self.config.head_dim), self.rank, self.tp_size
                ).reshape(-1)
            else:
                out[key] = value

        out["_attention_size"] = int(self.weights["_attention_size"]) // self.tp_size
        out["_kv_attention_size"] = int(self.weights["_kv_attention_size"]) // self.tp_size
        out["_mlp_size"] = int(self.weights["_mlp_size"]) // self.tp_size
        out["_tensor_parallel_size"] = self.tp_size
        out["_tensor_parallel_rank"] = self.rank
        return out

    def local_num_attention_heads(self) -> int:
        return int(self.config.num_attention_heads) // self.tp_size

    def local_num_key_value_heads(self) -> int:
        return int(self.config.num_key_value_heads) // self.tp_size

    def join_row_parallel(self, network, tensor):
        if not self.distributed:
            return tensor
        return _add_all_reduce_sum(network, tensor, self.tp_size)

    def region_plans(self) -> list[RegionPlan]:
        if not self.distributed or self.recipe is None:
            return []
        component = self.recipe.component
        plans: list[RegionPlan] = []
        if self.recipe.regions_by_kind("self_attn"):
            plans.append(
                RegionPlan(
                    selector=f"{component}.layers[*].self_attn",
                    policy="tensor_parallel",
                    tp_size=self.tp_size,
                    mesh_axes=["tp"],
                )
            )
        if self.recipe.regions_by_kind("mlp"):
            plans.append(
                RegionPlan(
                    selector=f"{component}.layers[*].mlp",
                    policy="tensor_parallel",
                    tp_size=self.tp_size,
                    mesh_axes=["tp"],
                )
            )
        return plans

    def collective_plans(self) -> list[CollectivePlan]:
        if not self.distributed or self.recipe is None:
            return []
        out: list[CollectivePlan] = []
        for region in self.recipe.regions:
            if region.kind == "self_attn":
                out.append(
                    CollectivePlan(
                        id=f"{region.name}.out.all_reduce",
                        type="all_reduce",
                        group="tp",
                        op="sum",
                    )
                )
            elif region.kind == "mlp":
                out.append(
                    CollectivePlan(
                        id=f"{region.name}.down.all_reduce",
                        type="all_reduce",
                        group="tp",
                        op="sum",
                    )
                )
        return out


def standard_decoder_sharding_policy(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: ParallelConfig,
    *,
    recipe: "ModelRecipe | None" = None,
) -> StandardDecoderShardingPolicy:
    policy = StandardDecoderShardingPolicy(
        config=config,
        weights=weights,
        parallel=parallel,
        recipe=recipe,
    )
    policy.validate()
    return policy
