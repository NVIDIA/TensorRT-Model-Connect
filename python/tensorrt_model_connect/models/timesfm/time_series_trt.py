# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM-owned native TensorRT utilities."""

from __future__ import annotations

import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import (
    _target_np_dtype,
)


trt = trt_compat.get_trt()


def maybe_return_replicated_tp_plan(weights: dict, parallel_config) -> bytes | None:
    if parallel_config is None or not getattr(parallel_config, "enabled", False):
        return None
    rank = int(getattr(parallel_config, "rank", -1))
    if rank > 0 and "_replicated_tp_engine_plan" in weights:
        return weights["_replicated_tp_engine_plan"]
    return None


def cache_replicated_tp_plan(weights: dict, parallel_config, plan: bytes) -> None:
    if parallel_config is not None and getattr(parallel_config, "enabled", False):
        weights["_replicated_tp_engine_plan"] = plan


def build_serialized_network(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    *,
    precision: str,
    verbose: bool = False,
    tag: str = "time_series",
) -> bytes:
    config = builder.create_builder_config()
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    if verbose:
        print(
            f"[trtmc build] {tag}: building native TRT network "
            f"({network.num_layers} layers, precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT {tag} engine build failed")
    return bytes(plan)


def create_network(*, verbose: bool = False) -> tuple[trt.Builder, trt.INetworkDefinition]:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    return builder, network


def add_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str = "fp32",
) -> trt.ITensor:
    target_dtype = _target_np_dtype(precision)
    target_trt_dtype = (
        trt.float16 if target_dtype == np.float16 else trt.float32)
    if inp.dtype != target_trt_dtype:
        inp = network.add_cast(inp, target_trt_dtype).get_output(0)
    w = np.ascontiguousarray(weight_out_in.T, dtype=target_dtype)
    out_features = int(weight_out_in.shape[0])
    out = graph_ops.add_matmul_rhs_constant(
        network,
        inp,
        int(weight_out_in.shape[1]),
        out_features,
        w,
        dtype=target_dtype,
    )
    if bias is not None:
        out = graph_ops.add_bias_sum(
            network,
            out,
            out_features,
            np.ascontiguousarray(bias, dtype=target_dtype),
            dtype=target_dtype,
        )
    return out


def add_scalar(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    return graph_ops.add_constant(
        network,
        shape,
        np.full(shape, value, dtype=dtype),
        dtype=dtype,
    )


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)
