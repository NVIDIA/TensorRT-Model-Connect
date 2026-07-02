# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST-owned native TensorRT utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _target_np_dtype,
    _transpose_2d,
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
    if precision == "fp32" and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision in {"fp16", "bf16"}:
        config.set_flag(trt.BuilderFlag.FP16)

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


def load_named_tensors(
    model_dir: str | Path,
    names: Iterable[str],
    *,
    precision: str = "fp32",
    transpose_2d: bool = False,
    optional: bool = False,
) -> WeightDict:
    readers = _open_safetensors(Path(model_dir))
    target_dtype = _target_np_dtype(precision)
    weights = WeightDict()
    for name in names:
        if not _has_tensor(readers, name):
            if optional:
                continue
            raise KeyError(f"Tensor not found: {name}")
        arr = _load_tensor(readers, name)
        if transpose_2d and arr.ndim == 2:
            weights[name] = _transpose_2d(arr, name, precision=precision)
        else:
            dtype = np.float32 if (
                name.endswith(("running_mean", "running_var"))
                or ".norm" in name
                or "layernorm" in name
                or "layer_norm" in name
            ) else target_dtype
            weights[name] = np.ascontiguousarray(arr, dtype=dtype)
    return weights


def add_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str = "fp32",
) -> trt.ITensor:
    target_dtype = _target_np_dtype(precision)
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


def add_std_scale(
    network: trt.INetworkDefinition,
    data: trt.ITensor,
    observed: trt.ITensor,
    *,
    channels: int,
    minimum_scale: float,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    mask = network.add_cast(observed, trt.float32).get_output(0)
    denominator = network.add_reduce(
        mask, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    one = add_scalar(network, (1, 1, channels), 1.0)
    denominator = network.add_elementwise(
        denominator, one, trt.ElementWiseOperation.MAX
    ).get_output(0)

    masked = network.add_elementwise(data, mask, trt.ElementWiseOperation.PROD).get_output(0)
    summed = network.add_reduce(
        masked, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    loc = network.add_elementwise(
        summed, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)

    centered = network.add_elementwise(
        data, loc, trt.ElementWiseOperation.SUB
    ).get_output(0)
    centered_masked = network.add_elementwise(
        centered, mask, trt.ElementWiseOperation.PROD
    ).get_output(0)
    sq = network.add_elementwise(
        centered_masked, centered_masked, trt.ElementWiseOperation.PROD
    ).get_output(0)
    var_sum = network.add_reduce(
        sq, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    ).get_output(0)
    var = network.add_elementwise(
        var_sum, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    eps = add_scalar(network, (1, 1, channels), minimum_scale)
    var_eps = network.add_elementwise(var, eps, trt.ElementWiseOperation.SUM).get_output(0)
    scale = network.add_unary(var_eps, trt.UnaryOperation.SQRT).get_output(0)
    scaled = network.add_elementwise(
        centered, scale, trt.ElementWiseOperation.DIV
    ).get_output(0)
    return scaled, loc, scale


def add_patchify(
    network: trt.INetworkDefinition,
    values: trt.ITensor,
    *,
    context_length: int,
    channels: int,
    patch_length: int,
    patch_stride: int,
    num_patches: int,
) -> trt.ITensor:
    new_sequence_length = patch_length + patch_stride * (num_patches - 1)
    sequence_start = context_length - new_sequence_length
    if sequence_start < 0:
        raise ValueError("Patch configuration exceeds context length")

    channel_tensors: list[trt.ITensor] = []
    for channel in range(channels):
        patch_tensors: list[trt.ITensor] = []
        for patch_idx in range(num_patches):
            start = sequence_start + patch_idx * patch_stride
            sliced = network.add_slice(
                values,
                start=(0, start, channel),
                shape=(1, patch_length, 1),
                stride=(1, 1, 1),
            ).get_output(0)
            shuf = network.add_shuffle(sliced)
            shuf.first_transpose = (0, 2, 1)
            shuf.reshape_dims = (1, 1, 1, patch_length)
            patch_tensors.append(shuf.get_output(0))
        cat_patches = network.add_concatenation(patch_tensors)
        cat_patches.axis = 2
        channel_tensors.append(cat_patches.get_output(0))
    cat_channels = network.add_concatenation(channel_tensors)
    cat_channels.axis = 1
    return cat_channels.get_output(0)


def add_batch_norm_last_dim(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    width: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float,
) -> trt.ITensor:
    denom = np.sqrt(np.asarray(running_var, dtype=np.float32) + float(eps))
    scale = np.asarray(gamma, dtype=np.float32) / denom
    bias = np.asarray(beta, dtype=np.float32) - np.asarray(running_mean, dtype=np.float32) * scale
    rank = len(tuple(inp.shape))
    shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    scale_t = graph_ops.add_constant(network, shape, scale.reshape(shape), dtype=np.float32)
    bias_t = graph_ops.add_constant(network, shape, bias.reshape(shape), dtype=np.float32)
    scaled = network.add_elementwise(inp, scale_t, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(scaled, bias_t, trt.ElementWiseOperation.SUM).get_output(0)


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def add_gelu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    if hasattr(trt.UnaryOperation, "ERF"):
        inv_sqrt2 = add_scalar(
            network, (1,) * len(tuple(inp.shape)), 1.0 / np.sqrt(2.0),
            dtype=np.float32)
        half = add_scalar(
            network, (1,) * len(tuple(inp.shape)), 0.5, dtype=np.float32)
        one = add_scalar(
            network, (1,) * len(tuple(inp.shape)), 1.0, dtype=np.float32)
        scaled = network.add_elementwise(
            inp, inv_sqrt2, trt.ElementWiseOperation.PROD).get_output(0)
        erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        one_plus = network.add_elementwise(
            erf, one, trt.ElementWiseOperation.SUM).get_output(0)
        half_x = network.add_elementwise(
            inp, half, trt.ElementWiseOperation.PROD).get_output(0)
        return network.add_elementwise(
            half_x, one_plus, trt.ElementWiseOperation.PROD).get_output(0)
    return graph_ops.add_gelu_new(network, inp, dtype=np.float32)


def add_squareplus(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    four = add_scalar(network, (1,) * len(tuple(inp.shape)), 4.0, dtype=np.float32)
    half = add_scalar(network, (1,) * len(tuple(inp.shape)), 0.5, dtype=np.float32)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD).get_output(0)
    sq_plus = network.add_elementwise(sq, four, trt.ElementWiseOperation.SUM).get_output(0)
    root = network.add_unary(sq_plus, trt.UnaryOperation.SQRT).get_output(0)
    summed = network.add_elementwise(inp, root, trt.ElementWiseOperation.SUM).get_output(0)
    return network.add_elementwise(summed, half, trt.ElementWiseOperation.PROD).get_output(0)
