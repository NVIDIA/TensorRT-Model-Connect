# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt-owned native TensorRT utilities."""

from __future__ import annotations

import sys

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import (
    _target_np_dtype,
)




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
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    return builder, network


def add_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight_out_in: np.ndarray,
    bias: np.ndarray | None,
    *,
    precision: str = "fp32",
    fp32_accumulation: bool = False,
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
        fp32_accumulation=fp32_accumulation,
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


def add_named_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    tensor.name = name
    network.mark_output(tensor)


def add_gelu(network: trt.INetworkDefinition, inp: trt.ITensor) -> trt.ITensor:
    dtype = np.float16 if inp.dtype == trt.float16 else np.float32
    inv_sqrt2 = add_scalar(
        network, (1,) * len(tuple(inp.shape)), 1.0 / np.sqrt(2.0),
        dtype=dtype)
    half = add_scalar(
        network, (1,) * len(tuple(inp.shape)), 0.5, dtype=dtype)
    one = add_scalar(
        network, (1,) * len(tuple(inp.shape)), 1.0, dtype=dtype)
    scaled = network.add_elementwise(
        inp, inv_sqrt2, trt.ElementWiseOperation.PROD).get_output(0)
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    one_plus = network.add_elementwise(
        erf, one, trt.ElementWiseOperation.SUM).get_output(0)
    half_x = network.add_elementwise(
        inp, half, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(
        half_x, one_plus, trt.ElementWiseOperation.PROD).get_output(0)
