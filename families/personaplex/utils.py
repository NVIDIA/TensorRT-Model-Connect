# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic helpers for TensorRT engine builders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from inspect import signature
from traceback import clear_frames
from typing import Callable

import numpy as np
import tensorrt as trt

from . import graph_ops


_PROCESS_LOGGER: trt.Logger | None = None


def _get_process_logger(*, verbose: bool) -> trt.Logger:
    """Return the logger that must outlive every TensorRT builder in this process."""
    global _PROCESS_LOGGER
    if _PROCESS_LOGGER is None:
        _PROCESS_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    return _PROCESS_LOGGER


@dataclass
class BuilderContext:
    """TensorRT objects shared by engine builders.

    TensorRT requires child objects to be released before their factory and
    requires the logger to outlive every object created through it.  Keeping
    the release order here avoids relying on Python frame-local destruction
    order, which does not satisfy that contract.
    """

    logger: trt.Logger | None
    builder: trt.Builder | None
    network: trt.INetworkDefinition | None
    config: trt.IBuilderConfig | None
    _closed: bool = False

    def close(self) -> None:
        """Release TensorRT objects once, in child-to-parent order."""
        if self._closed:
            return
        self._closed = True
        self.config = None
        self.network = None
        self.builder = None
        self.logger = None


BuilderContextFactory = Callable[[], BuilderContext]


def create_builder_context(
    *,
    verbose: bool,
    workspace_bytes: int | None = None,
    strongly_typed: bool = True,
    explicit_batch: bool = False,
    disable_tf32: bool = False,
    builder_optimization_level: int | None = None,
    max_num_tactics: int | None = None,
) -> BuilderContext:
    """Create a TensorRT builder, network, and config with common defaults."""
    context = BuilderContext(
        logger=_get_process_logger(verbose=verbose),
        builder=None,
        network=None,
        config=None,
    )
    try:
        context.builder = trt.Builder(context.logger)
        del explicit_batch
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
        context.network = context.builder.create_network(flags)
        context.config = context.builder.create_builder_config()
        if workspace_bytes is not None:
            context.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        if disable_tf32:
            context.config.clear_flag(trt.BuilderFlag.TF32)
        if builder_optimization_level is not None:
            context.config.builder_optimization_level = builder_optimization_level
        if max_num_tactics is not None:
            context.config.max_num_tactics = max_num_tactics
        return context
    except BaseException:
        context.close()
        raise


def with_builder_context(
    *,
    workspace_bytes: int | None = None,
    strongly_typed: bool = True,
    explicit_batch: bool = False,
    disable_tf32: bool = False,
    builder_optimization_level: int | None = None,
    max_num_tactics: int | None = None,
):
    """Give a builder function a lazy context with guaranteed cleanup.

    The context is created only when the wrapped function asks for it.  This
    matters for the standard decoder, which can dispatch to another builder
    before it needs its own TensorRT objects.  The wrapped function's frame is
    fully unwound before ``close`` runs, so local network tensors and aliases
    cannot outlive the ordered context teardown.
    """

    def decorate(function):
        public_signature = signature(function).replace(
            parameters=[
                parameter
                for name, parameter in signature(function).parameters.items()
                if name != "_builder_context_factory"
            ]
        )

        @wraps(function)
        def wrapped(*args, **kwargs):
            context: BuilderContext | None = None

            def context_factory() -> BuilderContext:
                nonlocal context
                if context is None:
                    context = create_builder_context(
                        verbose=bool(kwargs.get("verbose", False)),
                        workspace_bytes=workspace_bytes,
                        strongly_typed=strongly_typed,
                        explicit_batch=explicit_batch,
                        disable_tf32=disable_tf32,
                        builder_optimization_level=builder_optimization_level,
                        max_num_tactics=max_num_tactics,
                    )
                return context

            try:
                return function(
                    *args,
                    _builder_context_factory=context_factory,
                    **kwargs,
                )
            except BaseException as error:
                # A live traceback retains the failed builder frame and its
                # local TensorRT aliases.  Clear finished frames before the
                # context teardown so the logger still outlives every child.
                clear_frames(error.__traceback__)
                raise
            finally:
                if context is not None:
                    context.close()

        wrapped._trtmc_ordered_builder_context = True
        wrapped.__signature__ = public_signature
        return wrapped

    return decorate


def const_in_work_dtype(
    network: trt.INetworkDefinition,
    shape: tuple,
    values: np.ndarray,
    work_np_dtype: np.dtype,
    work_trt_dtype: trt.DataType,
) -> trt.ITensor:
    """Create a constant in storage dtype and cast it to runtime dtype."""
    const = graph_ops.add_constant(network, shape, values, dtype=work_np_dtype)
    if const.dtype != work_trt_dtype:
        const = network.add_cast(const, work_trt_dtype).get_output(0)
    return const


def norm_multi(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype,
) -> trt.ITensor:
    """Apply LayerNorm or RMSNorm from the same call site."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden, dtype=np.float32)
        return graph_ops.add_layer_norm(network, inp, hidden, gamma, beta, eps_tensor, dtype=dtype)
    return graph_ops.add_rms_norm(network, inp, hidden, gamma, eps_tensor, dtype=dtype)
