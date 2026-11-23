# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile Qwen projections without values, then serialize a complete refitted plan."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import tensorrt as trt


_SESSION = ContextVar("qwen_weightless_build", default=None)


@contextmanager
def weightless_build():
    """Retain refit sources for one engine and restore the scope on failure."""
    token = _SESSION.set({})
    try:
        yield
    finally:
        _SESSION.reset(token)


def add_placeholder(network, shape, array, name):
    """Only explicitly named large projection matrices may omit their values."""
    values = _SESSION.get()
    if values is None or name is None or array.nbytes < 1024 * 1024:
        return None
    if name in values:
        raise ValueError(f"Duplicate Qwen refit weight: {name}")
    concrete = trt.Weights(array)
    layer = network.add_constant(shape, trt.Weights(concrete.dtype, 0, concrete.size))
    if layer is None:
        raise RuntimeError(f"Cannot create Qwen placeholder: {name}")
    # Name the canonical descriptor: equal-sized null descriptors must still
    # refer to distinct weights in the network and in the refitter catalog.
    if not network.set_weights_name(layer.weights, name):
        raise RuntimeError(f"Cannot name Qwen refit weight: {name}")
    if not network.mark_weights_refittable(name):
        raise RuntimeError(f"Cannot mark Qwen refit weight: {name}")
    values[name] = array
    return layer


def build_serialized_network(builder, network, config, logger) -> bytes:
    """Refit before returning bytes, preserving the native runtime contract."""
    session = _SESSION.get()
    if session is not None and not session:
        raise RuntimeError("Qwen weightless build found no named projection weights")
    values = session or {}
    enabled = session is not None
    if enabled:
        config.set_flag(trt.BuilderFlag.REFIT_INDIVIDUAL)
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Qwen TensorRT engine build failed")
    if enabled:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("Cannot deserialize Qwen weightless plan")
        refitter = trt.Refitter(engine, logger)
        catalog = list(refitter.get_all_weights())
        if len(catalog) != len(set(catalog)) or set(catalog) != set(values):
            raise RuntimeError(f"Qwen refit catalog mismatch: {set(catalog) ^ set(values)}")
        handles = []
        for name, array in values.items():
            weights = trt.Weights(array)
            prototype = refitter.get_weights_prototype(name)
            if prototype.size != weights.size or prototype.dtype != weights.dtype:
                raise RuntimeError(f"Qwen refit shape or dtype mismatch: {name}")
            handles.append(weights)
            if not refitter.set_named_weights(name, weights):
                raise RuntimeError(f"Cannot set Qwen refit weight: {name}")
        missing = list(refitter.get_missing_weights())
        if missing:
            raise RuntimeError(f"Qwen refit is missing weights: {missing}")
        if not refitter.refit_cuda_engine():
            raise RuntimeError("Qwen weight refit failed")
        serialization = engine.create_serialization_config()
        if serialization is None:
            raise RuntimeError("Cannot create Qwen engine serialization config")
        serialization.clear_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS)
        if serialization.get_flag(trt.SerializationFlag.EXCLUDE_WEIGHTS):
            raise RuntimeError("Qwen engine serialization still excludes weights")
        plan = engine.serialize_with_config(serialization)
        if plan is None:
            raise RuntimeError("Cannot serialize refitted Qwen plan")
    return bytes(plan)
