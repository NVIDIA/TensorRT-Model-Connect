# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local mechanics shared by the OpenFold3 engine builders."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def create_network(*, verbose: bool) -> tuple[Any, Any, Any, Any]:
    """Create a network and return its logger so callers retain its lifetime."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return trt, builder, builder.create_network(flags), logger


def build_plan(builder: Any, network: Any, *, workspace_bytes: int) -> tuple[Any, float]:
    """Build a plan with the single-device deterministic OpenFold3 policy."""
    import tensorrt as trt

    config = builder.create_builder_config()
    config.builder_optimization_level = 4
    config.avg_timing_iterations = 8
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build an OpenFold3 engine")
    return plan, build_seconds


def write_plan(path: Path, plan: Any) -> int:
    """Persist one nonempty plan and return its byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plan)
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError("TensorRT produced an empty OpenFold3 engine")
    return size
