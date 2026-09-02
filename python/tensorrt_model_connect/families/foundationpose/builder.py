# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builders for the official FoundationPose NGC ONNX pair."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def _parser_errors(parser: Any) -> str:
    return "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))


def build_onnx_engine(
    path: str,
    *,
    kind: str,
    max_batch: int,
    precision: str,
    verbose: bool = False,
) -> bytes:
    if precision != "fp32":
        raise ValueError("FoundationPose ONNX engines currently support fp32 builds only")
    if kind not in {"refiner", "scorer"}:
        raise ValueError(f"Unsupported FoundationPose engine kind: {kind!r}")
    if max_batch <= 0:
        raise ValueError("FoundationPose max_batch must be positive")

    from tensorrt_model_connect import trt_compat

    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    parser = trt.OnnxParser(network, logger)
    model_path = Path(path)
    if not parser.parse(model_path.read_bytes()):
        raise RuntimeError(
            f"TensorRT could not parse FoundationPose {kind} ONNX {model_path}:\n"
            f"{_parser_errors(parser)}"
        )
    expected_outputs = {"output1", "output2"} if kind == "refiner" else {"output1"}
    inputs = {network.get_input(index).name: network.get_input(index) for index in range(network.num_inputs)}
    outputs = {network.get_output(index).name for index in range(network.num_outputs)}
    if set(inputs) != {"input1", "input2"} or outputs != expected_outputs:
        raise RuntimeError(
            f"FoundationPose {kind} ONNX contract mismatch: inputs={sorted(inputs)}, "
            f"outputs={sorted(outputs)}"
        )
    for name, tensor in inputs.items():
        if tuple(int(value) for value in tensor.shape)[1:] != (160, 160, 6):
            raise RuntimeError(
                f"FoundationPose {kind} input {name} has unsupported shape {tuple(tensor.shape)}"
            )

    profile = builder.create_optimization_profile()
    opt_batch = min(8, max_batch)
    for name in ("input1", "input2"):
        profile.set_shape(name, (1, 160, 160, 6), (opt_batch, 160, 160, 6),
                          (max_batch, 160, 160, 6))
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    workspace_gib = int(os.environ.get("TRTMC_FOUNDATIONPOSE_WORKSPACE_GIB", "8"))
    if workspace_gib <= 0 or workspace_gib > 64:
        raise ValueError("TRTMC_FOUNDATIONPOSE_WORKSPACE_GIB must be between 1 and 64")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(
            os.environ.get("TRTMC_FOUNDATIONPOSE_OPT_LEVEL", "4")
        )
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 0
    if verbose:
        print(
            f"[trtmc build] Building FoundationPose {kind} engine "
            f"(batch=1..{max_batch}, input=160x160x6, precision=fp32) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build the FoundationPose {kind} engine")
    return bytes(plan)
