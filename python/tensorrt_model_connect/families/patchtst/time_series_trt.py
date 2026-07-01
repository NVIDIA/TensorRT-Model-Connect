# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTST family-owned time-series TensorRT surface."""

from tensorrt_model_connect import trt_compat

from . import graph_ops  # noqa: F401
from .checkpoint_mapper import WeightDict  # noqa: F401


trt = trt_compat.get_trt()
_PROCESS_LOGGER: trt.Logger | None = None


def _get_process_logger(*, verbose: bool) -> trt.Logger:
    global _PROCESS_LOGGER
    if _PROCESS_LOGGER is None:
        _PROCESS_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    return _PROCESS_LOGGER


def create_network(*, verbose: bool = False) -> tuple[trt.Builder, trt.INetworkDefinition]:
    builder = trt.Builder(_get_process_logger(verbose=verbose))
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    return builder, network
