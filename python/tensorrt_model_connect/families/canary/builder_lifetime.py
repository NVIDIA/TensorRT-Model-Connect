# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-lifetime resources shared by Canary TensorRT builders."""

from __future__ import annotations

from typing import Any


_PROCESS_LOGGERS: dict[object, Any] = {}


def get_process_trt_logger(trt_module: Any, *, verbose: bool) -> Any:
    """Return the logger retained for this TensorRT module's process lifetime."""
    logger = _PROCESS_LOGGERS.get(trt_module)
    if logger is None:
        logger_type = trt_module.Logger
        severity = logger_type.VERBOSE if verbose else logger_type.WARNING
        logger = logger_type(severity)
        _PROCESS_LOGGERS[trt_module] = logger
    return logger
