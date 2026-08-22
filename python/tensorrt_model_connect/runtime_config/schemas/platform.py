# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for the ``platform`` namespace.

Infrastructure / host-level knobs that previously lived behind
``TRTMC_DATA_DIR``, ``TRTMC_TRT_LOG_STDERR`` and ``TRTMC_TRT_LOG_MIN_SEVERITY``.
Session and platform layers only — these are host/ops knobs, not bundled
configuration.
"""

from __future__ import annotations

from tensorrt_model_connect.runtime_config import (
    ConfigField,
    Layer,
    Schema,
    register_schema,
)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})


SCHEMA = Schema(
    namespace="platform",
    fields=(
        # Verbose TRT log streaming to stderr. When False, only warnings
        # and errors pass through the short "[trt] ..." path.
        ConfigField(
            name="trt_log_stderr",
            type_tag="bool",
            default=False,
            allowed_layers=_SESSION,
        ),
        # Minimum severity for the verbose TRT log stream. Values:
        # INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE.
        ConfigField(
            name="trt_log_min_severity",
            type_tag="string",
            default="INFO",
            allowed_layers=_SESSION,
            validator=lambda v: v in {
                "INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE",
            },
        ),
    ),
)


register_schema(SCHEMA)
