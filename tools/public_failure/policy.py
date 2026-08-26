# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed public-name mappings for protected CI failure reports."""

from __future__ import annotations

from typing import Final


POLICY_VERSION: Final = "2026-08-26"

FAILURE_CLASS_BY_INTERNAL_TYPE: Final = {
    "compare_fail": "accuracy_regression",
    "determinism_fail": "determinism_failure",
    "build_fail": "build_error",
    "trt_run_fail": "runtime_error",
    "reference_run_fail": "reference_error",
    "artifact_write_fail": "harness_error",
    "infrastructure_error": "infrastructure_error",
    "timeout": "timeout",
    "out_of_memory": "out_of_memory",
}

PUBLIC_STAGE_BY_INTERNAL_STAGE: Final = {
    "precheck": "precheck",
    "build": "build",
    "trt_run": "trt-run",
    "reference_run": "reference-run",
    "compare": "comparison",
    "determinism": "determinism",
    "artifact_write": "artifact-write",
    "model-proof": "model-proof",
}

# This intentionally starts small. Adding a value is a disclosure-policy change.
PUBLIC_MODELS: Final = frozenset(
    {
        "chronos_bolt",
        "codegen",
        "llama",
        "patchtsmixer",
        "personaplex",
        "sam3",
        "segformer",
    }
)
PUBLIC_BACKENDS: Final = frozenset({"native", "tensorrt", "trtfb"})
PUBLIC_GPU_TYPES: Final = frozenset({"A100", "H100", "H200", "L4", "GB200", "GB300"})

PUBLIC_REASON_CODES: Final = frozenset(
    {
        "artifact_write_failed",
        "build_failed",
        "determinism_check_failed",
        "infrastructure_failed",
        "metric_threshold_exceeded",
        "out_of_memory",
        "reference_failed",
        "runtime_failed",
        "timed_out",
        "unknown",
    }
)

PUBLIC_METRIC_NAMES: Final = frozenset(
    {
        "max_absolute_error",
        "max_relative_l2",
        "mean_relative_l2",
        "prediction_agreement_rate",
        "sample_agreement_rate",
    }
)
PUBLIC_METRIC_OPERATORS: Final = frozenset({"<", "<=", "==", ">=", ">"})


def public_failure_class(value: object) -> str:
    return FAILURE_CLASS_BY_INTERNAL_TYPE.get(str(value), "unknown")


def public_stage(value: object) -> str:
    return PUBLIC_STAGE_BY_INTERNAL_STAGE.get(str(value), "protected-ci")


def public_model(value: object) -> str:
    candidate = str(value)
    return candidate if candidate in PUBLIC_MODELS else "other-model"


def public_backend(value: object) -> str:
    candidate = str(value)
    return candidate if candidate in PUBLIC_BACKENDS else "other-backend"


def public_gpu_type(value: object) -> str:
    candidate = str(value)
    return candidate if candidate in PUBLIC_GPU_TYPES else "protected-gpu"


def public_reason_code(value: object) -> str:
    candidate = str(value)
    return candidate if candidate in PUBLIC_REASON_CODES else "unknown"
