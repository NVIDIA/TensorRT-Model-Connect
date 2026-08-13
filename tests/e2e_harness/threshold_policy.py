# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository policy for model-owned E2E threshold sidecars."""

from __future__ import annotations

from typing import Any


def requires_threshold_sidecar(testcase: dict[str, Any]) -> bool:
    """Return whether a testcase consumes configurable numeric thresholds.

    Fixed runtime-invariant contracts assert exact build and execution facts in
    their model-owned verifier and deliberately ignore ``ThresholdProfile``.
    Other invariant-only contracts may still use numeric quality or shape
    thresholds, so the exemption stays intentionally narrow.
    """
    return not (
        testcase.get("reference_backend") == "invariant_only"
        and testcase.get("oracle_level") == "L4_invariants"
        and testcase.get("user_contract") == "runtime_invariants"
    )
