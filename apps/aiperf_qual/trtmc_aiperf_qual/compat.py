# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read earlier configurations and reports without retaining two executors."""

from __future__ import annotations

from typing import Any, Mapping


def configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    policy = dict(result.get("performance") or {})
    legacy = policy.pop("l1", None)
    optional = policy.pop("l2", None)
    if legacy is not None:
        policy.update(legacy)
    if "performance" in result:
        result["performance"] = policy
    if optional is not None and "service_metrics" not in result:
        result["service_metrics"] = optional
    return result


def report(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    legacy = result.pop("performance_l1", [])
    optional = result.pop("performance_l2", {})
    result.setdefault("performance", legacy)
    result.setdefault("service_metrics", optional)
    return result
