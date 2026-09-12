# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate declared reference timing without selecting model-family policy."""

from __future__ import annotations

from typing import Any, Mapping


def timing_contract(
    *,
    runner: str,
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the existing declaration; the consumer's public Task boundary is fixed."""
    fields = ("timing_scope", "input_preparation_included", "asset_loading_included")
    missing = [name for name in fields if name not in declared]
    if missing:
        raise ValueError("timing declaration requires " + ", ".join(missing))
    scope = declared["timing_scope"]
    preparation = declared["input_preparation_included"]
    assets = declared["asset_loading_included"]
    if not isinstance(scope, str) or type(preparation) is not bool or type(assets) is not bool:
        raise ValueError("timing scope must be a string and inclusion flags must be booleans")
    if runner == "hf-transformers":
        valid = scope == "public_operation_call_wall" and preparation and not assets
    elif runner == "task-reference":
        valid = (scope == "task-model-call-wall" and not preparation and not assets) or (
            scope == "task-pipeline-call-wall" and preparation
        )
    else:
        raise ValueError(f"unsupported baseline runner: {runner}")
    if not valid:
        raise ValueError(f"inconsistent {runner} timing declaration: " + str({name: declared[name] for name in fields}))
    return {
        "timing_scope": scope,
        "candidate_timing_scope": "public_task_call_wall",
        "input_preparation_included": preparation,
        "asset_loading_included": assets,
    }
