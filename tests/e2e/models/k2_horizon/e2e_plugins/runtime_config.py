# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-driven runtime configuration helpers for K2-Horizon E2E cases."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .contracts import E2ECase


def _runtime_config(case: E2ECase) -> dict[str, Any]:
    config = case.metadata.get("runtime_config")
    if isinstance(config, dict):
        return config
    config = case.inputs.get("runtime_config")
    return config if isinstance(config, dict) else {}


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(name, nested)
    elif prefix:
        yield prefix, value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def runtime_config_set_tokens(case: E2ECase) -> list[str]:
    return [f"{name}={_format_value(value)}" for name, value in _flatten("", _runtime_config(case))]
