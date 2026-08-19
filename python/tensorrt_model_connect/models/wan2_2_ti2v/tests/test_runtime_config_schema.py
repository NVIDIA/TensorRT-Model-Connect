# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative Wan2.2 runtime-config contract."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import tensorrt_model_connect.runtime_config as runtime_config
from tensorrt_model_connect.runtime_config import (
    SchemaRegistry,
    resolve_cli_config,
)


def _load_schema_without_global_registration():
    captured = []
    original = runtime_config.register_schema
    runtime_config.register_schema = captured.append
    try:
        runpy.run_path(str(Path(__file__).parents[1] / "runtime_config_schema.py"))
    finally:
        runtime_config.register_schema = original
    assert len(captured) == 1
    return captured[0]


SCHEMA = _load_schema_without_global_registration()


def _registry() -> SchemaRegistry:
    registry = SchemaRegistry()
    registry.register_schema(SCHEMA)
    return registry


def test_defaults_are_portable_and_inert() -> None:
    bundle = resolve_cli_config(registry=_registry())

    assert bundle.get("wan2_2_ti2v", "easycache_enabled") is False
    assert bundle.get("wan2_2_ti2v", "easycache_threshold") == 0.02
    assert bundle.get("wan2_2_ti2v", "easycache_first_exact_steps") == 7
    assert bundle.get("wan2_2_ti2v", "easycache_last_exact_steps") == 2
    assert bundle.get("wan2_2_ti2v", "easycache_max_consecutive_reuse") == 1
    assert bundle.get("wan2_2_ti2v", "late_cfg_enabled") is False


def test_thor_receipt_values_resolve_through_set() -> None:
    bundle = resolve_cli_config(
        set_tokens=[
            "wan2_2_ti2v.easycache_enabled=true",
            "wan2_2_ti2v.easycache_threshold=1.0",
            "wan2_2_ti2v.easycache_max_consecutive_reuse=4",
            "wan2_2_ti2v.late_cfg_enabled=true",
        ],
        registry=_registry(),
    )

    assert bundle.get("wan2_2_ti2v", "easycache_enabled") is True
    assert bundle.get("wan2_2_ti2v", "easycache_threshold") == 1.0
    assert bundle.get("wan2_2_ti2v", "easycache_first_exact_steps") == 7
    assert bundle.get("wan2_2_ti2v", "easycache_last_exact_steps") == 2
    assert bundle.get("wan2_2_ti2v", "easycache_max_consecutive_reuse") == 4
    assert bundle.get("wan2_2_ti2v", "late_cfg_enabled") is True


@pytest.mark.parametrize(
    "token",
    [
        "wan2_2_ti2v.easycache_threshold=0",
        "wan2_2_ti2v.easycache_threshold=nan",
        "wan2_2_ti2v.easycache_first_exact_steps=-1",
        "wan2_2_ti2v.easycache_last_exact_steps=2147483648",
        "wan2_2_ti2v.easycache_max_consecutive_reuse=0",
    ],
)
def test_invalid_values_fail_before_pipeline_load(token: str) -> None:
    with pytest.raises(ValueError, match="Validator rejected"):
        resolve_cli_config(set_tokens=[token], registry=_registry())
