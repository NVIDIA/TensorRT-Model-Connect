# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative MiniMax-H3 workflow-selection contract."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

import tensorrt_model_connect.runtime_config as runtime_config
from tensorrt_model_connect.runtime_config import SchemaRegistry, resolve_cli_config


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


def test_workflow_defaults_to_legacy_t2va() -> None:
    bundle = resolve_cli_config(registry=_registry())
    assert bundle.get("minimax_h3", "workflow") == "t2va"
    assert bundle.get("minimax_h3", "first_block_cache") is False


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
def test_declared_workflow_routes_through_family_config(workflow: str) -> None:
    bundle = resolve_cli_config(
        set_tokens=[f"minimax_h3.workflow={workflow}"],
        registry=_registry(),
    )
    assert bundle.get("minimax_h3", "workflow") == workflow


def test_unknown_workflow_fails_before_checkpoint_load() -> None:
    with pytest.raises(ValueError, match="Validator rejected"):
        resolve_cli_config(
            set_tokens=["minimax_h3.workflow=unknown"],
            registry=_registry(),
        )
