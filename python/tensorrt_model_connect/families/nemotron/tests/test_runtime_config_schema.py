# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron build-time configuration contract."""

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


def test_builder_workspace_is_opt_in_and_typed() -> None:
    defaults = resolve_cli_config(registry=_registry())
    qualified = resolve_cli_config(
        set_tokens=["nemotron_decoder.builder_workspace_gib=2"],
        registry=_registry(),
    )

    assert defaults.get("nemotron_decoder", "builder_workspace_gib") == 0
    assert qualified.get("nemotron_decoder", "builder_workspace_gib") == 2


@pytest.mark.parametrize(
    "token",
    [
        "nemotron_decoder.builder_workspace_gib=-1",
        "nemotron_decoder.builder_workspace_gib=1.5",
    ],
)
def test_invalid_builder_workspace_fails_before_engine_build(token: str) -> None:
    with pytest.raises(ValueError):
        resolve_cli_config(set_tokens=[token], registry=_registry())
