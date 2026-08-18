# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

import tensorrt_model_connect.runtime_config as runtime_config
import tensorrt_model_connect.engine_builder as engine_builder
from tensorrt_model_connect.build_cli import _cmd_build
from tensorrt_model_connect.build_cli import _resolved_config_values
from tensorrt_model_connect.families.sam2_hoi.plugin import plugin
from tensorrt_model_connect.runtime_config import SchemaRegistry, resolve_cli_config


def _schema():
    captured = []
    original = runtime_config.register_schema
    runtime_config.register_schema = captured.append
    try:
        runpy.run_path(str(Path(__file__).parents[1] / "runtime_config_schema.py"))
    finally:
        runtime_config.register_schema = original
    assert len(captured) == 1
    return captured[0]


def _registry():
    registry = SchemaRegistry()
    registry.register_schema(_schema())
    return registry


def test_phase_a_default_is_inert() -> None:
    bundle = resolve_cli_config(registry=_registry())
    assert bundle.get("sam2_hoi", "phase_a_pafpn") is False


def test_phase_a_exact_cli_set_is_resolved() -> None:
    bundle = resolve_cli_config(
        set_tokens=["sam2_hoi.phase_a_pafpn=true"],
        registry=_registry(),
    )
    assert bundle.get("sam2_hoi", "phase_a_pafpn") is True
    config = type(
        "Config", (), {"raw": {"_family_build_options": _resolved_config_values(bundle)}}
    )()
    assert plugin._phase_a_enabled(config) is True


def test_phase_a_cli_forwards_the_resolved_model_option(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **kwargs: captured.update(kwargs),
    )
    output = tmp_path / "sam2-hoi-phase-a.bundle"
    args = SimpleNamespace(
        model="/reviewed/sam2-hoi",
        output=str(output),
        max_cache_length=256,
        precision="bf16",
        method="trt",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        fp8=False,
        fp8_scales=None,
        config=None,
        set_flags=["sam2_hoi.phase_a_pafpn=true"],
        _skip_profile_resolution=True,
    )

    assert _cmd_build(args) == 0
    assert captured["family_build_options"]["sam2_hoi"] == {"phase_a_pafpn": True}
    assert (tmp_path / "sam2-hoi-phase-a.effective_config.json").is_file()


@pytest.mark.parametrize("value", ["true", "yes", "1"])
def test_phase_a_boolean_cli_aliases_resolve_to_true(value: str) -> None:
    bundle = resolve_cli_config(
        set_tokens=[f"sam2_hoi.phase_a_pafpn={value}"],
        registry=_registry(),
    )
    assert bundle.get("sam2_hoi", "phase_a_pafpn") is True


@pytest.mark.parametrize("value", ["maybe", "2", "null"])
def test_phase_a_non_boolean_cli_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_cli_config(
            set_tokens=[f"sam2_hoi.phase_a_pafpn={value}"],
            registry=_registry(),
        )
