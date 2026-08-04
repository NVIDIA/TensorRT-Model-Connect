# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tensorrt_model_connect.families.minimax_h3.config import DEFAULT_WORKSPACE_LIMIT_BYTES
from tests.e2e.models.minimax_h3.build_native_components import (
    _positive_workspace_gib,
    _validate_resume_identity,
    _workspace_limits,
)


def test_workspace_limits_preserve_defaults_or_apply_exact_override() -> None:
    assert DEFAULT_WORKSPACE_LIMIT_BYTES == {
        "text_encoder.plan": 96 << 30,
        "adaln_precompute.plan": 64 << 30,
        "denoiser.plan": 96 << 30,
        "vae_tile_decoder.plan": 96 << 30,
    }
    assert _workspace_limits(None) == DEFAULT_WORKSPACE_LIMIT_BYTES
    assert _workspace_limits(8) == {filename: 8 << 30 for filename in DEFAULT_WORKSPACE_LIMIT_BYTES}
    assert _positive_workspace_gib("8") == 8


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "bad"])
def test_workspace_gib_parser_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        _positive_workspace_gib(raw)


def test_resume_identity_binds_workspace_limits() -> None:
    current = {
        "checkpoint_revision": "checkpoint",
        "source_revision": "source",
        "builder_source_sha256": "builder",
        "build_helper_sha256": "helper",
        "checkpoint_snapshot": {"snapshot": True},
        "profile": {"rows": 1},
        "assets": {"tokenizer.json": {}},
        "workspace_limit_bytes": _workspace_limits(None),
    }
    _validate_resume_identity(dict(current), current)

    previous = dict(current)
    previous["workspace_limit_bytes"] = _workspace_limits(8)
    with pytest.raises(ValueError, match="different workspace_limit_bytes"):
        _validate_resume_identity(previous, current)


def test_main_passes_cli_workspace_to_every_builder(tmp_path: Path, monkeypatch, capsys) -> None:
    from tests.e2e.models.minimax_h3 import build_native_components

    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    output = tmp_path / "plans"
    observed = {}
    module_specs = {
        "text_encoder_builder": ("build_text_encoder_engine", "text_encoder.plan"),
        "adaln_builder": ("build_adaln_precompute_engine", "adaln_precompute.plan"),
        "dit_builder": ("build_dit_engine", "denoiser.plan"),
        "vae_builder": ("build_vae_tile_decoder_engine", "vae_tile_decoder.plan"),
    }
    package = "tensorrt_model_connect.families.minimax_h3"
    for module_name, (builder_name, plan_name) in module_specs.items():
        fake_module = ModuleType(f"{package}.{module_name}")
        fake_module.checkpoint_keys = lambda *_args: ()

        def fake_builder(*_args, _plan_name=plan_name, workspace_bytes=None, **_kwargs):
            observed[_plan_name] = workspace_bytes
            return _plan_name.encode()

        setattr(fake_module, builder_name, fake_builder)
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    monkeypatch.setattr(
        build_native_components,
        "checkpoint_snapshot_record",
        lambda _model: {"inventory_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        build_native_components,
        "load_selected_component_state_dict",
        lambda *_args: {},
    )
    monkeypatch.setattr(build_native_components, "numpy_state", lambda state: state)
    monkeypatch.setattr(
        build_native_components,
        "validate_component_key_partition",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_native_components.py",
            "--model-path",
            str(model),
            "--output-dir",
            str(output),
            "--source-revision",
            "1" * 40,
            "--workspace-gib",
            "8",
        ],
    )

    assert build_native_components.main() == 0
    assert observed == {filename: 8 << 30 for filename in DEFAULT_WORKSPACE_LIMIT_BYTES}
    receipt = json.loads((output / "build_receipt.json").read_text())
    assert receipt["workspace_limit_bytes"] == observed
    assert set(receipt["components"]) == set(DEFAULT_WORKSPACE_LIMIT_BYTES)
    capsys.readouterr()
