# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    REF2VA_PLAN_FILENAMES,
    default_workspace_limit_bytes,
)
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
        "audio_vae_decoder.plan": 32 << 30,
    }
    assert _workspace_limits(None) == DEFAULT_WORKSPACE_LIMIT_BYTES
    assert _workspace_limits(8) == {filename: 8 << 30 for filename in DEFAULT_WORKSPACE_LIMIT_BYTES}
    assert _positive_workspace_gib("8") == 8
    split_defaults = default_workspace_limit_bytes(first_block_cache=True)
    assert _workspace_limits(None, first_block_cache=True) == split_defaults
    assert set(split_defaults) == {
        "text_encoder.plan",
        "adaln_precompute.plan",
        "denoiser_head.plan",
        "denoiser_tail.plan",
        "denoiser_finish.plan",
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    }
    assert tuple(_workspace_limits(None, workflow="fl2va")) == FL2VA_PLAN_FILENAMES
    assert _workspace_limits(8, workflow="fl2va") == {
        filename: 8 << 30 for filename in FL2VA_PLAN_FILENAMES
    }
    assert tuple(_workspace_limits(None, workflow="ref2va")) == REF2VA_PLAN_FILENAMES
    assert _workspace_limits(8, workflow="ref2va") == {
        filename: 8 << 30 for filename in REF2VA_PLAN_FILENAMES
    }


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
    current_with_workflow = {
        "workflow": "t2va",
        "checkpoint_partition": "transformer",
        **current,
    }
    _validate_resume_identity(dict(current), current_with_workflow)

    previous = dict(current_with_workflow)
    previous["workspace_limit_bytes"] = _workspace_limits(8)
    with pytest.raises(ValueError, match="different workspace_limit_bytes"):
        _validate_resume_identity(previous, current_with_workflow)


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_conditioned_workflows_reject_first_block_cache(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
) -> None:
    from tests.e2e.models.minimax_h3 import build_native_components

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_native_components.py",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "plans"),
            "--source-revision",
            "1" * 40,
            "--workflow",
            workflow,
            "--first-block-cache",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        build_native_components.main()


@pytest.mark.parametrize(
    ("workflow", "first_block_cache"),
    [("t2va", False), ("t2va", True), ("fl2va", False), ("ref2va", False)],
)
def test_main_passes_cli_workspace_to_every_builder(
    tmp_path: Path,
    monkeypatch,
    capsys,
    workflow: str,
    first_block_cache: bool,
) -> None:
    from tests.e2e.models.minimax_h3 import build_native_components

    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    text_config = model / "text_encoder" / "config.json"
    text_config.parent.mkdir(parents=True)
    text_config.write_text("{}")
    for relative in FL2VA_PROCESSOR_ASSET_SECTIONS:
        asset = model / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("{}")
    output = tmp_path / "plans"
    observed = {}
    module_specs = {
        "text_encoder_builder": (("build_text_encoder_engine", "text_encoder.plan"),),
        "adaln_builder": (("build_adaln_precompute_engine", "adaln_precompute.plan"),),
        "dit_builder": (
            ("build_dit_engine", "denoiser.plan"),
            ("build_fl2va_dit_engine", "fl2va_denoiser.plan"),
            ("build_ref2va_dit_engine", "ref2va_denoiser.plan"),
            ("build_dit_head_engine", "denoiser_head.plan"),
            ("build_dit_tail_engine", "denoiser_tail.plan"),
            ("build_dit_finish_engine", "denoiser_finish.plan"),
        ),
        "vae_builder": (("build_vae_tile_decoder_engine", "vae_tile_decoder.plan"),),
        "audio_vae_builder": (
            ("build_audio_vae_encoder_engine", "audio_vae_encoder.plan"),
            ("build_audio_vae_decoder_engine", "audio_vae_decoder.plan"),
        ),
        "language_conditioner_builder": (
            ("build_language_conditioner_engine", "language_conditioner.plan"),
        ),
        "vision_conditioner_builder": (
            ("build_vision_conditioner_engine", "vision_conditioner.plan"),
        ),
        "vae_encoder_builder": (("build_vae_encoder_tile_engine", "vae_encoder_tile.plan"),),
    }
    package = "tensorrt_model_connect.families.minimax_h3"
    for module_name, builders in module_specs.items():
        fake_module = ModuleType(f"{package}.{module_name}")
        fake_module.checkpoint_keys = lambda *_args: ()
        if module_name == "dit_builder":
            fake_module.head_checkpoint_keys = lambda *_args: ()
            fake_module.tail_checkpoint_keys = lambda *_args: ()
            fake_module.finish_checkpoint_keys = lambda *_args: ()
        for builder_name, plan_name in builders:

            def fake_builder(*_args, _plan_name=plan_name, workspace_bytes=None, **_kwargs):
                observed[_plan_name] = {
                    "workspace_bytes": workspace_bytes,
                    "args": _args,
                    "kwargs": _kwargs,
                }
                return _plan_name.encode()

            setattr(fake_module, builder_name, fake_builder)
        if module_name == "vae_encoder_builder":

            def fake_tile_builder(*_args, num_frames, workspace_bytes=None, **_kwargs):
                plan_name = f"vae_encoder_tile_t{num_frames}.plan"
                observed[plan_name] = {
                    "workspace_bytes": workspace_bytes,
                    "args": _args,
                    "kwargs": {"num_frames": num_frames, **_kwargs},
                }
                return plan_name.encode()

            fake_module.build_vae_encoder_tile_engine = fake_tile_builder
        monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    monkeypatch.setattr(
        build_native_components,
        "checkpoint_snapshot_record",
        lambda _model, *, workflow: {
            "inventory_sha256": "a" * 64,
            "workflow": workflow,
        },
    )
    loaded_paths = []

    def load_state(path, *_args):
        loaded_paths.append(Path(path))
        return {}

    monkeypatch.setattr(build_native_components, "load_selected_component_state_dict", load_state)
    monkeypatch.setattr(build_native_components, "numpy_state", lambda state: state)
    partition_paths = []

    def validate_partition(path, *_args):
        partition_paths.append(Path(path))

    monkeypatch.setattr(
        build_native_components,
        "validate_component_key_partition",
        validate_partition,
    )
    argv = [
        "build_native_components.py",
        "--model-path",
        str(model),
        "--output-dir",
        str(output),
        "--source-revision",
        "1" * 40,
        "--workspace-gib",
        "8",
    ]
    if workflow != "t2va":
        argv.extend(("--workflow", workflow))
    if first_block_cache:
        argv.append("--first-block-cache")
    monkeypatch.setattr(sys, "argv", argv)

    assert build_native_components.main() == 0
    expected_limits = {
        filename: 8 << 30
        for filename in default_workspace_limit_bytes(
            first_block_cache=first_block_cache,
            workflow=workflow,
        )
    }
    assert {name: call["workspace_bytes"] for name, call in observed.items()} == expected_limits
    receipt = json.loads((output / "build_receipt.json").read_text())
    assert receipt["workflow"] == workflow
    expected_partition = "transformer_ref" if workflow == "ref2va" else "transformer"
    assert receipt["checkpoint_partition"] == expected_partition
    assert receipt["checkpoint_snapshot"]["workflow"] == workflow
    assert receipt["workspace_limit_bytes"] == expected_limits
    assert set(receipt["components"]) == set(expected_limits)
    assert receipt["profile"]["first_block_cache"] is first_block_cache
    expected_assets = {"tokenizer.json"}
    if workflow in {"fl2va", "ref2va"}:
        expected_assets.update(FL2VA_PROCESSOR_ASSET_SECTIONS)
        assert observed["language_conditioner.plan"]["kwargs"]["workflow"] == workflow
        assert observed["vision_conditioner.plan"]["kwargs"]["workflow"] == workflow
    if workflow == "fl2va":
        assert observed["fl2va_denoiser.plan"]["kwargs"]["checkpoint_subfolder"] == "transformer"
        assert observed["vae_encoder_tile_t1.plan"]["kwargs"] == {
            "num_frames": 1,
        }
    if workflow == "ref2va":
        assert observed["ref2va_denoiser.plan"]["kwargs"]["checkpoint_subfolder"] == (
            "transformer_ref"
        )
        assert observed["vae_encoder_tile_t1.plan"]["kwargs"] == {"num_frames": 1}
        assert observed["vae_encoder_tile_t17.plan"]["kwargs"] == {"num_frames": 17}
        assert observed["audio_vae_encoder.plan"]["args"] == (model / "audio_vae",)
        assert partition_paths == [model / "transformer_ref"]
        assert model / "transformer_ref" in loaded_paths
        assert model / "transformer" not in loaded_paths
    assert set(receipt["assets"]) == expected_assets
    capsys.readouterr()
