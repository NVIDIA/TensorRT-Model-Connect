# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    DEFAULT_WORKSPACE_LIMIT_BYTES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
    SOL_ENGINE_1344X768_124_TO_345F,
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
        "vision_encoder.plan": 32 << 30,
        "adaln_precompute.plan": 64 << 30,
        "denoiser.plan": 96 << 30,
        "fl2va_keyframe_vae_encoder.plan": 32 << 30,
        "vae_tile_decoder.plan": 96 << 30,
        "audio_vae_decoder.plan": 96 << 30,
    }
    assert _workspace_limits(None) == DEFAULT_WORKSPACE_LIMIT_BYTES
    assert _workspace_limits(8) == {filename: 8 << 30 for filename in DEFAULT_WORKSPACE_LIMIT_BYTES}
    assert _positive_workspace_gib("8") == 8
    split_defaults = default_workspace_limit_bytes(first_block_cache=True)
    assert _workspace_limits(None, first_block_cache=True) == split_defaults
    assert set(split_defaults) == {
        "text_encoder.plan",
        "vision_encoder.plan",
        "adaln_precompute.plan",
        "denoiser_head.plan",
        "denoiser_tail.plan",
        "denoiser_finish.plan",
        "fl2va_keyframe_vae_encoder.plan",
        "vae_tile_decoder.plan",
        "audio_vae_decoder.plan",
    }


def test_dynamic_text_profile_preserves_the_537_token_maximum() -> None:
    profile = SOL_ENGINE_1344X768_124F

    assert (profile.min_text_rows, profile.opt_text_rows, profile.text_rows) == (1, 128, 537)
    assert (
        profile.min_sequence_length,
        profile.opt_sequence_length,
        profile.sequence_length,
    ) == (37711, 37838, 38247)
    assert profile.padded_sequence_length == profile.sequence_length
    profile.validate()


def test_dynamic_media_profile_preserves_124_and_345_frame_endpoints() -> None:
    profile = SOL_ENGINE_1344X768_124_TO_345F

    assert profile.video_row_profile == (21312, 37296, 108576)
    assert profile.audio_row_profile == (414, 414, 1150)
    assert profile.text_row_profile == (1, 128, 2641)
    assert profile.packed_row_profile == (21727, 37838, 112367)
    assert profile.padded_sequence_length == profile.sequence_length
    profile.validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_text_rows": 0},
        {"min_text_rows": 129, "opt_text_rows": 128},
        {"opt_text_rows": 538},
    ],
)
def test_dynamic_text_profile_rejects_invalid_bounds(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="1 <= min <= opt <= max"):
        MiniMaxH3Config(**overrides).validate()


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
        "denoiser_mode": "monolithic",
        "fast_h3": None,
    }
    _validate_resume_identity(dict(current), current)

    previous = dict(current)
    previous["workspace_limit_bytes"] = _workspace_limits(8)
    with pytest.raises(ValueError, match="different workspace_limit_bytes"):
        _validate_resume_identity(previous, current)


@pytest.mark.parametrize("denoiser_mode", ["monolithic", "first_block", "segmented_vsa"])
def test_main_passes_cli_workspace_to_every_builder(
    tmp_path: Path, monkeypatch, capsys, denoiser_mode: str
) -> None:
    from tests.e2e.models.minimax_h3 import build_native_components

    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    audio_vae_config = model / "audio_vae" / "config.json"
    audio_vae_config.parent.mkdir(parents=True)
    audio_vae_config.write_text("{}")
    output = tmp_path / "plans"
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    observed = {}
    module_specs = {
        "multimodal_text_encoder_builder": (
            ("build_multimodal_text_encoder_engine", "text_encoder.plan"),
        ),
        "multimodal_vision_builder": (
            ("build_multimodal_vision_encoder_engine", "vision_encoder.plan"),
        ),
        "adaln_builder": (("build_adaln_precompute_engine", "adaln_precompute.plan"),),
        "dit_builder": (
            ("build_dit_engine", "denoiser.plan"),
            ("build_dit_head_engine", "denoiser_head.plan"),
            ("build_dit_tail_engine", "denoiser_tail.plan"),
            ("build_dit_finish_engine", "denoiser_finish.plan"),
            ("build_dit_vsa_entry_engine", "denoiser_entry.plan"),
            ("build_dit_vsa_transition_engine", "denoiser_transition.plan"),
            ("build_dit_vsa_finish_engine", "denoiser_finish.plan"),
        ),
        "fl2va_vae_encoder_builder": (
            ("build_keyframe_vae_encoder_engine", "fl2va_keyframe_vae_encoder.plan"),
        ),
        "vae_builder": (("build_vae_tile_decoder_engine", "vae_tile_decoder.plan"),),
        "audio_vae_builder": (
            ("build_audio_vae_decoder_engine", "audio_vae_decoder.plan"),
        ),
    }
    package = "tensorrt_model_connect.families.minimax_h3"
    for module_name, builders in module_specs.items():
        fake_module = ModuleType(f"{package}.{module_name}")
        fake_module.checkpoint_keys = lambda *_args: ()
        if module_name == "dit_builder":
            fake_module.head_checkpoint_keys = lambda *_args: ()
            fake_module.tail_checkpoint_keys = lambda *_args: ()
            fake_module.finish_checkpoint_keys = lambda *_args: ()
            fake_module.vsa_entry_checkpoint_keys = lambda *_args: ()
            fake_module.vsa_transition_checkpoint_keys = lambda *_args: ()
            fake_module.vsa_finish_checkpoint_keys = lambda *_args: ()
        if module_name == "audio_vae_builder":
            fake_module.decoder_config_from_checkpoint = lambda *_args, **_kwargs: object()
        for builder_name, plan_name in builders:

            def fake_builder(*_args, _plan_name=plan_name, workspace_bytes=None, **_kwargs):
                resolved_name = _plan_name
                if _plan_name == "denoiser_transition.plan":
                    resolved_name = f"denoiser_transition_{_args[2]:02d}.plan"
                observed[resolved_name] = workspace_bytes
                return resolved_name.encode()

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
    adapter_partition_names = {
        "adaln_precompute",
        "denoiser_entry",
        *(f"denoiser_transition_{index:02d}" for index in range(49)),
        "denoiser_finish",
    }
    monkeypatch.setattr(
        build_native_components,
        "_adapter_target_partitions",
        lambda _profile: {name: (f"{name}.weight",) for name in adapter_partition_names},
    )
    adapter_metadata = {
        "schema_version": 1,
        "adapter_gate_tensor_count": 50,
    }
    identity = SimpleNamespace(
        partition_tensor_counts={name: 1 for name in adapter_partition_names},
        bundle_metadata=lambda: adapter_metadata,
    )
    monkeypatch.setattr(
        build_native_components,
        "validate_fast_h3_adapter",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        build_native_components,
        "merge_fast_h3_adapter_state",
        lambda *_args, **_kwargs: {"tensors": 1},
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
    if denoiser_mode == "first_block":
        argv.append("--first-block-cache")
    elif denoiser_mode == "segmented_vsa":
        argv.extend(("--fast-h3-adapter", str(adapter)))
    monkeypatch.setattr(sys, "argv", argv)

    assert build_native_components.main() == 0
    first_block_cache = denoiser_mode == "first_block"
    segmented_vsa = denoiser_mode == "segmented_vsa"
    expected_limits = {
        filename: 8 << 30
        for filename in default_workspace_limit_bytes(
            first_block_cache=first_block_cache,
            segmented_vsa=segmented_vsa,
        )
    }
    assert observed == expected_limits
    receipt = json.loads((output / "build_receipt.json").read_text())
    assert receipt["workspace_limit_bytes"] == observed
    assert set(receipt["components"]) == set(expected_limits)
    assert receipt["profile"]["first_block_cache"] is first_block_cache
    assert receipt["denoiser_mode"] == denoiser_mode
    assert (receipt["fast_h3"] is not None) is segmented_vsa
    assert receipt["profile"]["video_rows"] == 108576
    assert receipt["profile"]["audio_rows"] == 1150
    assert receipt["profile"]["padded_sequence_length"] == 112367
    capsys.readouterr()
