# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target binding and staged-loading checks for MiniMax-H3 bundles."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tensorrt_model_connect.families.minimax_h3.config import (
    FL2VA_PLAN_FILENAMES,
    FL2VA_PROCESSOR_ASSET_SECTIONS,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_PLAN_FILENAMES,
)
from tensorrt_model_connect.families.minimax_h3.provenance import (
    FIRST_BLOCK_CACHE_PLAN_FILENAMES,
    PLAN_FILENAMES,
)
from tests.e2e.models.minimax_h3 import pack_native_bundle


def test_target_metadata_comes_from_the_build_machine(monkeypatch) -> None:
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_trt_version",
        lambda: "11.1.0.106",
    )
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_gpu_name",
        lambda: "NVIDIA Thor",
    )

    assert pack_native_bundle._target_metadata() == (
        "11.1.0.106",
        "11.1",
        "NVIDIA Thor",
    )


def test_target_metadata_fails_closed_when_detection_is_incomplete(monkeypatch) -> None:
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_trt_version",
        lambda: "unknown",
    )
    monkeypatch.setattr(
        pack_native_bundle.engine_builder,
        "_get_gpu_name",
        lambda: "",
    )

    try:
        pack_native_bundle._target_metadata()
    except RuntimeError as error:
        assert "detected TensorRT version and GPU" in str(error)
    else:
        raise AssertionError("incomplete target metadata was accepted")


def test_staged_loading_partitions_every_bundle_section() -> None:
    policy = pack_native_bundle._bundle_loading_policy()

    assert policy == {
        "mode": "staged",
        "eager_sections": ["tokenizer.json", "config.json"],
        "lazy_sections": [
            "text_encoder_plan",
            "adaln_precompute_plan",
            "denoiser_plan",
            "vae_tile_decoder_plan",
            "audio_vae_decoder_plan",
        ],
    }
    assert set(policy["eager_sections"]) | set(policy["lazy_sections"]) == {
        *pack_native_bundle.PLAN_SECTIONS,
        "tokenizer.json",
        "config.json",
    }
    assert not set(policy["eager_sections"]) & set(policy["lazy_sections"])
    split = pack_native_bundle._bundle_loading_policy(
        pack_native_bundle.FIRST_BLOCK_CACHE_PLAN_SECTIONS
    )
    assert split["lazy_sections"][2:5] == [
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
    ]
    assert set(split["lazy_sections"]) == set(pack_native_bundle.FIRST_BLOCK_CACHE_PLAN_SECTIONS)
    fl2va = pack_native_bundle._bundle_loading_policy(
        pack_native_bundle.FL2VA_PLAN_SECTIONS,
        processor_sections=FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    assert fl2va == {
        "mode": "staged",
        "eager_sections": [
            "tokenizer.json",
            *FL2VA_PROCESSOR_ASSET_SECTIONS,
            "config.json",
        ],
        "lazy_sections": [
            f"{filename.removesuffix('.plan')}_plan" for filename in FL2VA_PLAN_FILENAMES
        ],
    }
    assert not set(fl2va["eager_sections"]) & set(fl2va["lazy_sections"])
    ref2va = pack_native_bundle._bundle_loading_policy(
        pack_native_bundle.REF2VA_PLAN_SECTIONS,
        processor_sections=FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    assert ref2va == {
        "mode": "staged",
        "eager_sections": [
            "tokenizer.json",
            *FL2VA_PROCESSOR_ASSET_SECTIONS,
            "config.json",
        ],
        "lazy_sections": [
            f"{filename.removesuffix('.plan')}_plan" for filename in REF2VA_PLAN_FILENAMES
        ],
    }
    assert not set(ref2va["eager_sections"]) & set(ref2va["lazy_sections"])


@pytest.mark.parametrize("first_block_cache", [False, True])
def test_packer_preserves_validated_workspace_mapping(
    tmp_path: Path, monkeypatch, capsys, first_block_cache: bool
) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    recorded = {}
    selected_plans = FIRST_BLOCK_CACHE_PLAN_FILENAMES if first_block_cache else PLAN_FILENAMES
    for filename in selected_plans:
        (plans / filename).write_bytes(filename.encode())
        recorded[filename] = {"bytes": len(filename), "sha256": "a" * 64}
    workspace_limits = {filename: 8 << 30 for filename in selected_plans}
    (plans / "build_receipt.json").write_text(
        json.dumps(
            {
                "build_helper_sha256": "b" * 64,
                "workspace_limit_bytes": workspace_limits,
            }
        )
    )
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    captured = {}

    monkeypatch.setattr(pack_native_bundle, "_target_metadata", lambda: ("11.1", "11.1", "Thor"))
    monkeypatch.setattr(
        pack_native_bundle,
        "validate_build_receipt",
        lambda *_args, **_kwargs: (
            "c" * 64,
            recorded,
            {"sha256": "d" * 64},
            {"inventory_sha256": "e" * 64},
        ),
    )

    def capture_bundle(_output, _info, sections) -> None:
        config_section = next(section for section in sections if section.name == "config.json")
        captured.update(json.loads(config_section.data))

    monkeypatch.setattr(pack_native_bundle, "write_bundle", capture_bundle)
    argv = [
        "pack_native_bundle.py",
        "--plans-dir",
        str(plans),
        "--model-path",
        str(model),
        "--output",
        str(tmp_path / "model.bundle"),
        "--source-revision",
        "1" * 40,
    ]
    if first_block_cache:
        argv.append("--first-block-cache")
    monkeypatch.setattr(sys, "argv", argv)

    assert pack_native_bundle.main() == 0
    assert captured["workspace_limit_bytes"] == workspace_limits
    assert captured["first_block_cache"] is first_block_cache
    assert captured["denoiser_cache_mode"] == ("first_block" if first_block_cache else "monolithic")
    assert captured["first_block_cache_threshold"] == 0.025
    assert captured["audio_sample_rate"] == 32000
    assert captured["audio_latent_frames"] == 207
    assert captured["audio_output_samples"] == 165600
    capsys.readouterr()


def test_fl2va_packer_binds_seven_plans_processor_assets_and_workflow(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    recorded = {}
    for index, filename in enumerate(FL2VA_PLAN_FILENAMES, start=1):
        payload = filename.encode()
        (plans / filename).write_bytes(payload)
        recorded[filename] = {"bytes": len(payload), "sha256": f"{index:064x}"}
    workspace_limits = {filename: 8 << 30 for filename in FL2VA_PLAN_FILENAMES}
    model = tmp_path / "model"
    asset_records = {}
    for index, relative in enumerate(
        ("tokenizer/tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS),
        start=20,
    ):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in FL2VA_PROCESSOR_ASSET_SECTIONS:
            blob = tmp_path / f"processor-asset-{index}.json"
            blob.write_text("{}")
            path.symlink_to(blob)
        else:
            path.write_text("{}")
        receipt_name = "tokenizer.json" if relative == "tokenizer/tokenizer.json" else relative
        asset_records[receipt_name] = {"bytes": 2, "sha256": f"{index:064x}"}
    (plans / "build_receipt.json").write_text(
        json.dumps(
            {
                "workflow": "fl2va",
                "checkpoint_partition": "transformer",
                "build_helper_sha256": "b" * 64,
                "workspace_limit_bytes": workspace_limits,
                "assets": asset_records,
            }
        )
    )
    validation = {}

    def validate(*_args, **kwargs):
        validation.update(kwargs)
        return (
            "c" * 64,
            recorded,
            asset_records["tokenizer.json"],
            {"inventory_sha256": "e" * 64},
        )

    monkeypatch.setattr(pack_native_bundle, "_target_metadata", lambda: ("11.1", "11.1", "Thor"))
    monkeypatch.setattr(pack_native_bundle, "validate_build_receipt", validate)
    captured = {}

    def capture_bundle(_output, _info, sections) -> None:
        captured["section_names"] = [section.name for section in sections]
        captured["section_sources"] = {
            section.name: getattr(section, "source_path", None) for section in sections
        }
        config_section = next(section for section in sections if section.name == "config.json")
        captured["config"] = json.loads(config_section.data)

    monkeypatch.setattr(pack_native_bundle, "write_bundle", capture_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_native_bundle.py",
            "--plans-dir",
            str(plans),
            "--model-path",
            str(model),
            "--output",
            str(tmp_path / "fl2va.bundle"),
            "--source-revision",
            "1" * 40,
            "--workflow",
            "fl2va",
        ],
    )

    assert pack_native_bundle.main() == 0
    assert validation["workflow"] == "fl2va"
    assert captured["section_names"] == [
        *pack_native_bundle.FL2VA_PLAN_SECTIONS,
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
        "config.json",
    ]
    for section_name in FL2VA_PROCESSOR_ASSET_SECTIONS:
        assert captured["section_sources"][section_name] == (model / section_name).resolve()
        assert not captured["section_sources"][section_name].is_symlink()
    config = captured["config"]
    assert config["workflow"] == "fl2va"
    assert config["checkpoint_partition"] == "transformer"
    assert config["first_block_cache"] is False
    assert config["denoiser_cache_mode"] == "monolithic"
    assert tuple(config["plan_sha256"]) == FL2VA_PLAN_FILENAMES
    assert tuple(config["asset_sha256"]) == (
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    assert config["processor_asset_sections"] == list(FL2VA_PROCESSOR_ASSET_SECTIONS)
    assert config["fl2va_keyframe_counts"] == [0, 1, 2]
    assert config["fl2va_keyframe_rows"] == 1008
    assert config["fl2va_vae_tile_size"] == 256
    assert config["fl2va_vae_tile_min_overlap"] == 64
    assert config["fl2va_vae_temporal_frames"] == [1]
    assert config["bundle_loading"] == pack_native_bundle._bundle_loading_policy(
        pack_native_bundle.FL2VA_PLAN_SECTIONS,
        processor_sections=FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    capsys.readouterr()


def test_ref2va_packer_binds_nine_plans_processor_assets_and_partition(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    recorded = {}
    for index, filename in enumerate(REF2VA_PLAN_FILENAMES, start=1):
        payload = filename.encode()
        (plans / filename).write_bytes(payload)
        recorded[filename] = {"bytes": len(payload), "sha256": f"{index:064x}"}
    workspace_limits = {filename: 8 << 30 for filename in REF2VA_PLAN_FILENAMES}
    model = tmp_path / "model"
    asset_records = {}
    for index, relative in enumerate(
        ("tokenizer/tokenizer.json", *FL2VA_PROCESSOR_ASSET_SECTIONS),
        start=20,
    ):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in FL2VA_PROCESSOR_ASSET_SECTIONS:
            blob = tmp_path / f"processor-asset-{index}.json"
            blob.write_text("{}")
            path.symlink_to(blob)
        else:
            path.write_text("{}")
        receipt_name = "tokenizer.json" if relative == "tokenizer/tokenizer.json" else relative
        asset_records[receipt_name] = {"bytes": 2, "sha256": f"{index:064x}"}
    (plans / "build_receipt.json").write_text(
        json.dumps(
            {
                "workflow": "ref2va",
                "checkpoint_partition": "transformer_ref",
                "build_helper_sha256": "b" * 64,
                "workspace_limit_bytes": workspace_limits,
                "assets": asset_records,
            }
        )
    )
    validation = {}

    def validate(*_args, **kwargs):
        validation.update(kwargs)
        return (
            "c" * 64,
            recorded,
            asset_records["tokenizer.json"],
            {"inventory_sha256": "e" * 64},
        )

    monkeypatch.setattr(pack_native_bundle, "_target_metadata", lambda: ("11.1", "11.1", "Thor"))
    monkeypatch.setattr(pack_native_bundle, "validate_build_receipt", validate)
    captured = {}

    def capture_bundle(_output, _info, sections) -> None:
        captured["section_names"] = [section.name for section in sections]
        captured["section_sources"] = {
            section.name: getattr(section, "source_path", None) for section in sections
        }
        config_section = next(section for section in sections if section.name == "config.json")
        captured["config"] = json.loads(config_section.data)

    monkeypatch.setattr(pack_native_bundle, "write_bundle", capture_bundle)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pack_native_bundle.py",
            "--plans-dir",
            str(plans),
            "--model-path",
            str(model),
            "--output",
            str(tmp_path / "ref2va.bundle"),
            "--source-revision",
            "1" * 40,
            "--workflow",
            "ref2va",
        ],
    )

    assert pack_native_bundle.main() == 0
    assert validation["workflow"] == "ref2va"
    assert captured["section_names"] == [
        *pack_native_bundle.REF2VA_PLAN_SECTIONS,
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
        "config.json",
    ]
    for section_name in FL2VA_PROCESSOR_ASSET_SECTIONS:
        assert captured["section_sources"][section_name] == (model / section_name).resolve()
        assert not captured["section_sources"][section_name].is_symlink()
    config = captured["config"]
    assert config["workflow"] == "ref2va"
    assert config["checkpoint_partition"] == "transformer_ref"
    assert config["first_block_cache"] is False
    assert config["denoiser_cache_mode"] == "monolithic"
    assert tuple(config["plan_sha256"]) == REF2VA_PLAN_FILENAMES
    assert tuple(config["asset_sha256"]) == (
        "tokenizer.json",
        *FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    assert config["processor_asset_sections"] == list(FL2VA_PROCESSOR_ASSET_SECTIONS)
    assert config["min_text_rows"] == 1
    assert config["opt_text_rows"] == 8192
    assert config["max_text_rows"] == REF2VA_MAX_TEXT_ROWS
    assert config["ref2va_max_condition_video_rows"] == REF2VA_MAX_CONDITION_VIDEO_ROWS
    assert config["ref2va_max_condition_audio_rows"] == REF2VA_MAX_CONDITION_AUDIO_ROWS
    assert config["ref2va_max_images"] == 9
    assert config["ref2va_max_videos"] == 3
    assert config["ref2va_max_audios"] == 3
    assert config["ref2va_max_references"] == 12
    assert config["ref2va_reference_min_seconds"] == 2
    assert config["ref2va_reference_max_seconds"] == 15
    assert config["ref2va_vae_tile_size"] == 256
    assert config["ref2va_vae_tile_min_overlap"] == 64
    assert config["ref2va_vae_temporal_frames"] == [1, 17]
    assert config["bundle_loading"] == pack_native_bundle._bundle_loading_policy(
        pack_native_bundle.REF2VA_PLAN_SECTIONS,
        processor_sections=FL2VA_PROCESSOR_ASSET_SECTIONS,
    )
    capsys.readouterr()
