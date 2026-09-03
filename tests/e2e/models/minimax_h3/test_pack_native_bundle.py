# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target binding and staged-loading checks for MiniMax-H3 bundles."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

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

    def capture_bundle(_output, info, sections) -> None:
        config_section = next(section for section in sections if section.name == "config.json")
        captured.update(json.loads(config_section.data))
        captured["bundle_max_cache_length"] = info.max_cache_length

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
    manifest = json.loads(
        (Path(pack_native_bundle.__file__).parent / "manifests" / "minimax-h3-768p.json").read_text()
    )
    assert captured["bundle_max_cache_length"] == manifest["max_cache_length"] == 32
    assert (
        captured["text_rows_min"],
        captured["text_rows_opt"],
        captured["text_rows_max"],
    ) == (1, 128, 537)
    capsys.readouterr()
