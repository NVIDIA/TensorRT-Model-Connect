# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target binding and staged-loading checks for MiniMax-H3 bundles."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tensorrt_model_connect.families.minimax_h3.provenance import (
    FASTH3_SEGMENTED_PLAN_FILENAMES,
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
            "vision_encoder_plan",
            "adaln_precompute_plan",
            "denoiser_plan",
            "fl2va_keyframe_vae_encoder_plan",
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
    assert split["lazy_sections"][3:6] == [
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
    ]
    assert set(split["lazy_sections"]) == set(pack_native_bundle.FIRST_BLOCK_CACHE_PLAN_SECTIONS)
    fast_h3 = pack_native_bundle._bundle_loading_policy(pack_native_bundle.FASTH3_PLAN_SECTIONS)
    assert len(fast_h3["lazy_sections"]) == len(FASTH3_SEGMENTED_PLAN_FILENAMES)
    assert fast_h3["lazy_sections"][3] == "denoiser_entry_plan"
    assert fast_h3["lazy_sections"][53] == "denoiser_finish_plan"


@pytest.mark.parametrize("denoiser_mode", ["monolithic", "first_block", "segmented_vsa"])
def test_packer_preserves_validated_workspace_mapping(
    tmp_path: Path, monkeypatch, capsys, denoiser_mode: str
) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    recorded = {}
    first_block_cache = denoiser_mode == "first_block"
    fast_h3 = denoiser_mode == "segmented_vsa"
    selected_plans = (
        FASTH3_SEGMENTED_PLAN_FILENAMES
        if fast_h3
        else FIRST_BLOCK_CACHE_PLAN_FILENAMES
        if first_block_cache
        else PLAN_FILENAMES
    )
    for filename in selected_plans:
        (plans / filename).write_bytes(filename.encode())
        recorded[filename] = {"bytes": len(filename), "sha256": "a" * 64}
    workspace_limits = {filename: 8 << 30 for filename in selected_plans}
    partition_counts = {
        "adaln_precompute": 40,
        "denoiser_entry": 16,
        **{f"denoiser_transition_{index:02d}": 16 for index in range(49)},
        "denoiser_finish": 16,
    }
    fast_h3_metadata = (
        {
            "schema_version": 1,
            "adapter_model_id": pack_native_bundle.FASTH3_VSA_ADAPTER_MODEL_ID,
            "adapter_source_revision": pack_native_bundle.FASTH3_VSA_ADAPTER_SOURCE_REVISION,
            "adapter_sha256": pack_native_bundle.FASTH3_VSA_ADAPTER_SHA256,
            "adapter_bytes": pack_native_bundle.FASTH3_VSA_ADAPTER_BYTES,
            "adapter_tensor_count": 856,
            "adapter_low_rank_tensor_count": 724,
            "adapter_diff_tensor_count": 82,
            "adapter_set_weight_tensor_count": 50,
            "adapter_gate_tensor_count": 50,
            "adapter_partition_tensor_counts": partition_counts,
            "adapter_base_revision": pack_native_bundle.FASTH3_VSA_ADAPTER_BASE_REVISION,
            "adapter_finetuned_revision": (
                pack_native_bundle.FASTH3_VSA_ADAPTER_FINETUNED_REVISION
            ),
        }
        if fast_h3
        else None
    )
    (plans / "build_receipt.json").write_text(
        json.dumps(
            {
                "build_helper_sha256": "b" * 64,
                "workspace_limit_bytes": workspace_limits,
                "denoiser_mode": denoiser_mode,
                "fast_h3": fast_h3_metadata,
            }
        )
    )
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}")
    audio_vae_config = model / "audio_vae" / "config.json"
    audio_vae_config.parent.mkdir(parents=True)
    audio_vae_config.write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        )
    )
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
    elif fast_h3:
        argv.append("--fast-h3")
    monkeypatch.setattr(sys, "argv", argv)

    assert pack_native_bundle.main() == 0
    assert captured["workspace_limit_bytes"] == workspace_limits
    assert captured["first_block_cache"] is first_block_cache
    assert captured["denoiser_cache_mode"] == denoiser_mode
    assert captured["first_block_cache_threshold"] == 0.08
    assert (
        captured["text_rows_min"],
        captured["text_rows_opt"],
        captured["text_rows_max"],
    ) == (1, 128, 2641)
    assert (
        captured["num_frames_min"],
        captured["num_frames_opt"],
        captured["num_frames_max"],
    ) == (124, 124, 345)
    assert (
        captured["video_rows_min"],
        captured["video_rows_opt"],
        captured["video_rows_max"],
    ) == (18870, 37296, 108576)
    assert (
        captured["audio_rows_min"],
        captured["audio_rows_opt"],
        captured["audio_rows_max"],
    ) == (414, 414, 1150)
    assert (
        captured["packed_sequence_length_min"],
        captured["packed_sequence_length_opt"],
        captured["packed_sequence_length_max"],
    ) == (19285, 37838, 112367)
    assert captured["audio_latent_frames"] == 207
    assert captured["audio_sample_rate"] == 32000
    assert captured["audio_hop_length"] == 800
    assert captured["audio_channels"] == 2
    assert captured["audio_vae_precision"] == "fp32"
    assert captured["vae_tile_batch_min"] == 15
    assert captured["explicit_canvas_sizes"] == [[544, 960], [960, 544]]
    assert captured["vae_tile_batch_opt"] == 28
    assert captured["vae_tile_batch_max"] == 33
    assert captured["attention_mode"] == ("native_vsa" if fast_h3 else "dense")
    if fast_h3:
        assert captured["num_inference_steps"] == 4
        assert captured["scheduler_grid_points"] == 5
        assert captured["transformer_forwards"] == 4
        assert captured["vsa"]["implementation"] == "native_cuda_segmented"
        assert captured["vsa"]["segment_count"] == 51
        assert captured["vsa"]["max_total_tiles"] == 2140
        assert captured["vsa"]["prefix_valid_sizes_profile"] == [8, 9, 60]
        assert captured["vsa"]["video_valid_sizes_profile"] == [360, 660, 2080]
        assert captured["vsa"]["runtime_metadata_abi"]["packed_row_to_tile_slot"] == {
            "dtype": "int32",
            "shape": ["S"],
            "profile": [19285, 37838, 112367],
        }
    capsys.readouterr()
