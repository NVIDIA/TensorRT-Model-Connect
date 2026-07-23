# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tools/count_diffusion_frame_pairs.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.count_diffusion_frame_pairs import (
    count_diffusion_frame_pairs,
    discover_diffusion_frame_pairs,
    main,
    validate_complete_diffusion_frame_pairs,
)


def _write_result(
    model_dir,
    task_strategy: str = "diffusion_media_generation",
    case_name: str = "model-diff",
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "result.json").write_text(
        json.dumps({
            "case_name": case_name,
            "case_config": {
                "name": case_name,
                "task_strategy": task_strategy,
                "inputs": {"prompt": "a cat"},
            },
        }),
        encoding="utf-8",
    )


def _write_frame_pair(model_dir, reference_dir: str) -> None:
    (model_dir / "frames").mkdir(parents=True, exist_ok=True)
    (model_dir / reference_dir).mkdir(parents=True, exist_ok=True)
    (model_dir / "frames" / "frame_0000.png").write_bytes(b"trt")
    (model_dir / reference_dir / "frame_0000.png").write_bytes(b"ref")


def test_counts_frames_plus_hf_frames(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "video-diffusion"
    _write_result(model_dir, case_name="video-diffusion")
    _write_frame_pair(model_dir, "hf_frames")

    pairs = discover_diffusion_frame_pairs(tmp_path / "artifacts")

    assert count_diffusion_frame_pairs(tmp_path / "artifacts") == 1
    assert pairs[0]["case_name"] == "video-diffusion"
    assert pairs[0]["trt_image"].endswith("frames/frame_0000.png")
    assert pairs[0]["hf_image"].endswith("hf_frames/frame_0000.png")
    assert "frame_indices" not in pairs[0]
    assert "trt_images" not in pairs[0]
    assert "hf_images" not in pairs[0]


def test_counts_frames_plus_ref_frames(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "image-diffusion"
    _write_result(model_dir, case_name="image-diffusion")
    _write_frame_pair(model_dir, "ref_frames")

    pairs = discover_diffusion_frame_pairs(tmp_path / "artifacts")

    assert count_diffusion_frame_pairs(tmp_path / "artifacts") == 1
    assert pairs[0]["hf_image"].endswith("ref_frames/frame_0000.png")


def test_malformed_result_json_is_ignored(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "broken"
    model_dir.mkdir(parents=True)
    _write_frame_pair(model_dir, "hf_frames")
    (model_dir / "result.json").write_text("{not-json", encoding="utf-8")

    assert discover_diffusion_frame_pairs(tmp_path / "artifacts") == []
    assert count_diffusion_frame_pairs(tmp_path / "artifacts") == 0


def test_non_diffusion_result_is_ignored(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "example-decoder"
    _write_result(
        model_dir,
        task_strategy="text_generation_causal",
        case_name="example-decoder",
    )
    _write_frame_pair(model_dir, "hf_frames")

    assert discover_diffusion_frame_pairs(tmp_path / "artifacts") == []
    assert count_diffusion_frame_pairs(tmp_path / "artifacts") == 0


def test_zero_pair_count_and_json_output(tmp_path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts"
    model_dir = artifacts_dir / "missing-reference"
    _write_result(model_dir)
    (model_dir / "frames").mkdir(parents=True)
    (model_dir / "frames" / "frame_0000.png").write_bytes(b"trt")

    assert main([str(artifacts_dir)]) == 0
    assert capsys.readouterr().out.strip() == "0"

    assert main([str(artifacts_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"count": 0, "pairs": []}


def test_complete_mode_requires_every_diffusion_result_to_have_a_pair(tmp_path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    complete = artifacts_dir / "complete"
    incomplete = artifacts_dir / "incomplete"
    _write_result(complete, case_name="complete")
    _write_frame_pair(complete, "hf_frames")
    _write_result(incomplete, case_name="incomplete")
    (incomplete / "frames").mkdir()
    (incomplete / "frames" / "frame_0000.png").write_bytes(b"trt")

    with pytest.raises(ValueError, match=r"missing=\['incomplete'\]"):
        validate_complete_diffusion_frame_pairs(artifacts_dir)

    with pytest.raises(SystemExit) as error:
        main([str(artifacts_dir), "--require-complete"])
    assert error.value.code == 2


def test_complete_mode_returns_the_exact_unique_case_inventory(tmp_path, capsys) -> None:
    artifacts_dir = tmp_path / "artifacts"
    for case_name in ("alpha", "beta"):
        model_dir = artifacts_dir / case_name
        _write_result(model_dir, case_name=case_name)
        _write_frame_pair(model_dir, "ref_frames")

    pairs = validate_complete_diffusion_frame_pairs(artifacts_dir)

    assert [pair["case_name"] for pair in pairs] == ["alpha", "beta"]
    assert main([str(artifacts_dir), "--require-complete"]) == 0
    assert capsys.readouterr().out.strip() == "2"


def test_native_visual_policy_selects_six_evenly_spaced_video_frames(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "wan22-ti2v-5b"
    _write_result(model_dir, case_name="wan22-ti2v-5b")
    result_path = model_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["case_config"]["inputs"]["video_num_frames"] = 121
    result["case_config"]["metadata"] = {"native_acceptance": {"vlm_frame_samples": 6}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    for dirname in ("frames", "hf_frames"):
        frame_dir = model_dir / dirname
        frame_dir.mkdir()
        for index in range(121):
            (frame_dir / f"frame_{index:04d}.png").write_bytes(b"frame")

    pair = discover_diffusion_frame_pairs(tmp_path / "artifacts")[0]

    assert pair["frame_indices"] == [0, 24, 48, 72, 96, 120]
    assert [Path(path).name for path in pair["trt_images"]] == [
        f"frame_{index:04d}.png" for index in pair["frame_indices"]
    ]
    assert [Path(path).name for path in pair["hf_images"]] == [
        f"frame_{index:04d}.png" for index in pair["frame_indices"]
    ]


@pytest.mark.parametrize("sample_count", [0, 7, True, "6"])
def test_native_visual_policy_rejects_invalid_sample_counts(tmp_path, sample_count) -> None:
    model_dir = tmp_path / "artifacts" / "invalid"
    _write_result(model_dir, case_name="invalid")
    result_path = model_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["case_config"]["inputs"]["video_num_frames"] = 1
    result["case_config"]["metadata"] = {"native_acceptance": {"vlm_frame_samples": sample_count}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _write_frame_pair(model_dir, "hf_frames")

    with pytest.raises(ValueError, match="vlm_frame_samples"):
        discover_diffusion_frame_pairs(tmp_path / "artifacts")


def test_native_visual_policy_rejects_missing_sample_count_and_truncated_frames(
    tmp_path,
) -> None:
    model_dir = tmp_path / "artifacts" / "truncated"
    _write_result(model_dir, case_name="truncated")
    result_path = model_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["case_config"]["inputs"]["video_num_frames"] = 121
    result["case_config"]["metadata"] = {"native_acceptance": {}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    _write_frame_pair(model_dir, "hf_frames")

    with pytest.raises(ValueError, match="vlm_frame_samples"):
        discover_diffusion_frame_pairs(tmp_path / "artifacts")

    result["case_config"]["metadata"]["native_acceptance"]["vlm_frame_samples"] = 6
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="complete contiguous"):
        discover_diffusion_frame_pairs(tmp_path / "artifacts")
