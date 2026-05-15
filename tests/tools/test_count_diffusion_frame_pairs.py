"""Tests for tools/count_diffusion_frame_pairs.py."""

from __future__ import annotations

import json

from tools.count_diffusion_frame_pairs import (
    count_diffusion_frame_pairs,
    discover_diffusion_frame_pairs,
    main,
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
    model_dir = tmp_path / "artifacts" / "wan"
    _write_result(model_dir, case_name="wan")
    _write_frame_pair(model_dir, "hf_frames")

    pairs = discover_diffusion_frame_pairs(tmp_path / "artifacts")

    assert count_diffusion_frame_pairs(tmp_path / "artifacts") == 1
    assert pairs[0]["case_name"] == "wan"
    assert pairs[0]["trt_image"].endswith("frames/frame_0000.png")
    assert pairs[0]["hf_image"].endswith("hf_frames/frame_0000.png")


def test_counts_frames_plus_ref_frames(tmp_path) -> None:
    model_dir = tmp_path / "artifacts" / "pixart"
    _write_result(model_dir, case_name="pixart")
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
    model_dir = tmp_path / "artifacts" / "qwen"
    _write_result(model_dir, task_strategy="text_generation_causal", case_name="qwen")
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
