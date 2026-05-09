from __future__ import annotations

import json

from tools.evaluate_diffusion_vlm_similarity import (
    _apply_gate,
    _discover_pairs,
    _gate_failure_is_reference_only,
    _load_xfail_waives,
    _parse_json,
)


def test_vlm_similarity_gate_fails_low_semantic_similarity():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 0.1,
        "trt_prompt_alignment_0_to_5": 0.0,
        "trt_visual_quality_0_to_5": 0.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": False,
    })

    assert result["failed"]
    assert any("semantic_similarity" in reason for reason in result["reasons"])


def test_vlm_similarity_gate_allows_minor_quality_delta():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": "false",
    })

    assert not result["failed"]


def test_vlm_similarity_gate_fails_invalid_reference_quality():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 2.0,
        "hf_visual_quality_0_to_5": 2.0,
        "is_regression": False,
    })

    assert result["failed"]
    assert any("hf_prompt_alignment" in reason for reason in result["reasons"])


def test_vlm_similarity_gate_fails_explicit_regression():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": True,
    })

    assert result["failed"]
    assert "is_regression is true" in result["reasons"]


def test_vlm_similarity_gate_fails_silhouette_reference_for_photo_prompt():
    result = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": False,
        "hf_description": "A silhouette of a cat sitting on a windowsill.",
    }, prompt="A photo of a cat sitting on a windowsill at sunset")

    assert result["failed"]
    assert any("photo prompt" in reason for reason in result["reasons"])


def test_vlm_reference_only_gate_failure_can_be_waived(tmp_path):
    waives = tmp_path / "waives.txt"
    waives.write_text(
        "z-image-turbo-l0 XFAIL (bad HF reference)\n"
        "other-model SKIP (not an xfail)\n",
        encoding="utf-8",
    )
    gate = {
        "failed": True,
        "reasons": [
            "HF reference description suggests non-photo/stylized output for a photo prompt",
        ],
    }

    assert "z-image-turbo-l0" in _load_xfail_waives(waives)
    assert "other-model" not in _load_xfail_waives(waives)
    assert _gate_failure_is_reference_only(gate)


def test_vlm_trt_gate_failure_is_not_reference_only():
    gate = {
        "failed": True,
        "reasons": [
            "trt_visual_quality_0_to_5=1.00 < 2.5",
            "HF reference description suggests non-photo/stylized output for a photo prompt",
        ],
    }

    assert not _gate_failure_is_reference_only(gate)


def test_parse_json_normalizes_internvl_quality_key_typo():
    parsed = _parse_json("""
```json
{"hf_visual_quality_5_to_5": 5, "semantic_similarity_0_to_5": 4}
```
""")

    assert parsed["hf_visual_quality_0_to_5"] == 5


def test_parse_json_recovers_scores_from_truncated_vlm_json():
    parsed = _parse_json("""
```json
{
  "semantic_similarity_0_to_5": 4.5,
  "trt_prompt_alignment_0_to_5": 4,
  "trt_visual_quality_0_to_5": 4,
  "is_regression": false,
  "reason": "repeated text
""")

    assert parsed["semantic_similarity_0_to_5"] == 4.5
    assert parsed["trt_prompt_alignment_0_to_5"] == 4
    assert parsed["trt_visual_quality_0_to_5"] == 4
    assert parsed["is_regression"] is False


def test_discover_pairs_accepts_ref_frames_alias(tmp_path):
    model_dir = tmp_path / "artifacts" / "pixart"
    (model_dir / "frames").mkdir(parents=True)
    (model_dir / "ref_frames").mkdir()
    (model_dir / "frames" / "frame_0000.png").write_bytes(b"trt")
    (model_dir / "ref_frames" / "frame_0000.png").write_bytes(b"ref")
    (model_dir / "result.json").write_text(
        json.dumps({
            "case_name": "pixart",
            "case_config": {
                "task_strategy": "diffusion_media_generation",
                "inputs": {"prompt": "a cat"},
            },
        }),
        encoding="utf-8",
    )

    pairs = _discover_pairs(tmp_path / "artifacts")

    assert len(pairs) == 1
    assert pairs[0]["case_name"] == "pixart"
    assert pairs[0]["hf_image"].endswith("ref_frames/frame_0000.png")
