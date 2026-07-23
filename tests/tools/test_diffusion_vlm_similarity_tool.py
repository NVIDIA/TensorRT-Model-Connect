# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import tools.evaluate_diffusion_vlm_similarity as vlm_similarity
from tools.evaluate_diffusion_vlm_similarity import (
    _LoadingWeightsProgressFilter,
    _aggregate_sample_results,
    _apply_gate,
    _discover_pairs,
    _expand_pair_samples,
    _load_assessment_config,
    _normalize_judgment_consistency,
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


def test_vlm_similarity_gate_allows_photographic_silhouette_reference():
    judgment = _normalize_judgment_consistency({
        "trt_description": "A cat sitting on a windowsill at sunset.",
        "semantic_similarity_0_to_5": 4.8,
        "trt_prompt_alignment_0_to_5": 4.8,
        "trt_visual_quality_0_to_5": 4.7,
        "hf_prompt_alignment_0_to_5": 4.8,
        "hf_visual_quality_0_to_5": 4.9,
        "trt_relative_to_hf": "similar",
        "is_regression": False,
        "hf_description": (
            "A silhouette of a cat sitting on a windowsill at sunset, with a "
            "vibrant sky in the background."
        ),
    }, prompt="A photo of a cat sitting on a windowsill at sunset")
    result = _apply_gate(
        judgment,
        prompt="A photo of a cat sitting on a windowsill at sunset",
    )

    assert not result["failed"]
    assert judgment["hf_prompt_alignment_0_to_5"] == 4.8
    assert judgment["hf_visual_quality_0_to_5"] == 4.9
    assert judgment["trt_relative_to_hf"] == "similar"


@pytest.mark.parametrize("medium", [
    "stylized", "vector", "cartoon", "drawing", "illustration",
])
def test_vlm_similarity_gate_fails_non_photo_reference_for_photo_prompt(medium):
    judgment = _normalize_judgment_consistency({
        "trt_description": "A cat sitting on a windowsill at sunset.",
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 5.0,
        "hf_visual_quality_0_to_5": 5.0,
        "trt_relative_to_hf": "similar",
        "is_regression": False,
        "hf_description": (
            f"A {medium} silhouette of a cat sitting on a windowsill."
        ),
    }, prompt="A photo of a cat sitting on a windowsill at sunset")
    result = _apply_gate(
        judgment,
        prompt="A photo of a cat sitting on a windowsill at sunset",
    )

    assert result["failed"]
    assert judgment["trt_relative_to_hf"] == "better"
    assert judgment["trt_relative_to_hf_original"] == "similar"
    assert judgment["hf_prompt_alignment_0_to_5"] == 2.0
    assert judgment["hf_visual_quality_0_to_5"] == 2.0
    assert any("photo prompt" in reason for reason in result["reasons"])


def test_vlm_similarity_gate_passes_when_photo_reference_is_fixed():
    judgment = _normalize_judgment_consistency({
        "trt_description": "A cat sitting on a windowsill at sunset.",
        "hf_description": "A photo of a cat sitting on a windowsill at sunset.",
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "trt_relative_to_hf": "similar",
        "is_regression": False,
    }, prompt="A photo of a cat sitting on a windowsill at sunset")
    result = _apply_gate(
        judgment,
        prompt="A photo of a cat sitting on a windowsill at sunset",
    )

    assert not result["failed"]
    assert judgment["trt_relative_to_hf"] == "similar"


def test_parse_json_normalizes_internvl_quality_key_typo():
    parsed = _parse_json("""
```json
{"hf_visual_quality_5_to_5": 5, "semantic_similarity_0_to_5": 4}
```
""")

    assert parsed["hf_visual_quality_0_to_5"] == 5


def test_loading_weights_progress_filter_drops_only_progress_lines():
    class Sink:
        def __init__(self):
            self.parts = []

        def write(self, text):
            self.parts.append(text)

        def flush(self):
            pass

    sink = Sink()
    filtered = _LoadingWeightsProgressFilter(sink)
    filtered.write("before\n")
    filtered.write("Loading weights:   0%|          | 1/824 [00:00<?, ?it/s]\r")
    filtered.write("after\n")

    assert "".join(sink.parts) == "before\nafter\n"


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
    model_dir = tmp_path / "artifacts" / "image-diffusion"
    (model_dir / "frames").mkdir(parents=True)
    (model_dir / "ref_frames").mkdir()
    (model_dir / "frames" / "frame_0000.png").write_bytes(b"trt")
    (model_dir / "ref_frames" / "frame_0000.png").write_bytes(b"ref")
    (model_dir / "result.json").write_text(
        json.dumps({
            "case_name": "image-diffusion",
            "case_config": {
                "task_strategy": "diffusion_media_generation",
                "inputs": {"prompt": "a cat"},
            },
        }),
        encoding="utf-8",
    )

    pairs = _discover_pairs(tmp_path / "artifacts")

    assert len(pairs) == 1
    assert pairs[0]["case_name"] == "image-diffusion"
    assert pairs[0]["hf_image"].endswith("ref_frames/frame_0000.png")


def test_expand_pair_samples_preserves_frame_order():
    pair = {
        "case_name": "wan22",
        "prompt": "cats boxing",
        "trt_images": [f"trt-{index}.png" for index in (0, 24, 48, 72, 96, 120)],
        "hf_images": [f"hf-{index}.png" for index in (0, 24, 48, 72, 96, 120)],
        "frame_indices": [0, 24, 48, 72, 96, 120],
    }

    samples = _expand_pair_samples(pair)

    assert [sample["frame_index"] for sample in samples] == [0, 24, 48, 72, 96, 120]
    assert samples[3]["trt_image"] == "trt-72.png"
    assert samples[3]["hf_image"] == "hf-72.png"
    assert all(sample["case_name"] == "wan22" for sample in samples)


def test_single_frame_judgment_preserves_legacy_output_schema(monkeypatch):
    pair = {
        "case_name": "image-diffusion",
        "prompt": "a cat",
        "trt_image": "trt.png",
        "hf_image": "reference.png",
    }

    def judge(_model, _processor, _device, sample, **_kwargs):
        return {
            **sample,
            "vlm_judgment": {
                "reason": "same subject",
                "vlm_gate": {"failed": False, "reasons": []},
            },
        }

    monkeypatch.setattr(vlm_similarity, "_judge_pair", judge)

    result = vlm_similarity._judge_pair_samples(
        None,
        None,
        "cpu",
        pair,
        max_side=256,
        max_new_tokens=128,
    )

    assert result["trt_image"] == "trt.png"
    assert result["hf_image"] == "reference.png"
    assert result["vlm_judgment"]["reason"] == "same subject"
    assert "sampled_frame_indices" not in result
    assert "sample_assessments" not in result


def test_aggregate_sample_results_fails_if_any_sample_fails():
    passing = {
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": False,
        "reason": "coherent",
        "vlm_gate": {"failed": False, "reasons": []},
    }
    failing = {
        **passing,
        "semantic_similarity_0_to_5": 1.0,
        "is_regression": True,
        "reason": "corrupted frame",
        "vlm_gate": {
            "failed": True,
            "reasons": ["semantic_similarity_0_to_5=1.00 < 3.0"],
        },
    }
    samples = [
        {"frame_index": 0, "vlm_judgment": passing},
        {"frame_index": 24, "vlm_judgment": passing},
        {"frame_index": 48, "vlm_judgment": failing},
    ]

    aggregate = _aggregate_sample_results(
        {"case_name": "wan22", "prompt": "cats boxing"},
        samples,
    )

    assert aggregate["sampled_frame_indices"] == [0, 24, 48]
    assert aggregate["vlm_judgment"]["vlm_gate"]["failed"]
    assert aggregate["vlm_judgment"]["is_regression"] is True
    assert aggregate["vlm_judgment"]["semantic_similarity_0_to_5"] == 1.0
    assert "frame 48" in aggregate["vlm_judgment"]["vlm_gate"]["reasons"][0]


def test_vlm_assessment_config_loader_uses_explicit_config(tmp_path):
    config = tmp_path / "assessment.json"
    config.write_text(
        json.dumps({
            "model_id": "example/vlm-judge",
            "max_side": 256,
            "max_new_tokens": 128,
            "timeout": "5m",
        }),
        encoding="utf-8",
    )

    loaded = _load_assessment_config(config)

    assert loaded["model_id"] == "example/vlm-judge"
    assert loaded["max_side"] == 256
    assert loaded["max_new_tokens"] == 128
