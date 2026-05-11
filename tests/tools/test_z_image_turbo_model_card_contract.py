"""Tests for the full Z-Image Turbo model-card E2E contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests import test_e2e
from tools.evaluate_diffusion_vlm_similarity import _apply_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "tests/e2e/models/z-image-turbo.json"
MODEL_CARD_PROMPT = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"


def test_z_image_turbo_uses_model_card_prompt_and_generation_settings() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert raw["hf_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert raw["reference_family"] == "diffusers_image_gen"
    assert raw["user_contract"] == "diffusion_image"
    assert raw["test_prompt"] == MODEL_CARD_PROMPT
    assert raw["image_height"] == 1024
    assert raw["image_width"] == 1024
    assert raw["num_inference_steps"] == 9
    assert raw["guidance_scale"] == 0.0
    assert raw["seed"] == 42


def test_z_image_turbo_prompt_does_not_trigger_photo_reference_waive_path() -> None:
    gate = _apply_gate({
        "semantic_similarity_0_to_5": 4.0,
        "trt_prompt_alignment_0_to_5": 4.0,
        "trt_visual_quality_0_to_5": 4.0,
        "hf_prompt_alignment_0_to_5": 4.0,
        "hf_visual_quality_0_to_5": 4.0,
        "is_regression": False,
        "hf_description": "A stylized astronaut in a lush jungle.",
    }, prompt=MODEL_CARD_PROMPT)

    assert not gate["failed"]


def test_z_image_turbo_is_not_globally_waived() -> None:
    waives = test_e2e._load_waives()

    assert "z-image-turbo" not in waives
