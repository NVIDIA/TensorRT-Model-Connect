"""Qwen-VL-owned diffusion VLM assessment defaults."""

from __future__ import annotations

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("diffusion_vlm_assessment.json")


def test_qwen_vl_diffusion_vlm_assessment_config_matches_current_ci_default() -> None:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert data["default"] is True
    assert data["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert data["max_side"] == 512
    assert data["max_new_tokens"] == 384
    assert data["timeout"] == "45m"
