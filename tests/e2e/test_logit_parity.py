"""E2E: diff_logits passes within the model's configured atol."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]

_SPEECH_RUNTIME_STRATEGIES = {
    "speech_to_text",
    "speech_to_text_rnnt",
    "speech_to_speech",
}


@pytest.mark.e2e
def test_logit_parity(model_entry):
    """Run diff_logits battery and verify it passes."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — use test_diffusion_pipeline for logit checks")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — use test_audio_pipeline")
    if (
        model_entry.get("test_type") == "transcription"
        or model_entry.get("runtime_strategy") in _SPEECH_RUNTIME_STRATEGIES
    ):
        pytest.skip("Speech model — no text logit parity")
    if str(model_entry.get("runtime_strategy") or "").endswith("_vision_language"):
        pytest.skip("VL model — diff_logits requires decoder-only models")
    if model_entry.get("skip_logit_parity"):
        pytest.skip("Model requires HF auth — skipping logit parity")
    hf_id = model_entry["hf_id"]
    atol = model_entry.get("logit_atol", 1e-3)
    max_cache = model_entry.get("max_cache_length", 64)

    diff_logits = PROJECT_DIR / "tools" / "diff_logits.py"
    result = subprocess.run(
        [sys.executable, str(diff_logits),
         "--model", hf_id,
         "--atol", str(atol),
         "--max-cache-length", str(max_cache),
         "--battery"],
        capture_output=True, text=True, timeout=600)

    assert result.returncode == 0, (
        f"diff_logits failed for {hf_id}:\n{result.stderr}\n{result.stdout}")
