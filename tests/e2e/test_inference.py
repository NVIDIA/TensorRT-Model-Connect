# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E: C++ binary produces non-empty inference output."""

from __future__ import annotations

import subprocess
import pytest

_SPEECH_RUNTIME_STRATEGIES = {
    "speech_to_text",
    "speech_to_text_rnnt",
    "speech_to_speech",
}


@pytest.mark.e2e
def test_inference_produces_text(model_entry, trtmc_binary, hf_python, ld_library_path):
    """trtmc run <bundle> should generate non-empty text."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — no text inference")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — use test_audio_pipeline")
    if (
        model_entry.get("test_type") == "transcription"
        or model_entry.get("runtime_strategy") in _SPEECH_RUNTIME_STRATEGIES
    ):
        pytest.skip("Speech model — use speech/audio pipeline tests")
    prompt = model_entry.get("prompt", "Hello")
    max_new = model_entry.get("max_new_tokens", 10)

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "run", model_entry["bundle_path"],
         "--prompt", prompt,
         "--max-new-tokens", str(max_new),
         "--hf-python", str(hf_python)],
        capture_output=True, text=True, timeout=120, env=env)

    assert result.returncode == 0, f"Inference failed: {result.stderr}"
    output = result.stdout.strip()
    assert len(output) > 0, "Inference produced no output"


@pytest.mark.e2e
def test_inference_deterministic(model_entry, trtmc_binary, hf_python, ld_library_path):
    """Two runs with the same prompt should produce identical output."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — no text inference")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — use test_audio_pipeline")
    if (
        model_entry.get("test_type") == "transcription"
        or model_entry.get("runtime_strategy") in _SPEECH_RUNTIME_STRATEGIES
    ):
        pytest.skip("Speech model — use speech/audio pipeline tests")
    prompt = model_entry.get("prompt", "Hello")
    max_new = min(model_entry.get("max_new_tokens", 10), 5)

    env = {"LD_LIBRARY_PATH": ld_library_path}
    cmd = [str(trtmc_binary), "run", model_entry["bundle_path"],
           "--prompt", prompt,
           "--max-new-tokens", str(max_new),
           "--hf-python", str(hf_python)]

    r1 = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

    assert r1.returncode == 0 and r2.returncode == 0
    assert r1.stdout.strip() == r2.stdout.strip(), "Non-deterministic output"
