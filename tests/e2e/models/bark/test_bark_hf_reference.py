# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bark-owned Hugging Face reference contract tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e.models.bark.e2e_plugins.references import hf_transformers
from tests.e2e_harness.contracts import RunContext, StageOutput, StageSpec
from tests.e2e_harness.manifest_loader import load_model_manifest


def _bark_small_model():
    manifest_path = Path(__file__).with_name("manifests") / "bark-small.json"
    return load_model_manifest(manifest_path)


def test_bark_small_and_nightly_probes_use_fp32_reference() -> None:
    model = _bark_small_model()

    assert {case.name for case in model.testcases} == {
        "bark-small",
        "bark-small-tts-probe01",
        "bark-small-tts-probe02",
    }
    assert all(case.metadata["reference_precision"] == "fp32" for case in model.testcases)


def test_bark_hf_reference_moves_model_and_inputs_to_available_device(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_subprocess(**kwargs) -> StageOutput:
        captured.update(kwargs)
        return StageOutput(stage_name="full_generation")

    monkeypatch.setattr(hf_transformers, "run_reference_subprocess", fake_subprocess)
    case = next(
        case for case in _bark_small_model().testcases if case.name == "bark-small-tts-probe02"
    )

    hf_transformers.HfTransformersReference()._run_text_to_audio_ref(
        case,
        StageSpec(name="full_generation"),
        RunContext(case=case, artifacts_dir=str(tmp_path)),
    )

    script = captured["command"][2]
    compile(script, "<bark-hf-reference>", "exec")
    assert '"cuda" if torch.cuda.is_available() else "cpu"' in script
    assert "torch_dtype=torch.float32" in script
    assert "model.to(device)" in script
    assert 'value.to(device) if hasattr(value, "to") else value' in script
    assert "random.seed(seed)" in script
    assert "torch.cuda.manual_seed_all(seed)" in script
