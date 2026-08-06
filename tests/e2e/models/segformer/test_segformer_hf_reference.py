# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tests.e2e.models.segformer.e2e_plugins.references import hf_transformers
from tests.e2e_harness.contracts import (
    E2ECase,
    RunContext,
    StageOutput,
    StageSpec,
)


def test_segmentation_reference_aligns_floating_inputs_with_model_dtype(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    captured: dict[str, object] = {}

    def fake_run_reference_subprocess(**kwargs):
        captured.update(kwargs)
        return StageOutput(stage_name="full_inference")

    monkeypatch.setattr(
        hf_transformers,
        "run_reference_subprocess",
        fake_run_reference_subprocess,
    )
    case = E2ECase(
        name="segformer-unit",
        hf_id="nvidia/segformer-b0-finetuned-ade-512-512",
        family="segformer",
        runtime_strategy="segformer_segmentation",
        task_strategy="segmentation",
        bundle="segformer-unit.bundle",
        inputs={"image": str(image)},
        metadata={"reference_precision": "fp16"},
    )

    hf_transformers.HfTransformersReference()._run_segmentation_ref(
        case,
        StageSpec(name="full_inference"),
        RunContext(
            case=case,
            artifacts_dir=str(tmp_path / "artifacts"),
            reference_python="/opt/venv/bin/python",
        ),
    )

    command = captured["command"]
    assert isinstance(command, list)
    script = command[command.index("-c") + 1]
    assert "reference_dtype = torch.float16" in script
    assert 'device = torch.device("cuda")' in script
    assert "model.eval().to(device)" in script
    assert "value.to(device=device, dtype=reference_dtype)" in script
    assert "if value.is_floating_point()" in script
