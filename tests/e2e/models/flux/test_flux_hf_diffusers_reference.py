# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux model-owned HF diffusers reference tests."""

from __future__ import annotations

import subprocess

import pytest

from tests.e2e.models.flux.e2e_plugins.references import hf_diffusers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


@pytest.mark.parametrize("model_type", ["flux", "flux.2"])
def test_flux_reference_uses_sequential_cpu_offload(monkeypatch, tmp_path, model_type):
    case = E2ECase(
        name=f"{model_type}-case",
        hf_id="black-forest-labs/example",
        family="flux",
        runtime_strategy="diffusion_flux",
        bundle="example.trtfb",
        inputs={"image_height": 384, "image_width": 384},
        metadata={"model_type": model_type},
    )
    context = RunContext(case=case, artifacts_dir=str(tmp_path))
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="Generated 1 frames\n", stderr="")

    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)
    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), context
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    script = cmd[cmd.index("-c") + 1]
    assert "pipe.enable_sequential_cpu_offload()" in script
    assert 'pipe.to("cuda")' not in script


def test_flux_reference_honors_nested_validation_precision(monkeypatch, tmp_path):
    case = E2ECase(
        name="flux-schnell",
        hf_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
        runtime_strategy="diffusion_flux",
        bundle="flux-schnell.trtfb",
        inputs={"image_height": 384, "image_width": 384},
        metadata={
            "model_type": "flux",
            "task_eval": {"reference_precision": "fp16"},
        },
    )
    context = RunContext(case=case, artifacts_dir=str(tmp_path))
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Generated 1 frames\n", stderr=""
        )

    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)
    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), context
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    script = cmd[cmd.index("-c") + 1]
    assert "reference_torch_dtype = torch.float16" in script
    assert "unpacked_latents.to(dtype=reference_torch_dtype)" in script
