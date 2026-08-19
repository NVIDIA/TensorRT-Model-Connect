# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess

from tensorrt_model_connect.models.flux.tests.e2e_plugins.references import hf_diffusers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_flux_reference_uses_per_sample_prompts_and_seeds(monkeypatch, tmp_path) -> None:
    case = E2ECase(
        name="flux-batch2",
        hf_id="example/flux",
        family="flux",
        runtime_strategy="diffusion_flux",
        inputs={
            "batch_prompts": ["A red cube", "A blue sphere"],
            "batch_seeds": [42, 42],
            "num_inference_steps": 4,
            "image_height": 64,
            "image_width": 64,
        },
    )
    ctx = RunContext(case=case, artifacts_dir=str(tmp_path))
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["script"] = cmd[2]
        return subprocess.CompletedProcess(cmd, 0, stdout="Generated 2 frames\n", stderr="")

    monkeypatch.setattr(hf_diffusers, "_resolve_cached_model_ref", lambda model: model)
    monkeypatch.setattr(hf_diffusers.subprocess, "run", _fake_run)

    hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    script = captured["script"]
    assert "prompts = ['A red cube', 'A blue sphere']" in script
    assert "batch_seeds = [42, 42]" in script
    assert "prompt=prompts if len(prompts) > 1 else prompts[0]" in script
    assert "for seed in batch_seeds" in script
