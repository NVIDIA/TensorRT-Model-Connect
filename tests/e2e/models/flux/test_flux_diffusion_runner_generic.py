# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned diffusion runner tests."""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e.models.flux.e2e_plugins.runners import diffusion as flux_diffusion


def _make_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="generic-media-case",
        hf_id="example/diffusion-model",
        family="flux",
        runtime_strategy="diffusion_flux",
        bundle="generic-media-case.trtfb",
        inputs=inputs or {},
    )


def _make_ctx(case: E2ECase, tmp_path) -> RunContext:
    binary_path = tmp_path / "trtmc"
    binary_path.write_text("", encoding="utf-8")
    return RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path=str(binary_path),
        engine_dir=str(tmp_path),
    )


def test_flux_diffusion_runner_uses_generate_video(monkeypatch, tmp_path):
    case = _make_case(
        inputs={
            "prompt": "A generated test scene",
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="Generated 1 frames\n", stderr="")

    monkeypatch.setattr(flux_diffusion.subprocess, "run", _fake_run)

    flux_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "generate-video"
    assert cmd[cmd.index("--num-steps") + 1] == "30"
    assert cmd[cmd.index("--guidance-scale") + 1] == "5.0"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--negative-prompt" not in cmd
    assert "--cfg-scale" not in cmd
    assert "--height" not in cmd
    assert "--width" not in cmd
    assert "--num-inference-steps" not in cmd
