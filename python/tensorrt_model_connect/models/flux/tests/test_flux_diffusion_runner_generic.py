# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux-owned diffusion runner tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tensorrt_model_connect.models.flux.tests.e2e_plugins.runners import diffusion as flux_diffusion


def _make_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="generic-media-case",
        hf_id="example/diffusion-model",
        family="flux",
        runtime_strategy="diffusion_flux",
        bundle="generic-media-case.bundle",
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

    output = flux_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "generate-video"
    assert cmd[cmd.index("--num-steps") + 1] == "30"
    assert cmd[cmd.index("--guidance-scale") + 1] == "5.0"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--initial-latents-raw" in cmd
    assert output.data["initial_latents_sha256"]
    assert "--negative-prompt" not in cmd
    assert "--cfg-scale" not in cmd
    assert "--height" not in cmd
    assert "--width" not in cmd
    assert "--num-inference-steps" not in cmd


def test_flux_diffusion_runner_executes_declared_batch(monkeypatch, tmp_path):
    case = _make_case(
        inputs={
            "batch_prompts": ["A red cube", "A blue sphere"],
            "batch_seeds": [42, 42],
            "expected_batch_size": 2,
            "num_inference_steps": 4,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        prompts_path = Path(cmd[cmd.index("--prompts-file") + 1])
        captured["prompts"] = prompts_path.read_text(encoding="utf-8").splitlines()
        output = Path(cmd[cmd.index("--output") + 1])
        output.with_name(f"{output.stem}_0{output.suffix}").write_bytes(b"first")
        output.with_name(f"{output.stem}_1{output.suffix}").write_bytes(b"second")
        return subprocess.CompletedProcess(cmd, 0, stdout="Saved 2 images\n", stderr="")

    monkeypatch.setattr(flux_diffusion.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        flux_diffusion.DiffusionMediaRunner,
        "_compute_frame_stats",
        staticmethod(lambda _path: {"count": 2, "mean": 0.5, "std": 0.2}),
    )

    output = flux_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert captured["prompts"] == ["A red cube", "A blue sphere"]
    assert cmd[cmd.index("--seed") + 1] == "42,42"
    assert cmd[cmd.index("--num-steps") + 1] == "4"
    assert output.data["num_frames"] == 2
    assert len(output.data["frame_paths"]) == 2
