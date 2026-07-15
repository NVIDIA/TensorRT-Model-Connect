# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan T2V HF-to-TRTMC initial-latent parity contract tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.wan_t2v.e2e_plugins.references import hf_diffusers
from tests.e2e.models.wan_t2v.e2e_plugins.runners import diffusion
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _case(seed: int = 42) -> E2ECase:
    return E2ECase(
        name="vbench_000001",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        family="wan_t2v",
        runtime_strategy="diffusion_wan",
        bundle="wan21-t2v-1.3b-l0.trtfb",
        inputs={
            "prompt": "A red robot walks through a garden",
            "video_num_frames": 5,
            "video_height": 384,
            "video_width": 672,
            "num_inference_steps": 1,
            "seed": seed,
            "use_shared_initial_latents": True,
        },
    )


def test_hf_and_trtmc_resolve_the_same_initial_latent(tmp_path) -> None:
    from tests.e2e.models.wan_t2v.e2e_plugins.parity import ensure_initial_latents

    case = _case(seed=43)
    hf_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "hf_artifacts"))
    trt_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "trtfb_artifacts"))

    hf = ensure_initial_latents(case, hf_ctx)
    trt = ensure_initial_latents(case, trt_ctx)

    assert hf.path == trt.path
    assert hf.sha256 == trt.sha256
    assert hf.shape == (1, 16, 2, 48, 84)
    assert hf.path.stat().st_size == 4 * 16 * 2 * 48 * 84


def test_trtmc_runner_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _case(seed=44)
    binary = tmp_path / "trtmc"
    binary.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "trtfb_artifacts"),
        binary_path=str(binary),
        engine_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        diffusion.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr=""
        ),
    )

    output = diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    command = output.metadata["command"]
    latent_path = command[command.index("--initial-latents-raw") + 1]
    assert "shared_initial_latents" in latent_path
    assert output.data["initial_latents_sha256"]


def test_hf_reference_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _case(seed=44)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "hf_artifacts"),
        reference_python="/opt/venv/bin/python",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hf_diffusers, "_resolve_cached_model_ref", lambda _id: "/model")
    monkeypatch.setattr(hf_diffusers.subprocess, "run", fake_run)

    output = hf_diffusers.HfDiffusersReference().run_stage(
        case, StageSpec(name="end_to_end"), ctx
    )

    script = captured["cmd"][2]
    assert "latents=initial_latents" in script
    assert "shared_initial_latents" in script
    assert output.data["initial_latents_sha256"]
