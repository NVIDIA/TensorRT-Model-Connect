# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PixArt HF-to-TRTMC parity contract tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.pixart.e2e_plugins.runners import diffusion
from tests.e2e.models.pixart.e2e_plugins.references import hf_diffusers
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _case(seed: int = 42, *, shared_initial_latents: bool = True) -> E2ECase:
    return E2ECase(
        name="partiprompts_000001",
        hf_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        family="pixart",
        runtime_strategy="diffusion_pixart",
        inputs={
            "image_height": 1024,
            "image_width": 1024,
            "seed": seed,
            "use_shared_initial_latents": shared_initial_latents,
        },
    )


def test_hf_and_trtmc_resolve_the_same_initial_latent(tmp_path) -> None:
    from tests.e2e.models.pixart.e2e_plugins.parity import ensure_initial_latents

    case = _case(seed=43)
    hf_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "hf_artifacts"))
    trt_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "trtfb_artifacts"))

    hf = ensure_initial_latents(case, hf_ctx)
    trt = ensure_initial_latents(case, trt_ctx)

    assert hf.path == trt.path
    assert hf.sha256 == trt.sha256
    assert hf.shape == (1, 4, 128, 128)
    assert hf.path.stat().st_size == 4 * 4 * 128 * 128


def test_trtmc_runner_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _case(seed=44)
    case.bundle = "pixart.trtfb"
    binary = tmp_path / "trtmc"
    binary.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "trtfb_artifacts"),
        binary_path=str(binary),
        engine_dir=str(tmp_path),
        model_plugin_dir="/runtime/models/pixart",
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
    assert command[command.index("--model-plugin-dir") + 1] == "/runtime/models/pixart"
    latent_path = command[command.index("--initial-latents-raw") + 1]
    assert latent_path.endswith("partiprompts_000001.seed-44.1024x1024.f32")
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
    assert "partiprompts_000001.seed-44.1024x1024.f32" in script
    assert output.data["initial_latents_sha256"]


def test_standard_trtmc_runner_keeps_seeded_generation_path(
    tmp_path, monkeypatch
) -> None:
    case = _case(seed=45, shared_initial_latents=False)
    case.bundle = "pixart.trtfb"
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

    assert "--initial-latents-raw" not in output.metadata["command"]
    assert output.data.get("initial_latents_sha256", "") == ""


def test_standard_hf_reference_keeps_seeded_generator_path(
    tmp_path, monkeypatch
) -> None:
    case = _case(seed=45, shared_initial_latents=False)
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
    assert "generator=torch.Generator" in script
    assert "latents=initial_latents" not in script
    assert output.data.get("initial_latents_sha256", "") == ""
