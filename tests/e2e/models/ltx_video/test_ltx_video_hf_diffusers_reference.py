# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video Hugging Face reference cache contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from types import ModuleType

from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns
from tests.e2e.models.ltx_video.e2e_plugins.contracts import ensure_initial_latents
from tests.e2e.models.ltx_video.e2e_plugins.references import hf_diffusers
from tests.e2e.models.ltx_video.e2e_plugins.runners import diffusion
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_cached_model_ref_uses_the_selective_snapshot_contract(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []
    expected_kwargs = {
        "allow_patterns": hf_snapshot_allow_patterns(),
        "local_files_only": True,
    }

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append((repo_id, kwargs))
        if kwargs != expected_kwargs:
            raise RuntimeError("selective snapshot rejected without its allowlist")
        return str(snapshot)

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    resolved = hf_diffusers._resolve_cached_model_ref("Lightricks/LTX-Video")

    assert resolved == str(snapshot)
    assert calls == [
        (
            "Lightricks/LTX-Video",
            expected_kwargs,
        )
    ]


def _parity_case(seed: int = 42) -> E2ECase:
    return E2ECase(
        name="vbench_000001",
        hf_id="Lightricks/LTX-Video",
        family="ltx_video",
        runtime_strategy="diffusion_ltx",
        bundle="ltx-video-l0.bundle",
        inputs={
            "prompt": "A red robot walks through a garden",
            "video_num_frames": 9,
            "video_height": 256,
            "video_width": 256,
            "num_inference_steps": 1,
            "seed": seed,
            "use_shared_initial_latents": True,
        },
    )


def test_hf_and_trtmc_resolve_the_same_initial_latent(tmp_path) -> None:
    case = _parity_case(seed=43)
    hf_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "hf_artifacts"))
    trt_ctx = RunContext(case=case, artifacts_dir=str(tmp_path / "bundle_artifacts"))

    hf = ensure_initial_latents(case, hf_ctx)
    trt = ensure_initial_latents(case, trt_ctx)

    assert hf.path == trt.path
    assert hf.sha256 == trt.sha256
    assert hf.shape == (1, 128, 128)
    assert hf.path.stat().st_size == 4 * 128 * 128


def test_trtmc_runner_consumes_and_reports_shared_initial_latent(
    tmp_path, monkeypatch
) -> None:
    case = _parity_case(seed=44)
    binary = tmp_path / "trtmc"
    binary.write_text("", encoding="utf-8")
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "bundle_artifacts"),
        binary_path=str(binary),
        engine_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        diffusion.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
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
    case = _parity_case(seed=44)
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
