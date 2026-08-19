# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned build and runtime command contracts."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect.models.wan2_2_ti2v.tests.e2e_plugins.runners.diffusion import DiffusionMediaRunner
from tests.e2e_harness.contracts import RunContext, StageSpec
from tests.e2e_harness.manifest_loader import load_manifest
from tests.e2e_harness.orchestrator import (
    _append_declared_build_cli_args,
    _build_repro_commands,
)
from tests.e2e_harness.registry import activate_model_plugins, reset


MODEL_DIR = Path(__file__).resolve().parent


def _case(manifest: str):
    return load_manifest(MODEL_DIR / "manifests" / manifest)


@pytest.mark.parametrize(
    ("manifest", "bundle", "steps", "height", "width"),
    (
        ("wan22-ti2v-5b.json", "wan22-ti2v-5b.bundle", "50", "704", "1280"),
        ("wan22-ti2v-5b-l0.json", "wan22-ti2v-5b-l0.bundle", "15", "384", "672"),
    ),
)
def test_repro_uses_the_fixed_profile_native_command(
    tmp_path: Path,
    manifest: str,
    bundle: str,
    steps: str,
    height: str,
    width: str,
) -> None:
    case = _case(manifest)
    context = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/work/build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/work/engines",
        model_plugin_dir="/work/build/models",
    )

    activate_model_plugins(MODEL_DIR)
    try:
        repro = _build_repro_commands(case, context, f"/work/engines/{bundle}", {})
    finally:
        reset()

    argv = shlex.split(repro["trt_inference"])
    assert argv[:3] == ["/work/build/trtmc", "generate-video", f"/work/engines/{bundle}"]
    assert argv[argv.index("--num-steps") + 1] == steps
    assert argv[argv.index("--height") + 1] == height
    assert argv[argv.index("--width") + 1] == width
    assert argv[argv.index("--model-plugin-dir") + 1] == "/work/build/models"
    assert "--hf-python" not in argv


@pytest.mark.parametrize("manifest", ["wan22-ti2v-5b.json", "wan22-ti2v-5b-l0.json"])
def test_manifest_profile_is_forwarded_to_bundle_build(manifest: str) -> None:
    case = _case(manifest)
    command: list[str] = []
    _append_declared_build_cli_args(command, case)
    assert command == [
        "--video-height",
        str(case.inputs["video_height"]),
        "--video-width",
        str(case.inputs["video_width"]),
        "--video-num-frames",
        str(case.inputs["video_num_frames"]),
        "--num-inference-steps",
        str(case.inputs["num_inference_steps"]),
    ]


def test_runner_cleans_frames_and_uses_strict_native_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_model_connect.models.wan2_2_ti2v.tests.e2e_plugins.runners import diffusion

    case = _case("wan22-ti2v-5b-l0.json")
    context = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/work/build/trtmc",
        engine_dir="/work/engines",
        model_plugin_dir="/work/build/models",
    )
    frames_dir = tmp_path / case.name / "frames"
    frames_dir.mkdir(parents=True)
    stale = frames_dir / "frame_9999.png"
    stale.touch()
    captured: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        captured["env"] = kwargs["env"]
        assert not stale.exists()
        for index in range(5):
            (frames_dir / f"frame_{index:04d}.png").touch()
        return subprocess.CompletedProcess(command, 0, stdout="generated", stderr="runtime-ready")

    monkeypatch.setattr(diffusion.subprocess, "run", run)
    monkeypatch.setattr(
        diffusion,
        "_frame_stats",
        lambda _paths: {
            "mean": 0.5,
            "std": 0.2,
            "width": 672,
            "height": 384,
            "dimensions_consistent": True,
        },
    )

    output = DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), context)

    env = captured["env"]
    assert env["TRTMC_MODEL_PLUGIN_DIR"] == "/work/build/models"
    assert env["TRTMC_MODEL_PLUGIN_STRICT"] == "1"
    assert output.data["frame_paths"] == [
        str(frames_dir / f"frame_{index:04d}.png") for index in range(5)
    ]
    assert output.data["frame_stats"]["dimensions_consistent"] is True
