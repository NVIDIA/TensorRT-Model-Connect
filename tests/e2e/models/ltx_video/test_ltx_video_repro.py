# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX Video model-owned repro command tests."""

from __future__ import annotations

from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import activate_model_plugins, reset


REPO_ROOT = Path(__file__).resolve().parents[4]


def _make_ctx(tmp_path) -> RunContext:
    return RunContext(
        case=E2ECase(
            name="case-a",
            hf_id="dummy/model",
            family="dummy",
            runtime_strategy="diffusion",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_ltx_video_repro_initial_latents_comes_from_model_plugin(tmp_path) -> None:
    activate_model_plugins(REPO_ROOT / "tests" / "e2e" / "models" / "ltx_video")
    try:
        case = E2ECase(
            name="ltx-video-case",
            hf_id="Lightricks/LTX-Video",
            family="ltx_video",
            runtime_strategy="diffusion",
            task_strategy="diffusion_media_generation",
            bundle="ltx-video.trtfb",
            inputs={
                "prompt": "A slow pan over mountains",
                "num_inference_steps": 8,
                "guidance_scale": 3.0,
                "seed": 7,
            },
            stages=[],
        )
        repro = _build_repro_commands(
            case,
            _make_ctx(tmp_path),
            "/tmp/engines/ltx-video.trtfb",
            {},
        )
    finally:
        reset()

    cmd = repro["trt_inference"]
    assert " generate-video " in f" {cmd} "
    assert "--num-steps 8" in cmd
    assert "--guidance-scale 3.0" in cmd
    assert "--seed 7" in cmd
    assert "--initial-latents-raw" in cmd
    assert "ltx-video-case/initial_latents.raw" in cmd
