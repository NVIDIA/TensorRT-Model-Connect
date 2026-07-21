# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-owned repro command tests."""

from __future__ import annotations

import shlex
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import activate_model_plugins, reset


MODEL_DIR = Path(__file__).resolve().parent


def _l0_case() -> E2ECase:
    return E2ECase(
        name="wan22-ti2v-5b-l0",
        hf_id="Wan-AI/Wan2.2-TI2V-5B",
        family="wan2_2_ti2v",
        runtime_strategy="diffusion_wan2_2_ti2v",
        task_strategy="diffusion_media_generation",
        bundle="wan22-ti2v-5b-l0.trtfb",
        inputs={
            "prompt": "Two boxers' cats fight on a spotlighted stage",
            "video_num_frames": 5,
            "video_height": 384,
            "video_width": 672,
            "num_inference_steps": 15,
            "text_max_length": 512,
            "guidance_scale": 5.0,
            "flow_shift": 5.0,
            "fps": 24,
            "seed": 42,
        },
        stages=[],
    )


def test_wan22_repro_uses_native_generate_video_command(tmp_path) -> None:
    case = _l0_case()
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        binary_path="/work/build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/work/engines",
        model_plugin_dir="/work/build/models",
    )
    bundle_path = "/work/engines/wan22-ti2v-5b-l0.trtfb"

    activate_model_plugins(MODEL_DIR)
    try:
        repro = _build_repro_commands(case, ctx, bundle_path, {})
    finally:
        reset()

    argv = shlex.split(repro["trt_inference"])
    assert argv == [
        "/work/build/trtmc",
        "generate-video",
        bundle_path,
        "--prompt",
        "Two boxers' cats fight on a spotlighted stage",
        "--output",
        "/tmp/trtmc_wan22_ti2v_frames",
        "--num-steps",
        "15",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--height",
        "384",
        "--width",
        "672",
        "--backend-dir",
        "/work/build",
        "--model-plugin-dir",
        "/work/build/models",
    ]
    assert "--max-new-tokens" not in argv
