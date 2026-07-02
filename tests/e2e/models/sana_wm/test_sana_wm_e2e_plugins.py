# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SANA-WM model-owned E2E plugin command wiring."""

from __future__ import annotations

from tests.e2e.models.sana_wm.e2e_plugins.commands import (
    build_sana_wm_reference_command,
    build_sana_wm_trt_command,
)
from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.registry import (
    activate_model_plugins,
    get_reference,
    get_repro_command_provider,
    get_runner,
    reset,
)


def _make_case() -> E2ECase:
    return E2ECase(
        name="sana-wm-bidirectional",
        hf_id="Efficient-Large-Model/SANA-WM_bidirectional",
        family="sana_wm",
        runtime_strategy="diffusion_sana_wm",
        task_strategy="diffusion_media_generation",
        reference_backend="hf_diffusers",
        bundle="sana-wm-bidirectional.trtfb",
        inputs={
            "prompt_file": "tests/e2e/models/sana_wm/assets/demo_0.txt",
            "image": "tests/e2e/models/sana_wm/assets/demo_0.png",
            "action": "w-80,jw-40,w-40,lw-60,w-100",
            "translation_speed": 0.055,
            "rotation_speed_deg": 1.2,
            "camera_intrinsics": [797.87866, 830.0503, 844.2675, 463.7225],
            "video_num_frames": 321,
            "num_inference_steps": 60,
            "cfg_scale": 5.0,
            "fps": 16,
            "flow_shift": 9.8,
            "no_action_overlay": True,
        },
    )


def _make_ctx(case: E2ECase) -> RunContext:
    return RunContext(
        case=case,
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_sana_wm_trt_command_uses_model_card_inputs() -> None:
    case = _make_case()
    cmd = build_sana_wm_trt_command(
        case,
        _make_ctx(case),
        "/tmp/engines/sana-wm-bidirectional.trtfb",
        "/tmp/trtmc_frames",
    )

    assert cmd[:4] == [
        "./build/trtmc",
        "generate-video",
        "/tmp/engines/sana-wm-bidirectional.trtfb",
        "--prompt",
    ]
    assert (
        "sana_wm.image_path=tests/e2e/models/sana_wm/assets/demo_0.png" in cmd
    )
    assert "sana_wm.action=w-80,jw-40,w-40,lw-60,w-100" in cmd
    assert "sana_wm.intrinsics=797.87866,830.0503,844.2675,463.7225" in cmd
    assert "sana_wm.num_frames=321" in cmd
    assert "sana_wm.flow_shift=9.8" in cmd
    assert "--guidance-scale" in cmd
    assert "5.0" in cmd
    assert "--hf-python" not in cmd
    assert "--no_action_overlay" not in cmd


def test_sana_wm_reference_command_uses_model_card_spellings() -> None:
    case = _make_case()
    cmd = build_sana_wm_reference_command(case, "/usr/bin/python3", "/tmp/hf_frames")

    assert cmd[:2] == [
        "/usr/bin/python3",
        "tests/e2e/models/sana_wm/reference/inference_sana_wm.py",
    ]
    assert "--output_dir" in cmd
    assert "/tmp/hf_frames" in cmd
    assert "--translation_speed" in cmd
    assert "--rotation_speed_deg" in cmd
    assert "--num_frames" in cmd
    assert "--step" in cmd
    assert "--cfg_scale" in cmd
    assert "--flow_shift" in cmd
    assert "--no_action_overlay" in cmd


def test_sana_wm_model_plugins_register_runner_reference_and_repro() -> None:
    reset()
    activate_model_plugins("tests/e2e/models/sana_wm")

    assert get_runner("diffusion_media_generation") is not None
    assert get_reference("hf_diffusers") is not None
    assert get_repro_command_provider("sana_wm") is not None
