# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen Image model-owned diffusion runner tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.qwen_image.e2e_plugins.runners import diffusion as qwen_image_diffusion
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def _make_qwen_image_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="qwen-image-case",
        hf_id="Qwen/Qwen-Image",
        family="qwen_image",
        runtime_strategy="diffusion_qwen_image",
        bundle="qwen-image-case.trtfb",
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


def _capture_subprocess(monkeypatch, module):
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="Saved /tmp/x.png (1024x1024)\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    return captured


def test_qwen_image_uses_run_entrypoint_with_all_flags(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "A red apple on a wooden table",
            "negative_prompt": " ",
            "num_inference_steps": 20,
            "cfg_scale": 4.0,
            "height": 1024,
            "width": 1024,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, qwen_image_diffusion)

    output = qwen_image_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "generate-video" not in cmd
    assert cmd[cmd.index("--prompt") + 1] == "A red apple on a wooden table"
    assert cmd[cmd.index("--negative-prompt") + 1] == " "
    assert cmd[cmd.index("--num-inference-steps") + 1] == "20"
    assert cmd[cmd.index("--cfg-scale") + 1] == "4.0"
    assert cmd[cmd.index("--height") + 1] == "1024"
    assert cmd[cmd.index("--width") + 1] == "1024"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--initial-latents-raw" in cmd
    assert output.data["initial_latents_sha256"]
    assert cmd[cmd.index("--output") + 1].endswith("frame_0000.png")


def test_qwen_image_omits_unset_flags(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "A cat on a beach",
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, qwen_image_diffusion)

    qwen_image_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--num-inference-steps" in cmd
    assert "--negative-prompt" not in cmd
    assert "--cfg-scale" not in cmd
    assert "--height" not in cmd
    assert "--width" not in cmd
    assert "--seed" not in cmd


def test_qwen_image_accepts_image_height_alias(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "image_height": 768,
            "image_width": 512,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, qwen_image_diffusion)

    qwen_image_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--height") + 1] == "768"
    assert cmd[cmd.index("--width") + 1] == "512"


def test_qwen_image_guidance_scale_falls_back_to_cfg_scale(monkeypatch, tmp_path):
    case = _make_qwen_image_case(
        inputs={"prompt": "scene", "guidance_scale": 3.5},
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, qwen_image_diffusion)

    qwen_image_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--cfg-scale") + 1] == "3.5"


def test_qwen_image_threads_image_input_for_edit(monkeypatch, tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_text("stub", encoding="utf-8")
    case = _make_qwen_image_case(
        inputs={"prompt": "turn it into watercolor", "image": str(image_path)},
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch, qwen_image_diffusion)

    qwen_image_diffusion.DiffusionMediaRunner().run_stage(
        case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--image") + 1] == str(image_path)
