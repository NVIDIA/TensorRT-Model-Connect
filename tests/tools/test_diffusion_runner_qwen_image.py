"""Unit tests for DiffusionMediaRunner Qwen-Image CLI alignment.

Trace: ARCH-E2E-001, UD-E2E-CLI, UD-FAM-QWEN-IMAGE-01
Intent: Validate that DiffusionMediaRunner builds the correct ``trtmc run``
    argv for Qwen-Image manifests, threading negative_prompt, cfg_scale,
    height, width, num_inference_steps, and seed through to the C++ binary.
Preconditions: ``trtmc run`` (not ``generate-video``) is the entrypoint for
    image-only diffusion families that publish ``diffusion_qwen_image``.
Postconditions: The argv contains the expected Qwen-Image flags when the
    fields are present in the manifest, and falls back to the existing
    ``generate-video`` flow for other diffusion strategies.
"""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.runners import diffusion


def _make_qwen_image_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="qwen-image-case",
        hf_id="Qwen/Qwen-Image",
        family="qwen_image",
        runtime_strategy="diffusion_qwen_image",
        bundle="qwen-image-case.trtfb",
        inputs=inputs or {},
    )


def _make_wan_case(inputs: dict | None = None) -> E2ECase:
    return E2ECase(
        name="wan-case",
        hf_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        family="wan_t2v",
        runtime_strategy="diffusion_wan",
        bundle="wan-case.trtfb",
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


def _capture_subprocess(monkeypatch):
    """Patch subprocess.run inside the diffusion runner to capture argv."""
    captured: dict = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="Saved /tmp/x.png (1024x1024)\n", stderr="")

    monkeypatch.setattr(diffusion.subprocess, "run", _fake_run)
    return captured


def test_qwen_image_uses_run_entrypoint_with_all_flags(monkeypatch, tmp_path):
    """Qwen-Image case should dispatch to ``trtmc run`` with all flags."""
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
    captured = _capture_subprocess(monkeypatch)

    diffusion.DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    # Entrypoint: trtmc run, not generate-video.
    assert cmd[1] == "run"
    assert "generate-video" not in cmd

    # Prompt and flags.
    assert "--prompt" in cmd
    idx = cmd.index("--prompt")
    assert cmd[idx + 1] == "A red apple on a wooden table"

    assert "--negative-prompt" in cmd
    idx = cmd.index("--negative-prompt")
    assert cmd[idx + 1] == " "

    assert "--num-inference-steps" in cmd
    idx = cmd.index("--num-inference-steps")
    assert cmd[idx + 1] == "20"

    assert "--cfg-scale" in cmd
    idx = cmd.index("--cfg-scale")
    assert cmd[idx + 1] == "4.0"

    assert "--height" in cmd
    idx = cmd.index("--height")
    assert cmd[idx + 1] == "1024"

    assert "--width" in cmd
    idx = cmd.index("--width")
    assert cmd[idx + 1] == "1024"

    assert "--seed" in cmd
    idx = cmd.index("--seed")
    assert cmd[idx + 1] == "42"

    # Output should be a frame_0000.png file (so comparator frame glob picks it).
    assert "--output" in cmd
    idx = cmd.index("--output")
    assert cmd[idx + 1].endswith("frame_0000.png")


def test_qwen_image_omits_unset_flags(monkeypatch, tmp_path):
    """Optional flags should be omitted when the manifest doesn't set them."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "A cat on a beach",
            "num_inference_steps": 8,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    diffusion.DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "run"
    assert "--num-inference-steps" in cmd
    # Optional fields not provided -> flag absent.
    assert "--negative-prompt" not in cmd
    assert "--cfg-scale" not in cmd
    assert "--height" not in cmd
    assert "--width" not in cmd
    assert "--seed" not in cmd


def test_qwen_image_accepts_image_height_alias(monkeypatch, tmp_path):
    """``image_height`` / ``image_width`` should map to ``--height`` / ``--width``."""
    case = _make_qwen_image_case(
        inputs={
            "prompt": "scene",
            "image_height": 768,
            "image_width": 512,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    diffusion.DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert "--height" in cmd
    assert cmd[cmd.index("--height") + 1] == "768"
    assert "--width" in cmd
    assert cmd[cmd.index("--width") + 1] == "512"


def test_qwen_image_guidance_scale_falls_back_to_cfg_scale(monkeypatch, tmp_path):
    """If ``cfg_scale`` is absent, ``guidance_scale`` should be used."""
    case = _make_qwen_image_case(
        inputs={"prompt": "scene", "guidance_scale": 3.5},
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    diffusion.DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert "--cfg-scale" in cmd
    assert cmd[cmd.index("--cfg-scale") + 1] == "3.5"


def test_wan_case_still_uses_generate_video(monkeypatch, tmp_path):
    """Non-Qwen-Image diffusion families must keep the generate-video flow."""
    case = _make_wan_case(
        inputs={
            "prompt": "A cat sitting on a beach",
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "seed": 42,
        },
    )
    ctx = _make_ctx(case, tmp_path)
    captured = _capture_subprocess(monkeypatch)

    diffusion.DiffusionMediaRunner().run_stage(case, StageSpec(name="end_to_end"), ctx)

    cmd = captured["cmd"]
    assert cmd[1] == "generate-video"
    assert "run" not in cmd[1:2]
    # Wan keeps --num-steps + --guidance-scale.
    assert "--num-steps" in cmd
    assert "--guidance-scale" in cmd
    # Qwen-Image-only flags must NOT be present on the Wan command.
    assert "--negative-prompt" not in cmd
    assert "--cfg-scale" not in cmd
    assert "--height" not in cmd
    assert "--width" not in cmd
    assert "--num-inference-steps" not in cmd
