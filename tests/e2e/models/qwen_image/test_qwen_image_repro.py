"""Qwen Image model-owned repro command tests."""

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
            runtime_strategy="diffusion_qwen_image",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_qwen_image_repro_command_comes_from_model_plugin(tmp_path) -> None:
    activate_model_plugins(REPO_ROOT / "tests" / "e2e" / "models" / "qwen_image")
    try:
        case = E2ECase(
            name="qwen-image-case",
            hf_id="Qwen/Qwen-Image",
            family="qwen_image",
            runtime_strategy="diffusion_qwen_image",
            task_strategy="diffusion_media_generation",
            bundle="qwen-image-case.trtfb",
            inputs={
                "prompt": "A red apple on a wooden table",
                "negative_prompt": " ",
                "num_inference_steps": 20,
                "cfg_scale": 4.0,
                "height": 1024,
                "width": 1024,
                "seed": 42,
            },
            stages=[],
        )
        repro = _build_repro_commands(
            case,
            _make_ctx(tmp_path),
            "/tmp/engines/qwen-image-case.trtfb",
            {},
        )
    finally:
        reset()

    cmd = repro["trt_inference"]
    assert " run " in f" {cmd} "
    assert "generate-video" not in cmd
    assert "--output /tmp/trtmc_qwen_image/output.png" in cmd
    assert "--num-inference-steps 20" in cmd
    assert "--negative-prompt ' '" in cmd
    assert "--cfg-scale 4.0" in cmd
    assert "--height 1024" in cmd
    assert "--width 1024" in cmd
    assert "--seed 42" in cmd
    assert "--initial-latents-raw" in cmd
