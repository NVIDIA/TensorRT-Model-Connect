"""SAM model-owned repro command tests."""

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
            runtime_strategy="sam_prompted_segmentation",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_sam_repro_command_comes_from_model_plugin(tmp_path) -> None:
    activate_model_plugins(REPO_ROOT / "tests" / "e2e" / "models" / "sam")
    try:
        case = E2ECase(
            name="sam-case",
            hf_id="facebook/sam-vit-base",
            family="sam",
            runtime_strategy="sam_prompted_segmentation",
            task_strategy="prompted_segmentation",
            bundle="sam-case.trtfb",
            inputs={
                "image": "data/test_img.jpeg",
                "point_x": 0.25,
                "point_y": 0.75,
                "is_foreground": False,
            },
            stages=[],
        )
        repro = _build_repro_commands(
            case,
            _make_ctx(tmp_path),
            "/tmp/engines/sam-case.trtfb",
            {},
        )
    finally:
        reset()

    cmd = repro["trt_inference"]
    assert " segment-prompted " in f" {cmd} "
    assert "--image data/test_img.jpeg" in cmd
    assert "--output /tmp/trtmc_masks" in cmd
    assert "--point-x 0.25" in cmd
    assert "--point-y 0.75" in cmd
    assert "--background" in cmd
    assert "--prompt" not in cmd
