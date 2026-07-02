# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model-owned repro command tests."""

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
            runtime_strategy="sam3_prompted_segmentation",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_sam3_repro_command_comes_from_model_plugin(tmp_path) -> None:
    activate_model_plugins(REPO_ROOT / "tests" / "e2e" / "models" / "sam3")
    try:
        case = E2ECase(
            name="sam3-case",
            hf_id="facebook/sam3",
            family="sam3",
            runtime_strategy="sam3_prompted_segmentation",
            task_strategy="prompted_segmentation",
            bundle="sam3-case.trtfb",
            inputs={
                "image": "data/test_img.jpeg",
                "prompt": "red car",
            },
            stages=[],
        )
        repro = _build_repro_commands(
            case,
            _make_ctx(tmp_path),
            "/tmp/engines/sam3-case.trtfb",
            {},
        )
    finally:
        reset()

    cmd = repro["trt_inference"]
    assert " segment-prompted " in f" {cmd} "
    assert "--image data/test_img.jpeg" in cmd
    assert "--output /tmp/trtmc_masks" in cmd
    assert "--prompt 'red car'" in cmd
    assert "--point-x" not in cmd
    assert "--point-y" not in cmd
