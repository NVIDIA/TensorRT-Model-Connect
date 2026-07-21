# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for selective Qwen EdgeLLM GitHub workflows."""

from __future__ import annotations

import re
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
ADAPTER_TEST_ROOT = REPOSITORY / "tests/e2e/models/qwen/edge_llm_adapter"


def test_reusable_a100_workflow_is_exact_pinned_and_runner_provisioned() -> None:
    workflow = (
        REPOSITORY / ".github/workflows/qwen-edgellm-a100.yml"
    ).read_text(encoding="utf-8")

    assert "  workflow_call:" in workflow
    assert "  workflow_dispatch:" in workflow
    assert 'ref: ${{ inputs.revision }}' in workflow
    assert 'test "$(git rev-parse HEAD^{commit})" = "$REVISION"' in workflow
    assert "TRTMC_EDGELLM_A100_RUNNER_LABELS" in workflow
    assert '[\"self-hosted\",\"Linux\",\"X64\",\"A100\"]' in workflow
    assert "'[\"self-hosted\"]'" not in workflow
    for variable in (
        "TRTMC_EDGELLM_CUDA_ROOT",
        "TRTMC_EDGELLM_TENSORRT_PYTHON_WHEEL",
        "TRTMC_EDGELLM_TENSORRT_ROOT",
    ):
        assert f"vars.{variable}" in workflow
    assert "arguments+=(--profile \"$PROFILE\")" in workflow
    assert "profile: ${{ inputs.profile }}" not in workflow
    assert "timeout-minutes: 360" in workflow
    assert "actions/upload-artifact" not in workflow
    assert "evidence/" not in workflow
    assert "ssh " not in workflow
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", workflow)


def test_a100_rollout_is_manual_until_an_x86_runner_is_provisioned() -> None:
    nightly = (REPOSITORY / ".github/workflows/nightly.yml").read_text(encoding="utf-8")
    premerge = (REPOSITORY / ".github/workflows/trtmc-ci.yml").read_text(encoding="utf-8")
    manual = (
        REPOSITORY / ".github/workflows/qwen-edgellm-a100.yml"
    ).read_text(encoding="utf-8")
    launcher_text = (ADAPTER_TEST_ROOT / "qualify_a100.py").read_text(encoding="utf-8")

    assert "\n  qwen-edgellm-a100:" not in nightly
    assert "\n  qwen-edgellm-a100:" not in premerge
    assert "qwen-edgellm-a100.yml" not in nightly
    assert "qwen-edgellm-a100.yml" not in premerge
    assert "workflow_dispatch:" in manual
    assert "- nightly" in manual
    assert 'default: ""' in manual
    assert "_discover_profiles()" in launcher_text
    assert "_run_coexistence_if_complete" in launcher_text

