# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the RWKV continuation contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.models.rwkv.e2e_plugins.contract import RwkvCausalContinuationPlugin
from tests.e2e_harness.contracts import (
    E2ECase,
    StageOutput,
    StageStatus,
    ThresholdProfile,
)


def test_rwkv_acceptance_manifest_retains_full_precision() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "rwkv-169m.json"
    manifest = json.loads(manifest_path.read_text())

    assert manifest["precision"] == "fp32"
    assert manifest["testcases"][0]["reference_precision"] == "fp32"


_REFERENCE = "the capital of the French Republic. The capital of"


def _case() -> E2ECase:
    return E2ECase(
        name="rwkv-169m",
        hf_id="RWKV/rwkv-4-169m-pile",
        family="rwkv",
        runtime_strategy="rwkv_recurrent",
        reference_family="causal_base_continuation",
        user_contract="continuation_parity",
        inputs={"prompt": "The capital of France is"},
    )


def _verify(trt_text: str):
    return RwkvCausalContinuationPlugin().verify(
        StageOutput(stage_name="full_generation", text=trt_text),
        StageOutput(stage_name="full_generation", text=_REFERENCE),
        _case(),
        ThresholdProfile(
            task_strategy="text_generation_causal",
            metrics={"normalized_text_edit_distance": 0.2},
        ),
    )


def test_rejects_nightly_continuation_above_declared_ned_limit() -> None:
    """Regression for the false pass in Nightly run 29445075597."""
    result = _verify("the capital of the French Republic. The")

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].value == pytest.approx(0.22)
    assert result.metrics["ned"].threshold == pytest.approx(0.2)
    assert not result.metrics["ned"].passed


def test_accepts_continuation_within_declared_ned_limit() -> None:
    result = _verify(_REFERENCE)

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["ned"].value == pytest.approx(0.0)
    assert result.metrics["ned"].threshold == pytest.approx(0.2)
    assert result.metrics["ned"].passed
