# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for InternLM's strict HF generation contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e_harness.contracts import (
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e.models.internlm.e2e_plugins.contract import plugin


_MODEL_DIR = Path(__file__).resolve().parent
_THRESHOLD_PATH = _MODEL_DIR / "thresholds" / "internlm2-1.8b.json"


def _case():
    return SimpleNamespace(inputs={"prompt": ""}, metadata={})


def _threshold():
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        metrics={
            "contract_ned_threshold": 0.15,
            "contract_token_agreement_rate": 1.0,
        },
    )


def _output(
    text: str,
    token_ids: list[int] | None,
    *,
    cpp_returncode: int = 0,
    cpp_runtime_error: str = "",
) -> StageOutput:
    data = {"cpp_returncode": cpp_returncode}
    if token_ids is not None:
        data["token_ids"] = token_ids
    if cpp_runtime_error:
        data["cpp_runtime_error"] = cpp_runtime_error
    return StageOutput(stage_name="full_generation", data=data, text=text)


def test_official_manifest_requires_exact_generated_tokens() -> None:
    thresholds = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_contract_rejects_token_divergence_with_identical_text() -> None:
    result = plugin.verify(
        _output("Paris is the capital.", [1, 2, 3]),
        _output("Paris is the capital.", [1, 2, 4]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["exact_match"].passed
    assert not result.metrics["generated_token_agreement_rate"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_contract_rejects_approximate_text_with_identical_tokens() -> None:
    result = plugin.verify(
        _output("Paris is the capital.", [1, 2, 3]),
        _output("Paris is the capital!", [1, 2, 3]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["exact_match"].passed
    assert result.metrics["ned"].passed
    assert result.metrics["generated_token_exact"].passed


@pytest.mark.parametrize("missing_side", ["trt", "hf"])
def test_contract_fails_closed_when_generated_token_ids_are_missing(
    missing_side: str,
) -> None:
    trt_tokens = None if missing_side == "trt" else [1, 2, 3]
    hf_tokens = None if missing_side == "hf" else [1, 2, 3]
    result = plugin.verify(
        _output("Paris is the capital.", trt_tokens),
        _output("Paris is the capital.", hf_tokens),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["generated_token_ids_available"].passed
    assert missing_side.upper() in result.message.upper()


def test_contract_accepts_exact_generated_tokens_and_text() -> None:
    result = plugin.verify(
        _output("Paris is the capital.", [1, 2, 3]),
        _output("Paris is the capital.", [1, 2, 3]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["exact_match"].passed
    assert result.metrics["generated_token_agreement_rate"].passed
    assert result.metrics["generated_token_exact"].passed


def test_contract_fails_closed_on_native_runtime_error() -> None:
    result = plugin.verify(
        _output(
            "Paris is the capital.",
            [1, 2, 3],
            cpp_returncode=-1,
            cpp_runtime_error="[TRT] ERROR: enqueue failed",
        ),
        _output("Paris is the capital.", [1, 2, 3]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["cpp_returncode_ok"].passed
    assert "enqueue failed" in result.message
