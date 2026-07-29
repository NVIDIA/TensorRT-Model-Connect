# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for StableLM continuation accuracy and its CI sentinel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tests.e2e_harness.contracts import (
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e.models.stablelm.e2e_plugins.contract import plugin


_MODEL_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _MODEL_DIR / "manifests" / "stablelm2-1.6b.json"
_THRESHOLD_PATH = _MODEL_DIR / "thresholds" / "stablelm2-1.6b.json"
_STABLELM_REVISION = "f499ead74c53749bd93cebc6ce8bc0d7bdf1eaef"
_MMLU_000002_PROMPT_SHA256 = (
    "de28bf6b76387fa017c1ffa8e379f87c46439a0b0fc0186798c0f7b4b6a17330"
)
_QA_COMMON_PREFIX = [
    362,
    271,
    10086,
    682,
    10105,
    311,
    279,
    24524,
    304,
    1901,
    62,
    18,
    13,
    865,
    61,
    17,
    489,
    220,
]


def _case():
    return SimpleNamespace(inputs={"prompt": ""}, metadata={})


def _threshold():
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        metrics={
            "contract_ned_threshold": 0.25,
            "contract_token_agreement_rate": 1.0,
        },
    )


def _output(text: str, token_ids: list[int]) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={"cpp_returncode": 0, "token_ids": token_ids},
        text=text,
    )


def test_stablelm_premerge_case_replays_the_published_accuracy_signal():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    testcase = manifest["testcases"][0]
    prompt_sha256 = hashlib.sha256(
        testcase["prompt"].encode("utf-8")).hexdigest()
    thresholds = json.loads(
        _THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    assert manifest["hf_revision"] == _STABLELM_REVISION
    assert prompt_sha256 == _MMLU_000002_PROMPT_SHA256
    assert testcase["max_new_tokens"] >= 19
    assert testcase["reference_precision"] == manifest["precision"] == "fp16"
    assert manifest["max_cache_length"] >= 384
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_stablelm_contract_rejects_the_published_token_18_divergence():
    result = plugin.verify(
        _output("same decoded continuation", [*_QA_COMMON_PREFIX, 17]),
        _output("same decoded continuation", [*_QA_COMMON_PREFIX, 16]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert not result.metrics["generated_token_agreement_rate"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_stablelm_contract_accepts_exact_generated_tokens():
    exact_tokens = [*_QA_COMMON_PREFIX, 16]
    result = plugin.verify(
        _output("same decoded continuation", exact_tokens),
        _output("same decoded continuation", exact_tokens),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_agreement_rate"].passed
    assert result.metrics["generated_token_exact"].passed


def test_stablelm_contract_requires_reference_token_ids_for_strict_gate():
    trt_output = _output("same decoded continuation", [*_QA_COMMON_PREFIX, 16])
    reference_output = StageOutput(
        stage_name="full_generation",
        data={},
        text="same decoded continuation",
    )

    result = plugin.verify(
        trt_output,
        reference_output,
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["generated_token_ids_available"].passed
