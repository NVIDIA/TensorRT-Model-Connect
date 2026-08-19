# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for GPT-NeoX accuracy and its premerge case."""

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
from tensorrt_model_connect.models.gpt_neox.tests.e2e_plugins.contract import plugin


_MODEL_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _MODEL_DIR / "manifests" / "pythia-70m.json"
_THRESHOLD_PATH = _MODEL_DIR / "thresholds" / "pythia-70m.json"
_MMLU_000003_PROMPT_SHA256 = (
    "61f41777dda57d6f63957816782b22ea5951a914e04c60ee15ca2235bdb1eb0e"
)


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


def test_pythia_premerge_case_replays_the_observed_accuracy_failure():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    testcase = manifest["testcases"][0]
    prompt_sha256 = hashlib.sha256(
        testcase["prompt"].encode("utf-8")).hexdigest()
    thresholds = json.loads(
        _THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    assert prompt_sha256 == _MMLU_000003_PROMPT_SHA256
    assert testcase["max_new_tokens"] >= 26
    assert manifest["max_cache_length"] >= 393
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_pythia_contract_rejects_a_token_divergence_with_identical_text():
    result = plugin.verify(
        _output("same decoded continuation", [329, 187, 16708]),
        _output("same decoded continuation", [329, 187, 11793]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert not result.metrics["generated_token_agreement_rate"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_pythia_contract_accepts_exact_generated_tokens():
    result = plugin.verify(
        _output("same decoded continuation", [329, 187, 16708]),
        _output("same decoded continuation", [329, 187, 16708]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_agreement_rate"].passed
    assert result.metrics["generated_token_exact"].passed
