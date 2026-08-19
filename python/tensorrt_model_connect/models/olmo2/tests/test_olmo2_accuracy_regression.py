# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for OLMo2 accuracy and its bounded premerge case."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from tests.e2e_harness.contracts import (
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from tensorrt_model_connect.models.olmo2.tests.e2e_plugins.contract import plugin
from tensorrt_model_connect.models.olmo2.tests.e2e_plugins.references import hf_transformers


_MODEL_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _MODEL_DIR / "manifests" / "olmo2-1b.json"
_THRESHOLD_PATH = _MODEL_DIR / "thresholds" / "olmo2-1b.json"
_HF_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
_MMLU_000000_PROMPT_SHA256 = "ff9b00310e214beacf4b324da8c9e6c1de4e49cb422f1bcb1f7037b78f40a2a3"


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


def test_olmo2_premerge_case_replays_the_observed_accuracy_failure():
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["testcases"]) == 1
    testcase = manifest["testcases"][0]
    prompt_sha256 = hashlib.sha256(testcase["prompt"].encode("utf-8")).hexdigest()
    thresholds = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    assert prompt_sha256 == _MMLU_000000_PROMPT_SHA256
    assert manifest["hf_revision"] == _HF_REVISION
    assert manifest["precision"] == "fp32"
    assert testcase["max_new_tokens"] >= 8
    assert manifest["max_cache_length"] >= 345
    assert testcase["reference_precision"] == "fp32"
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_olmo2_reference_resolves_the_pinned_checkpoint(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(hf_id, **kwargs):
        captured["hf_id"] = hf_id
        captured["kwargs"] = kwargs
        return str(tmp_path)

    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    resolved = hf_transformers._resolve_olmo2_cached_model_ref(
        "allenai/OLMo-2-0425-1B", _HF_REVISION
    )

    assert resolved == str(tmp_path)
    assert captured == {
        "hf_id": "allenai/OLMo-2-0425-1B",
        "kwargs": {
            "local_files_only": True,
            "revision": _HF_REVISION,
        },
    }


def test_olmo2_contract_rejects_a_token_divergence_with_identical_text():
    result = plugin.verify(
        _output("same decoded continuation", [426, 271, 10086, 279, 8547, 315, 279, 2115]),
        _output("same decoded continuation", [426, 271, 10086, 279, 8547, 315, 279, 9070]),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert not result.metrics["generated_token_agreement_rate"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_olmo2_contract_accepts_exact_generated_tokens():
    token_ids = [426, 271, 10086, 279, 8547, 315, 279, 9070]
    result = plugin.verify(
        _output("same decoded continuation", token_ids),
        _output("same decoded continuation", token_ids),
        _case(),
        _threshold(),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_agreement_rate"].passed
    assert result.metrics["generated_token_exact"].passed
