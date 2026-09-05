# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for OLMo2 accuracy and its bounded premerge case."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from families.olmo2.tests import test_e2e as e2e


_TEST_DIR = Path(__file__).resolve().parent
_MANIFEST_PATH = _TEST_DIR / "manifests" / "olmo2-1b.json"
_THRESHOLD_PATH = _TEST_DIR / "thresholds" / "olmo2-1b.json"
_HF_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
_TOKEN_IDS = [426, 271, 10086, 279, 8547, 315, 279, 9070]


def _verify(actual_ids: list[int], reference_ids: list[int]) -> None:
    e2e._assert_correctness(
        {"token_ids": actual_ids, "text": "same decoded continuation"},
        {"max_new_tokens": 8},
        {"contract_token_agreement_rate": 1.0},
        reference_ids,
        "same decoded continuation",
        None,
        "same decoded continuation",
    )


def test_premerge_case_replays_the_bounded_accuracy_signal() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["testcases"]) == 1
    case = manifest["testcases"][0]
    thresholds = json.loads(_THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold_overrides"]

    prompt = case["prompt"]
    assert prompt.startswith("The following are multiple choice questions (with answers)")
    assert prompt.count("\nAnswer:") == 6
    assert prompt.endswith(
        "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n"
        "A. 0\nB. 4\nC. 2\nD. 6\nAnswer:"
    )
    assert manifest["hf_revision"] == _HF_REVISION
    assert manifest["precision"] == case["reference_precision"] == "fp32"
    assert case["max_new_tokens"] >= 8
    assert manifest["max_sequence_length"] >= 345
    assert thresholds["contract_token_agreement_rate"] == 1.0


def test_reference_resolves_the_pinned_checkpoint(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    captured = {}

    def snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    manifest = {"hf_id": "allenai/OLMo-2-0425-1B", "hf_revision": _HF_REVISION}
    assert e2e._checkpoint(manifest) == tmp_path
    assert captured == {
        "repo_id": "allenai/OLMo-2-0425-1B",
        "revision": _HF_REVISION,
    }


def test_contract_rejects_token_divergence_with_identical_text() -> None:
    with pytest.raises(AssertionError):
        _verify([*_TOKEN_IDS[:-1], 2115], _TOKEN_IDS)


def test_contract_accepts_exact_generated_tokens() -> None:
    _verify(_TOKEN_IDS, _TOKEN_IDS)
