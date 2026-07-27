# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned manifest contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest
from tools import task_eval


@pytest.mark.parametrize(
    "manifest_name",
    ["qwen3-0.6b-fp8.json", "qwen3-0.6b-fp8-tp4.json"],
)
def test_qwen3_fp8_manifest_declares_hf_text_generation_contract(
    manifest_name: str,
) -> None:
    manifest_path = Path(__file__).with_name("manifests") / manifest_name
    case = load_manifest(manifest_path)

    assert case.hf_id == "Qwen/Qwen3-0.6B"
    assert case.task_strategy == "text_generation_causal"
    assert case.user_contract == "text-generation"
    assert not case.metadata.get("skip_reason")


def test_fp8_and_topp_use_deterministic_mmlu_validation_contract() -> None:
    manifest_dir = Path(__file__).with_name("manifests")
    topp_e2e = load_manifest(manifest_dir / "qwen3-0.6b-topp.json")
    models = {
        name: task_eval.manifest_record(manifest_dir / f"{name}.json")
        for name in ("qwen3-0.6b-fp8", "qwen3-0.6b-topp")
    }
    suite = task_eval.suite_by_id(
        task_eval.load_suites(),
        "mmlu_five_shot_mcq",
    )

    for model in models.values():
        assert model["reference_backend"] == "hf_transformers"
        assert model["reference_family"] == "chat_qwen3_posttrained"
        assert model["user_contract"] == "chat_response"
        assert task_eval.suite_match_reason(suite, model) == (True, "selected")
    assert set(models) < set(suite["default_model_names"])
    assert topp_e2e.reference_backend == "invariant_only"
    assert topp_e2e.reference_family == "sampling_top_p"
    assert topp_e2e.user_contract == "sampling"
