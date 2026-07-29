# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e_harness.manifest_loader import load_manifest
from tools.validation import catalog as validation_catalog


def test_premerge_native_manifest_uses_family_build_defaults() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests") / "qwen3-0.6b-native-l0.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert case.metadata["ci_tier"] == "l0_only"
    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert case.metadata["reference_precision"] == "bf16"


def test_fp16_manifest_keeps_legacy_build_contract() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests") / "qwen3-0.6b-fp16.json"
    )
    case = load_manifest(manifest_path)

    assert case.metadata["precision"] == "fp16"
    assert case.inputs["max_cache_length"] == 256


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
        name: validation_catalog.manifest_record(manifest_dir / f"{name}.json")
        for name in ("qwen3-0.6b-fp8", "qwen3-0.6b-topp")
    }
    suite = validation_catalog.suite_by_id(
        validation_catalog.load_suites(),
        "mmlu_five_shot_mcq",
    )

    for model in models.values():
        assert model["reference_backend"] == "hf_transformers"
        assert model["reference_family"] == "chat_qwen3_posttrained"
        assert model["user_contract"] == "chat_response"
        assert validation_catalog.suite_match_reason(suite, model) == (
            True,
            "selected",
        )
    assert models["qwen3-0.6b-fp8"]["task_eval"]["reference_precision"] == "bf16"
    assert set(models) < set(suite["default_model_names"])
    assert topp_e2e.reference_backend == "invariant_only"
    assert topp_e2e.reference_family == "sampling_top_p"
    assert topp_e2e.user_contract == "sampling"
