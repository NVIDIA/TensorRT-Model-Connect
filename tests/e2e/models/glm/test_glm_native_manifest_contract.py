# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-owned manifest contract tests for the native KV path."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from tests.e2e.models.glm.e2e_plugins.contract import plugin
from tests.e2e_harness.contracts import E2ECase, StageOutput, StageStatus, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


def test_native_manifest_uses_official_model_and_family_build_defaults() -> None:
    family_dir = Path(__file__).parent
    repository = Path(__file__).resolve().parents[4]
    manifest_path = family_dir / "manifests" / "glm-4-9b-l0.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)
    model_manifest = tomllib.loads((family_dir / "MODEL.toml").read_text(encoding="utf-8"))
    builder_manifest = tomllib.loads(
        (repository / "python/tensorrt_model_connect/families/glm/MODEL.toml").read_text(
            encoding="utf-8"
        )
    )
    runtime_manifest = tomllib.loads(
        (repository / "src/runtime/models/glm/MODEL.toml").read_text(encoding="utf-8")
    )

    assert raw["hf_id"] == "THUDM/glm-4-9b-hf"
    assert raw["runtime_strategy"] == "glm_decoder_kv_cache"
    assert "precision" not in raw
    assert "max_cache_length" not in raw
    assert case.metadata["ci_tier"] == "l0_only"
    assert case.metadata["reference_precision"] == "bf16"
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert case.threshold_overrides["contract_token_agreement_rate"] == 1.0
    assert model_manifest["test_manifests"] == [
        "manifests/glm-4-9b-l0.json",
        "manifests/glm-4-9b.json",
    ]
    assert builder_manifest["default_build_route"] == "build_routing.py|prefer_native_default"
    assert any(
        test.startswith("test_glm_native_kv_cache|") for test in runtime_manifest["runtime_tests"]
    )


def _contract_case() -> E2ECase:
    return E2ECase(
        name="glm-4-9b-l0",
        hf_id="THUDM/glm-4-9b-hf",
        family="glm",
        runtime_strategy="glm_decoder_kv_cache",
        task_strategy="text_generation_causal",
        reference_family="causal_base_continuation",
        user_contract="continuation_parity",
        inputs={"prompt": "hello"},
    )


def _threshold() -> ThresholdProfile:
    return ThresholdProfile(
        task_strategy="text_generation_causal",
        metrics={"contract_token_agreement_rate": 1.0},
    )


def test_native_contract_rejects_generated_token_divergence_even_when_text_matches() -> None:
    trt = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [10, 20, 30, 40, 50]},
        text="same decoded text",
    )
    ref = StageOutput(
        stage_name="full_generation",
        data={"token_ids": [10, 20, 31, 40, 50]},
        text="same decoded text",
    )

    result = plugin.verify(trt, ref, _contract_case(), _threshold())

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["contract_token_agreement_rate"].value == pytest.approx(0.8)
    assert not result.metrics["contract_token_agreement_rate"].passed
    assert result.metrics["ned"].passed


def test_native_contract_rejects_text_divergence_even_when_tokens_match() -> None:
    result = plugin.verify(
        StageOutput(
            stage_name="full_generation",
            data={"token_ids": [10, 20, 30]},
            text="same decoded text",
        ),
        StageOutput(
            stage_name="full_generation",
            data={"token_ids": [10, 20, 30]},
            text="same decoded text!",
        ),
        _contract_case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["exact_match"].passed
    assert result.metrics["ned"].passed
    assert result.metrics["contract_token_agreement_rate"].passed


@pytest.mark.parametrize("missing_side", ["trt", "ref"])
def test_native_contract_fails_closed_when_generated_token_ids_are_missing(
    missing_side: str,
) -> None:
    trt_data = {} if missing_side == "trt" else {"token_ids": [10]}
    ref_data = {} if missing_side == "ref" else {"token_ids": [10]}

    result = plugin.verify(
        StageOutput(stage_name="full_generation", data=trt_data, text="same"),
        StageOutput(stage_name="full_generation", data=ref_data, text="same"),
        _contract_case(),
        _threshold(),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["generated_token_ids_present"].passed
