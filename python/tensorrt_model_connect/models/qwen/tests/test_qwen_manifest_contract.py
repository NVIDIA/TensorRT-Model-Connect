# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e_harness import model_runner
from tests.e2e_harness.contracts import RunContext
from tests.e2e_harness.manifest_loader import get_model_by_name, load_manifest
from tests.e2e_harness.orchestrator import _build_repro_commands
from tests.e2e_harness.registry import reset
from tools.validation import catalog as validation_catalog


MODEL_DIR = Path(__file__).parent


class _Config:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name: str, default=None):
        return self._options.get(name, default)


def _case_matches(_case, _filters) -> bool:
    return True


def _is_multi_device(case) -> bool:
    return case.metadata.get("ci_tier") == "multi_device"


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


def test_native_kv_regression_exceeds_one_prefill_profile() -> None:
    manifest_path = (
        Path(__file__).with_name("manifests")
        / "qwen3-0.6b-regression-native-kv-chunked-prefill.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = load_manifest(manifest_path)

    assert case.hf_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert "precision" not in manifest
    assert "max_cache_length" not in manifest
    assert "precision" not in case.metadata
    assert "max_cache_length" not in case.inputs
    assert case.metadata["test_category"] == "regression"
    assert case.metadata["ci_tier"] == "default"
    assert case.reference_backend == "invariant_only"
    assert case.oracle_level == "L4_invariants"
    assert case.reference_family == "qwen_native_kv_chunked_prefill_regression"
    assert case.user_contract == "runtime_invariants"
    assert case.inputs["prompt_repeat"] == {
        "text": "a",
        "separator": " ",
        "count": 32768,
        "suffix": "\n",
    }
    assert case.inputs["expected_prompt_tokens"] == 32769
    assert case.metadata["expected_kv_cache_rows"] == 40960
    assert case.metadata["expected_prefill_chunks"] == 2
    assert case.metadata["expected_prefill_chunk_limit"] == 32768
    assert case.inputs["max_new_tokens"] == 2
    assert case.inputs["temperature"] == 0.0
    assert case.inputs["top_k"] == 1
    assert case.metadata["regression"] == {
        "id": "qwen3-native-kv-full-context-chunked-prefill",
        "issue": "https://github.com/NVIDIA/TensorRT-Model-Connect/pull/673",
        "previous_failure": (
            "The Qwen prefill path could only submit a prompt within the "
            "prefill engine's 32768-token profile in one call, so it could not "
            "progress through the model's larger 40960-token native KV capacity."
        ),
        "prevents": (
            "A model-only Qwen build losing the BF16 native-KV route or failing "
            "to split a 32769-token prompt across the full 40960-token cache "
            "before decode and clean teardown."
        ),
    }


def test_native_kv_repro_preserves_model_only_build(tmp_path) -> None:
    reset()
    manifest = (
        MODEL_DIR
        / "manifests"
        / "qwen3-0.6b-regression-native-kv-chunked-prefill.json"
    )
    case = load_manifest(manifest)
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path / "artifacts"),
        binary_path="./build/trtmc",
        hf_python="/usr/bin/python3",
        engine_dir=str(tmp_path),
    )
    bundle = str(tmp_path / case.bundle)

    repro = _build_repro_commands(case, ctx, bundle, {})

    assert "--max-cache-length" not in repro["build_bundle"]
    assert "--precision" not in repro["build_bundle"]
    assert f"--model-revision {case.hf_revision}" in repro["build_bundle"]
    resolved_prompt = tmp_path / "artifacts" / case.name / "resolved_prompt.txt"
    assert f"--prompts-file {resolved_prompt}" in repro["trt_inference"]
    assert "--max-new-tokens 2" in repro["trt_inference"]
    assert "--temperature 0.0" in repro["trt_inference"]
    assert "--e2e-category regression" in repro["rerun_test_rebuild"]


def test_category_filter_separates_e2e_from_historical_regressions() -> None:
    regression = get_model_by_name(
        "qwen3-0.6b-regression-native-kv-chunked-prefill", MODEL_DIR
    )
    ordinary = get_model_by_name("qwen3-0.6b-fp16", MODEL_DIR)
    assert regression is not None
    assert ordinary is not None

    config = _Config(
        **{
            "--e2e-category": "regression",
            "--e2e-exclude-ci-tier": [],
        }
    )
    selected_regressions = model_runner.selected_testcases(
        regression,
        config=config,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )
    selected_ordinary = model_runner.selected_testcases(
        ordinary,
        config=config,
        case_matches_model=_case_matches,
        is_multi_device_case=_is_multi_device,
    )

    assert [case.name for case in selected_regressions] == [
        "qwen3-0.6b-regression-native-kv-chunked-prefill"
    ]
    assert selected_ordinary == []


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
    fp8_e2e = load_manifest(manifest_dir / "qwen3-0.6b-fp8.json")
    topp_e2e = load_manifest(manifest_dir / "qwen3-0.6b-topp.json")
    models = {
        name: validation_catalog.manifest_record(manifest_dir / f"{name}.json")
        for name in ("qwen3-0.6b-fp8", "qwen3-0.6b-topp")
    }
    suite = validation_catalog.suite_by_id(
        validation_catalog.load_suites(
            _owners={"qwen"},
            _require_all_suites=False,
        ),
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
    fp8 = models["qwen3-0.6b-fp8"]
    assert fp8["precision"] == "fp16"
    assert fp8["bundle"] == "qwen3-0.6b-fp8-fp16base.bundle"
    assert fp8["task_eval"]["reference_precision"] == "fp16"
    assert fp8["max_cache_length"] == 256
    assert fp8_e2e.metadata["build_args"] == {
        "decoder_engine_layout": "dual_profile",
    }
    assert fp8_e2e.inputs["prompt"].startswith(
        "The following are multiple choice questions (with answers) "
        "about miscellaneous."
    )
    assert fp8_e2e.metadata["expected_answers"] == ["C"]
    assert fp8_e2e.inputs["max_new_tokens"] == 1
    assert fp8_e2e.metadata["contract_config"] == {
        "use_chat_template": False,
        "enable_thinking": False,
    }
    assert set(models) < set(suite["default_model_names"])
    assert topp_e2e.reference_backend == "invariant_only"
    assert topp_e2e.reference_family == "sampling_top_p"
    assert topp_e2e.user_contract == "sampling"
