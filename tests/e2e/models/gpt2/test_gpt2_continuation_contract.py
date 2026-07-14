# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the GPT-2 continuation contract."""

from __future__ import annotations

import json
from pathlib import Path

from tests.e2e.models.gpt2.e2e_plugins.contract import Gpt2CausalContinuationPlugin
from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile


def _case(*, preserve_prompt_echo: bool = False) -> E2ECase:
    metadata = {}
    if preserve_prompt_echo:
        metadata["contract_config"] = {"preserve_prompt_echo": True}
    return E2ECase(
        name="gpt2-continuation-contract",
        hf_id="openai-community/gpt2",
        family="gpt2",
        runtime_strategy="gpt2_decoder_kv_cache",
        reference_family="causal_base_continuation",
        user_contract="continuation_parity",
        inputs={"prompt": "The capital of France is"},
        metadata=metadata,
    )


def _verify(trt_text: str, ref_text: str, *, preserve_prompt_echo: bool = False):
    return Gpt2CausalContinuationPlugin().verify(
        StageOutput(stage_name="full_generation", text=trt_text),
        StageOutput(stage_name="full_generation", text=ref_text),
        _case(preserve_prompt_echo=preserve_prompt_echo),
        ThresholdProfile(
            task_strategy="text_generation_causal",
            metrics={"contract_ned_threshold": 0.25},
        ),
    )


def test_rejects_prompt_prefixed_trt_continuation() -> None:
    result = _verify("The capital of France is Paris.", "Paris.")

    assert not result.passed
    assert not result.metrics["prompt_excluded"].passed


def test_accepts_clean_trt_continuation() -> None:
    result = _verify("Paris.", "Paris.")

    assert result.passed
    assert result.metrics["prompt_excluded"].passed


def test_allows_prompt_echo_when_explicitly_configured() -> None:
    output = "The capital of France is Paris."
    result = _verify(output, output, preserve_prompt_echo=True)

    assert result.passed
    assert result.metrics["prompt_excluded"].passed


def test_acceptance_build_uses_stable_execution_configuration() -> None:
    manifest_path = Path(__file__).parent / "manifests" / "gpt2-125m.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["e2e_parallel_resource"] == "exclusive_gpu"
    assert manifest["precision"] == "fp32"
    assert manifest["testcases"] == [
        {
            "name": "gpt2-125m",
            "trace_id": "IT-E2E-GPT2-01",
            "reference_family": "causal_base_continuation",
            "user_contract": "continuation_parity",
            "reference_precision": "fp32",
            "prompt": "The capital of France is",
            "max_new_tokens": 20,
        }
    ]
