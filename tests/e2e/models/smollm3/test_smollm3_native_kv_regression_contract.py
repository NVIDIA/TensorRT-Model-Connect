# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolLM3 native-KV chunked-prefill regression contract tests."""

from pathlib import Path

from tests.e2e.models.smollm3.e2e_plugins.contract import (
    SmolLM3NativeKvChunkedPrefillRegressionPlugin,
)
from tests.e2e.models.smollm3.e2e_plugins.runners.text_generation import (
    _prompt_from_case,
)
from tests.e2e_harness.contracts import StageOutput, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


def _case():
    path = (
        Path(__file__).with_name("manifests")
        / "minitron-4b-width-regression-native-kv-chunked-prefill.json"
    )
    return load_manifest(path)


def _reference_output() -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={"_invariant_only": True},
    )


def test_repeat_prompt_fixture_is_deterministic() -> None:
    prompt = _prompt_from_case(_case())

    assert len(prompt.split()) == 32768
    assert prompt.startswith("a a a")
    assert prompt.endswith("a\n")


def test_chunked_prefill_contract_requires_native_capacity_and_all_chunks() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [17, 13],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=131072 (bundle max=131072, row=1 B)",
                        "[trtmc.prefill] tokens=32770 launches=513 max_chunk=64",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=953.48 launches=513",
                    ]
                ),
            }
        },
    )

    result = SmolLM3NativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.passed


def test_chunked_prefill_contract_rejects_single_prefill_call() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [17, 13],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=131072 (bundle max=131072, row=1 B)",
                        "[trtmc.prefill] tokens=32770 launches=1 max_chunk=32770",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=953.48 launches=1",
                    ]
                ),
            }
        },
    )

    result = SmolLM3NativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert "chunked_prefill_executed" in result.message


def test_chunked_prefill_contract_rejects_oversized_chunk() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [17, 13],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=131072 (bundle max=131072, row=1 B)",
                        "[trtmc.prefill] tokens=32770 launches=513 max_chunk=65",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=953.48 launches=513",
                    ]
                ),
            }
        },
    )

    result = SmolLM3NativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert "prefill_chunk_limit_observed" in result.message


def test_chunked_prefill_contract_rejects_runtime_token_count_mismatch() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [17, 13],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=131072 (bundle max=131072, row=1 B)",
                        "[trtmc.prefill] tokens=32769 launches=513 max_chunk=64",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=953.48 launches=513",
                    ]
                ),
            }
        },
    )

    result = SmolLM3NativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert "prefill_chunk_limit_observed" in result.message
