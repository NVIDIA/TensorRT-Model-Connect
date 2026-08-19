# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen native-KV chunked-prefill regression contract tests."""

from pathlib import Path

from tensorrt_model_connect.models.qwen.tests.e2e_plugins.contract import (
    QwenNativeKvChunkedPrefillRegressionPlugin,
)
from tensorrt_model_connect.models.qwen.tests.e2e_plugins.runners.text_generation import (
    _prompt_from_case,
)
from tests.e2e_harness.contracts import StageOutput, ThresholdProfile
from tests.e2e_harness.manifest_loader import load_manifest


def _case():
    path = (
        Path(__file__).with_name("manifests")
        / "qwen3-0.6b-regression-native-kv-chunked-prefill.json"
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


def test_native_kv_contract_requires_full_capacity_chunking_and_decode() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [123, 456],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=40960 (bundle max=40960, row=1 B)",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=100 launches=2",
                    ]
                ),
            }
        },
    )

    result = QwenNativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.passed


def test_native_kv_contract_rejects_single_prefill_launch() -> None:
    output = StageOutput(
        stage_name="full_generation",
        data={
            "cpp_returncode": 0,
            "prompt_token_count": 32769,
            "token_ids": [123, 456],
        },
        metadata={
            "cpp": {
                "trt_engine_decode_s": 0.001,
                "stderr": "\n".join(
                    [
                        "[trtmc] KV cache rows=40960 (bundle max=40960, row=1 B)",
                        '[trtmc.engine_timing] label="prefill_engine_plan:prefill" '
                        "execute_ms=100 launches=1",
                    ]
                ),
            }
        },
    )

    result = QwenNativeKvChunkedPrefillRegressionPlugin().verify(
        output,
        _reference_output(),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert "chunked_prefill_executed" in result.message
