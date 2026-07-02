# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.e2e_harness.contracts import E2ECase, StageOutput, ThresholdProfile
from tests.e2e.models.qwen.e2e_plugins.contract import QwenSamplingPlugin


plugin = QwenSamplingPlugin()


def _case() -> E2ECase:
    return E2ECase(
        name="example-decoder-topp",
        hf_id="example-org/example-decoder",
        family="example_decoder",
        runtime_strategy="example_decoder_decoder_kv_cache",
        reference_backend="invariant_only",
        reference_family="sampling_top_p",
        user_contract="sampling",
        inputs={
            "prompt": "The capital of France is",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "seed": 42,
        },
    )


def test_sampling_contract_accepts_forwarded_top_p_flags() -> None:
    trt_output = StageOutput(
        stage_name="full_generation",
        text="The capital of France is Paris.",
        data={"cpp_returncode": 0},
        metadata={
            "cpp": {
                "command": [
                    "./build/trtmc",
                    "run",
                    "example-decoder.trtfb",
                    "--temperature",
                    "0.7",
                    "--top-p",
                    "0.9",
                    "--top-k",
                    "50",
                    "--seed",
                    "42",
                ]
            }
        },
    )

    result = plugin.verify(
        trt_output,
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert result.passed
    assert result.metrics["sampling_flags_forwarded"].passed


def test_sampling_contract_rejects_missing_top_p_flag() -> None:
    trt_output = StageOutput(
        stage_name="full_generation",
        text="The capital of France is Paris.",
        data={"cpp_returncode": 0},
        metadata={
            "cpp": {
                "command": [
                    "./build/trtmc",
                    "run",
                    "example-decoder.trtfb",
                    "--temperature",
                    "0.7",
                    "--top-k",
                    "50",
                    "--seed",
                    "42",
                ]
            }
        },
    )

    result = plugin.verify(
        trt_output,
        StageOutput("full_generation"),
        _case(),
        ThresholdProfile(task_strategy="text_generation_causal"),
    )

    assert not result.passed
    assert not result.metrics["sampling_flags_forwarded"].passed
    assert "--top-p" in result.metrics["sampling_flags_forwarded"].note
