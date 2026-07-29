# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict HF token/text parity coverage for GPT-NeoX."""

from types import SimpleNamespace

from tests.e2e_harness.contracts import (
    StageOutput,
    StageStatus,
    ThresholdProfile,
)
from tests.e2e.models.gpt_neox.e2e_plugins.contract import plugin
from tests.e2e.models.gpt_neox.e2e_plugins.runners.text_generation import (
    _detect_trt_runtime_error,
)


def _output(text: str, token_ids: list[int]) -> StageOutput:
    return StageOutput(
        stage_name="full_generation",
        data={"cpp_returncode": 0, "token_ids": token_ids},
        text=text,
    )


def _verify(trt: StageOutput, ref: StageOutput):
    return plugin.verify(
        trt,
        ref,
        SimpleNamespace(inputs={"prompt": ""}, metadata={}),
        ThresholdProfile(
            task_strategy="text_generation_causal",
            metrics={},
        ),
    )


def test_contract_accepts_exact_hf_tokens_and_text():
    result = _verify(
        _output("same continuation", [253, 5347, 273]),
        _output("same continuation", [253, 5347, 273]),
    )

    assert result.status == StageStatus.PASSED.value
    assert result.metrics["generated_token_exact"].passed
    assert result.metrics["generated_text_exact"].passed


def test_contract_rejects_token_difference_hidden_by_same_text():
    result = _verify(
        _output("same continuation", [253, 5347, 273]),
        _output("same continuation", [253, 5347, 323]),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["ned"].passed
    assert not result.metrics["generated_token_exact"].passed


def test_contract_rejects_decoded_text_difference():
    result = _verify(
        _output("same continuation", [253, 5347, 273]),
        _output("same continuation\n", [253, 5347, 273]),
    )

    assert result.status == StageStatus.FAILED.value
    assert result.metrics["generated_token_exact"].passed
    assert not result.metrics["generated_text_exact"].passed


def test_contract_rejects_missing_trt_token_ids():
    trt = _output("same continuation", [253, 5347, 273])
    trt.data.pop("token_ids")

    result = _verify(
        trt,
        _output("same continuation", [253, 5347, 273]),
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["generated_token_ids_available"].passed


def test_contract_rejects_missing_hf_token_ids():
    ref = _output("same continuation", [253, 5347, 273])
    ref.data.pop("token_ids")

    result = _verify(
        _output("same continuation", [253, 5347, 273]),
        ref,
    )

    assert result.status == StageStatus.FAILED.value
    assert not result.metrics["generated_token_ids_available"].passed


def test_contract_rejects_failed_runtime_before_parity():
    trt = _output("same continuation", [253, 5347, 273])
    trt.data["cpp_returncode"] = -1

    result = _verify(
        trt,
        _output("same continuation", [253, 5347, 273]),
    )

    assert result.status == StageStatus.ERROR.value
    assert not result.metrics["cpp_returncode_ok"].passed


def test_runner_detects_tensor_rt_error_with_zero_process_return_code():
    stderr = (
        "[TRT] ERROR: IExecutionContext::enqueueV3: "
        "Error Code 1: Cuda Runtime (illegal memory access)"
    )

    assert _detect_trt_runtime_error(stderr) == stderr
