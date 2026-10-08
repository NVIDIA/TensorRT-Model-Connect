# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.qwen.tests.runtime_receipt import assert_native_kv_receipt, assert_prompt_token_count


def _case() -> dict:
    return {
        "expected_kv_cache_rows": 256,
        "expected_prefill_chunks": 2,
        "expected_prefill_chunk_limit": 64,
        "max_new_tokens": 2,
    }


def _payload(stderr: str) -> dict:
    return {"runtime_stderr": stderr, "token_ids": [1, 2], "decode_ms": 1.0}


def test_receipt_requires_capacity_chunking_and_decode() -> None:
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=256)",
            "[trtmc.prefill] tokens=65 launches=2 max_chunk=64",
        ]
    )
    assert_native_kv_receipt(_payload(stderr), _case(), 65)


def test_receipt_rejects_missing_runtime_marker() -> None:
    with pytest.raises(AssertionError):
        assert_native_kv_receipt(
            _payload("[trtmc.prefill] tokens=65 launches=2 max_chunk=64"), _case(), 65
        )


def test_two_chunk_parity_does_not_claim_a_fixed_prompt_token_sum() -> None:
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=256)",
            "[trtmc.prefill] tokens=95 launches=2 max_chunk=64",
        ]
    )
    assert_native_kv_receipt(_payload(stderr), _case(), 96)


def test_long_regression_keeps_exact_prompt_and_observed_token_gates() -> None:
    case = {**_case(), "expected_prompt_tokens": 65}
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=256)",
            "[trtmc.prefill] tokens=64 launches=2 max_chunk=64",
        ]
    )
    with pytest.raises(AssertionError):
        assert_native_kv_receipt(_payload(stderr), case, 65)
    with pytest.raises(AssertionError):
        assert_native_kv_receipt(_payload(stderr), case, 64)


@pytest.mark.parametrize("tokens", [20, 24, 144])
def test_prompt_count_matches_without_kv_expectations(tokens: int) -> None:
    assert_prompt_token_count(
        _payload(f"[trtmc.prefill] tokens={tokens} launches=1 max_chunk={tokens}"), tokens
    )


def test_prompt_count_rejects_extra_thinking_prefix() -> None:
    with pytest.raises(AssertionError, match="native prompt has 24 tokens; reference has 20"):
        assert_prompt_token_count(
            _payload("[trtmc.prefill] tokens=24 launches=1 max_chunk=24"), 20
        )


def test_prompt_count_requires_native_receipt() -> None:
    with pytest.raises(AssertionError, match="did not report"):
        assert_prompt_token_count(_payload(""), 20)


@pytest.mark.parametrize("tagged", [False, True])
def test_prompt_count_checks_each_tensor_parallel_rank(tagged: bool) -> None:
    receipt = "[trtmc.prefill] tokens=20 launches=1 max_chunk=20"
    rank0 = "[1,0]<stderr>:" if tagged else ""
    rank1 = "[1,1]<stderr>:" if tagged else ""
    assert_prompt_token_count(_payload(f"{rank0}{receipt}\n{rank1}{receipt}"), 20)
    with pytest.raises(AssertionError, match="native prompt has 24 tokens"):
        assert_prompt_token_count(
            _payload(f"{rank0}{receipt}\n{rank1}[trtmc.prefill] tokens=24 launches=1 max_chunk=24"), 20
        )
