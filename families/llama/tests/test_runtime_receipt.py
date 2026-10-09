# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.llama.build_routing import native_kv_build_capability
from families.llama.config import ModelConfig
from families.llama.tests.runtime_receipt import assert_native_kv_receipt


def _case() -> dict:
    return {
        "expected_prompt_tokens": 65,
        "expected_runtime_prefill_tokens": 66,
        "expected_kv_cache_rows": 256,
        "expected_prefill_chunks": 2,
        "expected_prefill_chunk_limit": 64,
        "max_new_tokens": 2,
    }


def _payload(stderr: str) -> dict:
    return {"runtime_stderr": stderr, "token_ids": [1, 2], "decode_ms": 1.0}


def test_long_prefill_manifest_selects_the_native_bf16_route() -> None:
    manifest = json.loads(
        (
            Path(__file__).with_name("manifests")
            / "minitron-4b-width-regression-native-kv-chunked-prefill.json"
        ).read_text(encoding="utf-8")
    )
    config = ModelConfig.create_tiny(
        "llama",
        architectures=["LlamaForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=manifest["max_sequence_length"],
        hidden_act="silu",
    )

    capability = native_kv_build_capability(
        config,
        precision=manifest["precision"],
        max_cache_length=manifest["max_sequence_length"],
    )

    assert manifest["precision"] == "bf16"
    assert capability.eligible, capability.reason


def test_receipt_distinguishes_raw_prompt_from_runtime_bos() -> None:
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=256)",
            "[trtmc.prefill] tokens=66 launches=2 max_chunk=64",
        ]
    )
    assert_native_kv_receipt(_payload(stderr), _case(), 65)


def test_receipt_rejects_missing_decode_step() -> None:
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=256)",
            "[trtmc.prefill] tokens=66 launches=2 max_chunk=64",
        ]
    )
    payload = _payload(stderr)
    payload["decode_ms"] = 0.0
    with pytest.raises(AssertionError):
        assert_native_kv_receipt(payload, _case(), 65)


@pytest.mark.parametrize("launches, max_chunk", [(1, 16), (2, 8)])
def test_prefill_count_includes_bos_without_requiring_a_kv_only_case(launches, max_chunk):
    from families.llama.tests.runtime_receipt import assert_prefill_token_count

    stderr = f"[trtmc.prefill] tokens=16 launches={launches} max_chunk={max_chunk}"
    assert_prefill_token_count(stderr, 16)


@pytest.mark.parametrize(
    "stderr",
    [
        "",
        "[trtmc.prefill] tokens=18 launches=1 max_chunk=18",
        "[trtmc.prefill] tokens=8 launches=1 max_chunk=8\n"
        "[trtmc.prefill] tokens=8 launches=1 max_chunk=8",
    ],
)
def test_prefill_count_rejects_missing_wrong_or_repeated_receipts(stderr):
    from families.llama.tests.runtime_receipt import assert_prefill_token_count

    with pytest.raises(AssertionError):
        assert_prefill_token_count(stderr, 16)
