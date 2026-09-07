# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from families.smollm3.build_routing import native_kv_build_capability
from families.smollm3.config import ModelConfig
from families.smollm3.tests.runtime_receipt import assert_native_kv_receipt


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
    config = ModelConfig.create_tiny(
        "smollm3",
        architectures=["SmolLM3ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=65536,
        hidden_act="silu",
    )
    config.raw["no_rope_layer_interval"] = 4

    capability = native_kv_build_capability(
        config,
        precision="bf16",
        max_cache_length=65536,
    )

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


def test_receipt_rejects_a_bundle_max_numeric_prefix() -> None:
    stderr = "\n".join(
        [
            "[trtmc] KV cache rows=256 (bundle max=2560)",
            "[trtmc.prefill] tokens=66 launches=2 max_chunk=64",
        ]
    )
    with pytest.raises(AssertionError):
        assert_native_kv_receipt(_payload(stderr), _case(), 65)
