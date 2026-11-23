# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned MiniCPM5-1B configuration and checkpoint-selection coverage."""

import json

import pytest
from pathlib import Path

from ..build_routing import native_kv_cache_geometry
from ..config import ModelConfig
from .test_e2e import _CASES, _prompt


def test_minicpm5_1b_checkpoint_is_selectable():
    manifest, case = _CASES["minicpm5-1b"]
    assert manifest["hf_id"] == "openbmb/MiniCPM5-1B"
    assert manifest["hf_revision"] == "87179e5c1f455ef22e6223592d2d61351b525bfc"
    assert manifest["family"] == "llama" and manifest["tensor_parallel_size"] == 1
    assert case["use_chat_template"] and not case["enable_thinking"]


def test_minicpm5_1b_preserves_explicit_attention_width_and_stop_tokens():
    raw = json.loads((Path(__file__).parent / "fixtures/minicpm5-1b-config.json").read_text())
    config = ModelConfig.from_json(json.dumps(raw))
    assert config.hidden_size == 1536
    assert config.head_dim == 128
    assert config.attention_size == 2048
    assert config.attention_size != config.hidden_size
    assert native_kv_cache_geometry(config, 131072) == (24576, 3221225472)
    assert config.eos_token_id == [1, 130073]
    assert config.raw["eos_token_id"] == [1, 130073]


def test_minicpm5_empty_case_preserves_reference_and_native_bos_checks():
    manifest, case = _CASES["minicpm5-1b-empty"]
    assert manifest["hf_revision"] == "87179e5c1f455ef22e6223592d2d61351b525bfc"
    assert case["prompt"] == "" and not case["use_chat_template"]
    assert case["expected_prompt_token_ids"] == [0]
    assert case["expected_runtime_prefill_tokens"] == 1
    assert case["reference_precision"] == "fp32"
    assert case.get("reference") != "none" and not case.get("kv_contract_only", False)


def test_minicpm5_empty_case_reaches_e2e_prompt_consumers():
    _, case = _CASES["minicpm5-1b-empty"]
    assert _prompt(case) == ""


@pytest.mark.parametrize("expected_ids", [None, []])
def test_e2e_still_rejects_unframed_empty_prompts(expected_ids):
    case = {"name": "empty-without-framing", "prompt": ""}
    if expected_ids is not None:
        case["expected_prompt_token_ids"] = expected_ids
    with pytest.raises(AssertionError):
        _prompt(case)
