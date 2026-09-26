# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned MiniCPM5-1B configuration and checkpoint-selection coverage."""

import json
from pathlib import Path

from ..build_routing import native_kv_cache_geometry
from ..config import ModelConfig
from .test_e2e import _CASES


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
