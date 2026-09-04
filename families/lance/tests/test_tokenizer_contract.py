# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

from families.lance import model


def test_tokenizer_uses_the_nested_lance_decoder_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "llm_config.json"
    config_path.write_text('{"model_type":"qwen2_5_vl"}', encoding="utf-8")
    tokenizer_config = SimpleNamespace(model_type="qwen2_5_vl")
    calls = []

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("config", path, kwargs))
            return tokenizer_config

    class Tokenizer:
        @staticmethod
        def encode(text, add_special_tokens=True):
            return [1, 7, 2] if add_special_tokens else [7]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=AutoConfig, AutoTokenizer=AutoTokenizer),
    )

    assert model._tokenizer_runtime_contract(tmp_path) == {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": [1],
        "tokenizer_suffix_ids": [2],
    }
    assert calls[0] == ("config", str(config_path), {"trust_remote_code": True})
    assert calls[1][0:2] == ("tokenizer", str(tmp_path))
    assert calls[1][2]["config"] is tokenizer_config
