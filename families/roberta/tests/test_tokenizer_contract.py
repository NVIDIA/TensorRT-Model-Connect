# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from families.roberta.model import _native_tokenizer_json


def test_legacy_camembert_schema_is_emitted_as_explicit_unigram(tmp_path) -> None:
    pre_tokenizer = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"unk_id": 0, "vocab": [["<unk>", 0.0]]},
                "pre_tokenizer": pre_tokenizer,
            }
        ),
        encoding="utf-8",
    )

    payload = json.loads(_native_tokenizer_json(tmp_path))

    assert payload["model"]["type"] == "Unigram"
    assert payload["pre_tokenizer"] == pre_tokenizer


def test_tokenizer_model_type_must_match_its_vocab_schema(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": [["<unk>", 0.0]]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        _native_tokenizer_json(tmp_path)
