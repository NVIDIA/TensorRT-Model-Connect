# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the framework-free SAM3 tokenizer asset contract."""

from __future__ import annotations

import json

import pytest

from tensorrt_model_connect.families.sam3.tokenizer_contract import (
    Sam3TokenizerContractError,
    validate_sam3_tokenizer_json,
)


def _payload(*, vocab: dict[str, object] | None = None, merges: list[object] | None = None) -> str:
    return json.dumps(
        {
            "model": {
                "type": "BPE",
                "vocab": vocab if vocab is not None else {"a": 0, "b": 1, "ab": 2},
                "merges": merges if merges is not None else ["a b"],
            }
        }
    )


def _payload_with_added_tokens(added_tokens: object) -> str:
    document = json.loads(_payload())
    document["added_tokens"] = added_tokens
    return json.dumps(document)


def test_sam3_tokenizer_contract_accepts_dense_bpe() -> None:
    assert validate_sam3_tokenizer_json(_payload(), expected_vocab_size=3) == (3, 1)
    assert validate_sam3_tokenizer_json(
        _payload(merges=[["a", "b"]]),
        expected_vocab_size=3,
    ) == (3, 1)
    assert validate_sam3_tokenizer_json(
        _payload_with_added_tokens([{"id": 2, "content": "ab"}]),
        expected_vocab_size=3,
    ) == (3, 1)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload().replace('"BPE"', '"bpe"'), "must contain a BPE model"),
        (_payload(vocab={}), "non-empty BPE vocab"),
        (_payload(vocab={"a": 0, "b": 2}), "unique and dense"),
        (_payload(vocab={"a": 0, "b": 0}), "unique and dense"),
        (_payload(vocab={"a": 0, "b": True}), "must be integers"),
        (_payload(merges=[]), "non-empty BPE merges"),
        (_payload(merges=["not-a-pair"]), "invalid BPE merge"),
        (_payload(merges=[["a", 1]]), "invalid BPE merge"),
        (
            _payload(vocab={"a": 0, "b": 1, "c": 2}, merges=["a b"]),
            "operands and result must exist",
        ),
        (_payload_with_added_tokens({}), "added_tokens must be a list"),
        (
            _payload_with_added_tokens([{"id": 2, "content": "b"}]),
            "same in-vocab integer ID",
        ),
    ],
)
def test_sam3_tokenizer_contract_rejects_invalid_bpe(payload: str, message: str) -> None:
    with pytest.raises(Sam3TokenizerContractError, match=message):
        validate_sam3_tokenizer_json(payload)


def test_sam3_tokenizer_contract_rejects_text_encoder_size_mismatch() -> None:
    with pytest.raises(Sam3TokenizerContractError, match="expected 4, found 3"):
        validate_sam3_tokenizer_json(_payload(), expected_vocab_size=4)


def test_sam3_tokenizer_contract_rejects_non_boolean_added_token_special_flag() -> None:
    payload = _payload_with_added_tokens([{"id": 2, "content": "ab", "special": 1}])

    with pytest.raises(Sam3TokenizerContractError, match="special flag must be a boolean"):
        validate_sam3_tokenizer_json(payload)


def test_sam3_tokenizer_contract_rejects_duplicate_json_keys() -> None:
    payload = _payload().replace('"type": "BPE"', '"type": "BPE", "type": "BPE"')

    with pytest.raises(Sam3TokenizerContractError, match="duplicate object key 'type'"):
        validate_sam3_tokenizer_json(payload)
