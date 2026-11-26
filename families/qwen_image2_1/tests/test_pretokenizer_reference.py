# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-tokenizer fixtures against the Hugging Face ``tokenizers`` library.

The native tokenizer is checked against ``data/pretokenizer_cases.json`` by the C++ test
``qwen_image2_1_pretokenizer``. These tests keep that file honest: its pieces must be what the
library produces today with the same tokenizer definition, and the definition must carry the
checkpoint's own normalizer and pre-tokenizer.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from families.qwen_image2_1.bundle_config import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    TOKENIZER_NORMALIZER,
    TOKENIZER_PRE_TOKENIZER,
)
from families.qwen_image2_1.tests import pretokenizer_fixture as fixture

tokenizers = pytest.importorskip("tokenizers")


def _cases() -> list[tuple[str, str, list[str]]]:
    document = json.loads(fixture.CASES_PATH.read_text(encoding="utf-8"))
    return [
        (group["group"], case["text"], case["pieces"])
        for group in document["groups"]
        for case in group["cases"]
    ]


CASES = _cases()


@pytest.fixture(scope="module")
def reference():
    return tokenizers.Tokenizer.from_file(str(fixture.TOKENIZER_PATH))


def test_boundary_tokenizer_file_is_the_documented_construction() -> None:
    stored = json.loads(fixture.TOKENIZER_PATH.read_text(encoding="utf-8"))
    assert stored == fixture.boundary_tokenizer()
    assert stored["normalizer"] == TOKENIZER_NORMALIZER
    assert stored["pre_tokenizer"] == TOKENIZER_PRE_TOKENIZER
    vocab = stored["model"]["vocab"]
    assert len(vocab) == 512 and sorted(vocab.values()) == list(range(512))
    assert stored["model"]["merges"] == []


def test_ids_of_the_boundary_tokenizer_spell_out_the_pieces(reference) -> None:
    ids = reference.encode("ab c", add_special_tokens=False).ids
    assert ids == [ord("a"), 256 + ord("b"), ord(" "), 256 + ord("c")]
    assert fixture.pieces(ids) == ["ab", " c"]
    with pytest.raises(ValueError, match="no end marker"):
        fixture.pieces([ord("a")])
    with pytest.raises(ValueError, match="inside a pre-token"):
        fixture.pieces([ord("a"), 512])


def test_case_file_covers_what_it_claims() -> None:
    assert len(CASES) >= 100
    texts = [text for _, text, _ in CASES]
    assert len(set(texts)) == len(texts), "a text appears twice"
    groups = {group for group, _, _ in CASES}
    assert {
        "an apostrophe before letters",
        "contractions in any letter case",
        "numbers of every kind",
        "letters of many scripts",
        "whitespace before letters, numbers and symbols",
        "line breaks",
    } <= groups
    # The frame the runtime wraps around every prompt is among the cases.
    assert any(PROMPT_PREFIX.split("\n", 2)[2] in text and text.endswith(PROMPT_SUFFIX) for text in texts)


@pytest.mark.parametrize(("group", "text", "expected"), CASES, ids=[text for _, text, _ in CASES])
def test_case_pieces_are_the_reference_pieces(reference, group: str, text: str, expected: list[str]) -> None:
    # The checkpoint normalises to NFC before splitting and the runtime does not; the cases stay
    # with text that NFC leaves alone, so that they pin the splitting and nothing else.
    assert unicodedata.normalize("NFC", text) == text, group
    assert "".join(expected) == text
    actual = fixture.pieces(reference.encode(text, add_special_tokens=False).ids)
    assert actual == expected
