# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The merge-order fixture against the Hugging Face ``tokenizers`` library.

The native tokenizer is checked against ``data/merge_cases.json`` by the C++ test
``qwen_image2_1_tokenizer``. These tests keep that file honest: its ids must be what the library
produces today with ``data/merge_tokenizer.json``, and the cases must separate the reference
merge order from the one that merges every occurrence of the best pair per pass. Its
``space_marker_cases`` must be the library's ids with ``space_marker_tokenizer``.
"""

from __future__ import annotations

import json

import pytest

from families.qwen_image2_1.bundle_config import TOKENIZER_NORMALIZER, TOKENIZER_PRE_TOKENIZER
from families.qwen_image2_1.tests import pretokenizer_fixture as fixture

tokenizers = pytest.importorskip("tokenizers")

_DOCUMENT = json.loads(fixture.MERGE_CASES_PATH.read_text(encoding="utf-8"))
CASES = [(case["text"], case["ids"]) for case in _DOCUMENT["cases"]]
SPACE_MARKER_CASES = [(case["text"], case["ids"]) for case in _DOCUMENT["space_marker_cases"]]


@pytest.fixture(scope="module")
def reference():
    return tokenizers.Tokenizer.from_file(str(fixture.MERGE_TOKENIZER_PATH))


def _every_occurrence_per_pass(symbols: list[str], rank: dict[tuple[str, str], int]) -> list[str]:
    """The other merge order: the best pair, all its occurrences left to right, then again."""
    while len(symbols) > 1:
        ranked = [(rank[pair], pair) for pair in zip(symbols, symbols[1:]) if pair in rank]
        if not ranked:
            break
        first, second = min(ranked)[1]
        merged: list[str] = []
        index = 0
        while index < len(symbols):
            if symbols[index : index + 2] == [first, second]:
                merged.append(first + second)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        symbols = merged
    return symbols


def test_merge_tokenizer_file_is_the_documented_construction() -> None:
    stored = json.loads(fixture.MERGE_TOKENIZER_PATH.read_text(encoding="utf-8"))
    assert stored == fixture.merge_tokenizer()
    assert stored["normalizer"] == TOKENIZER_NORMALIZER
    assert stored["pre_tokenizer"] == TOKENIZER_PRE_TOKENIZER
    model = stored["model"]
    assert model["merges"] == [list(rule) for rule in fixture.MERGE_RULES]
    assert sorted(model["vocab"].values()) == list(range(len(model["vocab"])))
    # Every rule is one the reference accepts: parts and result have token ids.
    for first, second in fixture.MERGE_RULES:
        assert {first, second, first + second} <= set(model["vocab"])


def test_merge_table_has_the_properties_that_separate_the_orders() -> None:
    rules = list(fixture.MERGE_RULES)
    last_rank = {rule: index for index, rule in enumerate(rules)}
    producer = {first + second: rule for rule in reversed(rules) for first, second in [rule]}
    # A rule that uses a token ranks ahead of the rule that produces it.
    assert last_rank[("ab", "a")] < last_rank[producer["ab"]]
    # A repeated rule keeps its later rank, which puts the rule using its result ahead of it.
    assert rules.count(("c", "d")) == 2 and last_rank[("cd", "c")] < last_rank[("c", "d")]
    # Two rules have the same result.
    assert len({rule for rule in rules if "".join(rule) == "efg"}) == 2


def test_cases_separate_the_merge_orders() -> None:
    vocab = fixture.merge_tokenizer()["model"]["vocab"]
    rank = {rule: index for index, rule in enumerate(fixture.MERGE_RULES)}
    words = [(text, ids) for text, ids in CASES if text.isalpha()]
    assert len(words) >= 20
    other = {
        text: [vocab[piece] for piece in _every_occurrence_per_pass(list(text), rank)]
        for text, _ in words
    }
    assert other["ababa"] == [vocab["ab"], vocab["aba"]]
    assert dict(words)["ababa"] == [vocab["aba"], vocab["b"], vocab["a"]]
    assert sum(other[text] != ids for text, ids in words) >= 5


def test_case_file_covers_what_it_claims() -> None:
    texts = [text for text, _ in CASES]
    assert len(texts) >= 40 and len(set(texts)) == len(texts)
    assert any(" " in text for text in texts), "a piece that starts with a space"
    assert any("<|im_end|>" in text for text in texts), "added tokens around merged pieces"


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[text for text, _ in CASES])
def test_case_ids_are_the_reference_ids(reference, text: str, expected: list[int]) -> None:
    assert reference.encode(text, add_special_tokens=False).ids == expected


def test_space_marker_cases_cover_text_with_and_without_the_marker() -> None:
    definition = fixture.space_marker_tokenizer()
    marker = definition["added_tokens"][-1]
    assert marker["content"] == fixture.SPACE_MARKER == "\u2581"
    # The marker is a token of this tokenizer only as an added token, after all other ids.
    assert fixture.SPACE_MARKER not in definition["model"]["vocab"]
    assert marker["id"] == len(definition["model"]["vocab"]) + len(definition["added_tokens"]) - 1
    assert definition["pre_tokenizer"] == TOKENIZER_PRE_TOKENIZER
    with_marker = [ids for text, ids in SPACE_MARKER_CASES if fixture.SPACE_MARKER in text]
    without_marker = [ids for text, ids in SPACE_MARKER_CASES if fixture.SPACE_MARKER not in text]
    assert len(with_marker) >= 4 and len(without_marker) >= 4
    assert all(marker["id"] in ids for ids in with_marker)
    assert not any(marker["id"] in ids for ids in without_marker)
    # Text without the marker encodes as it does without the added token.
    plain = dict(CASES)
    shared = [(text, ids) for text, ids in SPACE_MARKER_CASES if text in plain]
    assert len(shared) >= 2 and all(plain[text] == ids for text, ids in shared)


@pytest.mark.parametrize(
    ("text", "expected"), SPACE_MARKER_CASES, ids=[text for text, _ in SPACE_MARKER_CASES]
)
def test_space_marker_case_ids_are_the_reference_ids(text: str, expected: list[int]) -> None:
    reference = tokenizers.Tokenizer.from_str(json.dumps(fixture.space_marker_tokenizer()))
    assert reference.encode(text, add_special_tokens=False).ids == expected
