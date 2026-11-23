# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from .test_e2e import _assert_correctness


@pytest.fixture
def tokenizer():
    words = [
        "<|start|>",
        "<|message|>",
        "<|eom|>",
        "assistant",
        "to=user",
        "to=self",
        "ice",
        "Gibbs",
    ]
    tokenizer = Tokenizer(WordLevel(dict(zip(words, range(len(words))))))
    tokenizer.add_special_tokens(words[:3])
    return tokenizer


@pytest.mark.parametrize(
    ("thinking", "ids", "error"),
    [
        (False, [6, 7], None),
        (True, [5, 1, 2, 0, 3, 4, 1, 6, 7], None),
        (False, [5, 1, 6, 7], "unexpected turn"),
        (True, [3, 4, 6, 7], "no generated public turn"),
        (True, [5, 1, 6, 7, 2, 0, 3, 4, 1, 7], "ice"),
        (True, [2, 0, 3, 5, 1, 6, 7], "not addressed to the user"),
        (False, [6, 7] * 8, "token budget"),
    ],
)
def test_public_answer(tokenizer, thinking, ids, error):
    payload = {"token_ids": ids, "text": tokenizer.decode(ids)}
    case = {
        "max_new_tokens": 12,
        "enable_thinking": thinking,
        "expected_terms": [["ice"], ["gibbs"]],
    }
    if error is None:
        _assert_correctness(payload, case, tokenizer)
    else:
        with pytest.raises(AssertionError, match=error):
            _assert_correctness(payload, case, tokenizer)
