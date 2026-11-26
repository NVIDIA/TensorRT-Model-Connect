# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from families.bailing_hybrid.model import _validate_tokenizer


@pytest.fixture
def tokenizer():
    return {
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
        "decoder": {"type": "ByteLevel", "add_prefix_space": True},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True},
    }


def test_supported_pretokenizer_does_not_reject_postprocessor_prefix(tokenizer):
    _validate_tokenizer(tokenizer)


@pytest.mark.parametrize("option", ["add_prefix_space", "use_regex"])
@pytest.mark.parametrize("missing", [False, True])
def test_unsupported_or_missing_bytelevel_options_are_rejected(tokenizer, option, missing):
    document = copy.deepcopy(tokenizer)
    bytelevel = document["pre_tokenizer"]["pretokenizers"][1]
    if missing:
        bytelevel.pop(option)
    else:
        bytelevel[option] = True
    with pytest.raises(ValueError, match="pre-tokenizer options"):
        _validate_tokenizer(document)
