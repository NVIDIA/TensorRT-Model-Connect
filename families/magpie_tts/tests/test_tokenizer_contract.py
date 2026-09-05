# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from families.magpie_tts.magpie_tokenizer import _vocab_size


def test_tokenizer_vocab_size_comes_from_the_authoritative_mapping() -> None:
    assert _vocab_size(SimpleNamespace(_id2token={0: "a", 1: "b"})) == 2

    with pytest.raises(ValueError, match="_id2token"):
        _vocab_size(SimpleNamespace())
