# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("soundfile")

from families.canary.reference.adapter import decoded_observation  # noqa: E402


def test_decoded_work_uses_returned_tokens_including_special_tokens():
    value = SimpleNamespace(text="Hello.", y_sequence=np.array([17, 42, 3], dtype=np.int64))
    result = decoded_observation(value, 2.5)
    assert result == {"text": "Hello.", "input_audio_seconds": 2.5,
                      "token_ids": [17, 42, 3], "output_tokens": 3}


def test_empty_decode_has_zero_work_and_missing_tokens_are_not_inferred():
    assert decoded_observation({"text": "", "y_sequence": []}, 1.0)["output_tokens"] == 0
    assert decoded_observation("Two spoken words", 1.0) == {
        "text": "Two spoken words", "input_audio_seconds": 1.0}


@pytest.mark.parametrize("sequence", [[[0.1, 0.9]], [1.5], [True]])
def test_logits_or_invalid_tokens_cannot_be_reported_as_decode_work(sequence):
    with pytest.raises(ValueError, match="integer decoded sequence"):
        decoded_observation({"text": "hello", "y_sequence": sequence}, 1.0)
