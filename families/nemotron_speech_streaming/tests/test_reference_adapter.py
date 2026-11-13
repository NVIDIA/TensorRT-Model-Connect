# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("soundfile")

from families.nemotron_speech_streaming.reference.adapter import (  # noqa: E402
    decoded_observation, without_language_tags,
)


def test_the_native_transcript_leaves_out_the_prompted_models_language_tags():
    text = "This was a formable array of advantages. <en-US> Slavery was playing with loaded dice. <en-US>"
    assert without_language_tags(text) == "This was a formable array of advantages. Slavery was playing with loaded dice."
    assert without_language_tags("Is it not Louise? <de-DE>") == "Is it not Louise?"
    assert without_language_tags("a <b> c") == "a <b> c"  # only language tags


def test_decode_work_preserves_tokens_removed_from_displayed_transcript():
    value = SimpleNamespace(text="Hello. <en-US>", y_sequence=np.array([17, 42, 3], dtype=np.int64))
    result = decoded_observation(value, 2.5)
    assert result == {"text": "Hello.", "input_audio_seconds": 2.5,
                      "token_ids": [17, 42, 3], "output_tokens": 3}


def test_empty_decode_has_zero_work_and_missing_tokens_are_not_inferred():
    assert decoded_observation({"text": "", "y_sequence": []}, 1.0)["output_tokens"] == 0
    assert decoded_observation("Two spoken words <en-US>", 1.0) == {
        "text": "Two spoken words", "input_audio_seconds": 1.0}


@pytest.mark.parametrize("sequence", [[[0.1, 0.9]], [1.5], [True]])
def test_logits_or_invalid_tokens_cannot_be_reported_as_decode_work(sequence):
    with pytest.raises(ValueError, match="integer decoded sequence"):
        decoded_observation({"text": "hello", "y_sequence": sequence}, 1.0)
