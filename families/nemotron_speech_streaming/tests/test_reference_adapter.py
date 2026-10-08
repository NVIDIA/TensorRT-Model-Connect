# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("soundfile")

from families.nemotron_speech_streaming.reference.adapter import without_language_tags  # noqa: E402


def test_the_native_transcript_leaves_out_the_prompted_models_language_tags():
    text = "This was a formable array of advantages. <en-US> Slavery was playing with loaded dice. <en-US>"
    assert without_language_tags(text) == "This was a formable array of advantages. Slavery was playing with loaded dice."
    assert without_language_tags("Is it not Louise? <de-DE>") == "Is it not Louise?"
    assert without_language_tags("a <b> c") == "a <b> c"  # only language tags
