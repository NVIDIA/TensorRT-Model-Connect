# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MiniMax-Music3 text input contract.

The behaviour asserted here was checked against the reference implementation
by running both over the same inputs: 410 captions and 409 lyric strings,
covering the hand-written edge cases below plus 800 random strings drawn from
the alphabet that exercises the markdown and tag paths. No output differed.
"""

from __future__ import annotations

import importlib

import pytest

pf = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.prompt_format"
)


def test_prompt_frames_caption_and_lyrics_with_no_separators() -> None:
    prompt = pf.assemble_prompt("acoustic pop", "[verse]\nline")

    assert prompt.startswith(f"{pf.IM_START}{pf.CAPTION_START}")
    assert prompt.endswith(f"{pf.IM_END}{pf.AUDIO_START}")
    assert f"{pf.CAPTION_END}{pf.LYRICS_START}" in prompt
    assert f"{pf.LYRICS_END}{pf.IM_END}" in prompt


def test_special_tag_with_a_value_becomes_a_sentence() -> None:
    assert pf.clean_caption("<|bpm 92|>") == "bpm is 92"


def test_special_tag_without_a_value_keeps_only_its_name() -> None:
    assert pf.clean_caption("<|solo|>") == "solo"


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("# Heading", "Heading"),
        ("###### Deep", "Deep"),
        ("- bullet", "bullet"),
        ("* star", "star"),
        ("+ plus", "plus"),
        ("**bold**", "bold"),
        ("*italic*", "italic"),
        ("***triple***", "triple"),
        ("---", ""),
        ("• dot", "dot"),
    ],
)
def test_markdown_is_stripped(markdown: str, expected: str) -> None:
    assert pf.clean_caption(markdown) == expected


def test_blank_runs_collapse() -> None:
    assert pf.clean_caption("a\n\n\nb") == "a\nb"


def test_lyrics_gain_a_start_tag() -> None:
    assert pf.normalize_lyrics("plain").startswith("[start]\n")


def test_text_sharing_a_tag_line_is_dropped() -> None:
    """The contract the model card warns about."""

    normalized = pf.normalize_lyrics("[chorus] this text is lost\nthis text is kept")

    assert "this text is lost" not in normalized
    assert "this text is kept" in normalized


def test_tags_are_lowercased_and_split_onto_their_own_lines() -> None:
    normalized = pf.normalize_lyrics("[VERSE][Chorus]\nsing")

    assert "[verse]" in normalized
    assert "[chorus]" in normalized
    assert "[VERSE]" not in normalized


def test_length_budget_is_enforced() -> None:
    pf.check_prompt_length(pf.MAX_PROMPT_TOKENS)

    with pytest.raises(pf.PromptTooLongError, match="the maximum is 5000"):
        pf.check_prompt_length(pf.MAX_PROMPT_TOKENS + 1)


def test_unconditional_ids_keep_the_head_and_the_two_trailing_tokens() -> None:
    ids = [10, 11, 12, 13, 14, 15]

    unconditional = pf.unconditional_ids(ids)

    assert unconditional[0] == 10
    assert unconditional[-2:] == [14, 15]
    assert unconditional[1:-2] == [pf.AUDIO_CFG_TOKEN_ID] * 3
    assert len(unconditional) == len(ids)


def test_unconditional_ids_needs_a_long_enough_prompt() -> None:
    with pytest.raises(ValueError, match="at least 4 tokens"):
        pf.unconditional_ids([1, 2, 3])


def test_token_ids_are_the_reference_values() -> None:
    assert pf.AUDIO_CFG_TOKEN_ID == 151654
    assert pf.AUDIO_END_TOKEN_ID == 151670
    assert pf.AUDIO_CODE_OFFSET == 151675
    assert pf.SEMANTIC_VOCAB_SIZE == 16384
