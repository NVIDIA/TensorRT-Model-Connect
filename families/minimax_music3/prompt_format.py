# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's text input contract.

None of this is in the checkpoint: the special-token frame, the two token ids
that classifier-free guidance needs, and the two normalisations applied to the
description and the lyrics all live in the reference implementation
(diffusers v0.40.0, ``modular_pipelines/minimax_music3/encoders.py``, Apache-2.0,
(c) The MiniMax Team and The HuggingFace Team). A runtime that assembles the
prompt differently tokenises to different ids and generates different audio,
so the contract is reproduced here rather than approximated.

The assembled prompt is::

    <|im_start|><|caption_start|>{caption}<|caption_end|>
    <|lyrics_start|>{lyrics}<|lyrics_end|><|im_end|><|audio_start|>

with no separators beyond the tags themselves.
"""

from __future__ import annotations

import re

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
CAPTION_START, CAPTION_END = "<|caption_start|>", "<|caption_end|>"
LYRICS_START, LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

#: Token ids the reference uses directly rather than resolving through the
#: tokenizer.
AUDIO_END_TOKEN_ID = 151670
AUDIO_CFG_TOKEN_ID = 151654
AUDIO_CODE_OFFSET = 151675

SEMANTIC_VOCAB_SIZE = 16384
MAX_PROMPT_TOKENS = 5000

#: Autoregressive sampling, from the reference.
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_SAMPLING_TOP_K = 50

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


class PromptTooLongError(ValueError):
    """Raised when the assembled prompt exceeds the checkpoint's token budget."""


def clean_caption(caption: str) -> str:
    """Normalise a music description to the form the checkpoint expects.

    Rewrites ``<|key value|>`` spans as ``key is value`` and strips the
    markdown the input contract accepts but the model was not trained on.
    """

    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)

    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())

    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """Normalise lyrics to the form the checkpoint expects.

    A line that begins with one or more structure tags keeps only those tags;
    any text sharing the line is dropped, which is the contract the model card
    warns about. Tags are lowercased, each is put on its own line, and the
    result is prefixed with ``[start]``.
    """

    kept: list[str] = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        kept.append(match.group(1).strip() if match else line)

    text = "\n".join(kept)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble_prompt(caption: str, lyrics: str) -> str:
    """Return the special-token prompt the language model is given."""

    return (
        f"{IM_START}{CAPTION_START}{clean_caption(caption)}{CAPTION_END}"
        f"{LYRICS_START}{normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


def check_prompt_length(token_count: int) -> None:
    """Raise when a tokenised prompt exceeds the checkpoint's budget."""

    if token_count > MAX_PROMPT_TOKENS:
        raise PromptTooLongError(
            f"The assembled prompt has {token_count} tokens; "
            f"the maximum is {MAX_PROMPT_TOKENS}"
        )


def unconditional_ids(input_ids: list[int]) -> list[int]:
    """Return the classifier-free counterpart of a tokenised prompt.

    Every token except the first and the two trailing structure tokens is
    replaced by the audio-CFG token, mirroring the reference's
    ``unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID``.
    """

    if len(input_ids) < 4:
        raise ValueError(
            f"a prompt needs at least 4 tokens to build its CFG counterpart, "
            f"got {len(input_ids)}"
        )
    head, tail = input_ids[:1], input_ids[-2:]
    body = [AUDIO_CFG_TOKEN_ID] * (len(input_ids) - 3)
    return head + body + tail
