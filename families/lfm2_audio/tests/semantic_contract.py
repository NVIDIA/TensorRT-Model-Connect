# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete content contracts for the pinned English model-card fixtures.

Autoregressive token/code identity is a diagnostic, not a speech-content metric.
Unknown content fails for review; there is no similarity or word-error threshold.
"""

from __future__ import annotations

import re
import unicodedata


def words(text):
    return re.findall(
        r"[+-]?\d+(?:[.,]\d+)*|\w+(?:'\w+)?",
        unicodedata.normalize("NFKC", text).casefold().replace("’", "'").replace("−", "-"),
    )


def spoken_content(intended, transcript, *, alternatives=False):
    """Require all spoken words, allowing spelling-only ASR homophones.

    An 'or' immediately before a quoted slogan is an optional list connective.
    No other deletion, insertion, name, number or negation is discarded.
    """
    aliases = {"you're": "your", "they're": "their"}
    expected = words(intended)
    actual = words(transcript)
    optional = set()
    if alternatives:
        value = unicodedata.normalize("NFKC", intended).casefold().replace("’", "'")
        for match in re.finditer(r'\bor\s*(?=["“])', value):
            optional.add(len(words(value[: match.start()])))
    states = {0}
    for index, token in enumerate(expected):
        next_states = set()
        for position in states:
            if index in optional:
                next_states.add(position)
            if position < len(actual) and aliases.get(token, token) == aliases.get(
                actual[position], actual[position]
            ):
                next_states.add(position + 1)
        states = next_states
    return {
        "equivalent": bool(expected) and len(actual) in states,
        "intended_words": expected,
        "transcript_words": actual,
        "optional_list_connective_indices": sorted(optional),
        "homophone_spelling_pairs": [["your", "you're"], ["their", "they're"]],
    }


def dialogue_content(text, turn):
    value = " ".join(words(text))
    if re.search(r"\b(?:not|no|never|cannot|can't|don't|won't)\b", value):
        return False
    if turn == 0:
        topic = re.search(
            r"\b(?:woodworking|wood|handcrafted|crafting|crafted|home|furniture)\b", value
        )
    elif turn == 1:
        topic = re.search(r"\b(?:chairs?|seats?|seating)\b", value)
    else:
        raise ValueError("the fixture has exactly two dialogue turns")
    proposal = re.search(r"\b(?:how about|slogans?|crafted|crafting|handcrafted)\b", value)
    quality = re.search(
        r"\b(?:dream|quality|crafting|crafted|handcrafted|timeless|elegant|comfort\w*|excellence|skill)\b",
        value,
    )
    return bool(topic and proposal and quality)


def red_light_content(text):
    value = " ".join(words(text))
    if re.search(
        r"\b(?:don't|doesn't|shouldn't|needn't|mustn't|can't|cannot|won't|never|not|avoid|skip|forgo|wrong)\b"
        r"(?: \w+){0,4} (?:stop|stopping)\b"
        r"|\b(?:stop|stopping) (?:is |is considered )?(?:optional|unnecessary|not required|not needed)\b"
        r"|\b(?:go|proceed|drive) (?:through|past)\b",
        value,
    ):
        return False
    return bool(re.search(r"\bred (?:traffic )?light\b", value) and re.search(r"\bstop\b", value))


def interleaved_schedule(trace):
    """Check the documented six-text/twelve-audio block order and stop frame."""
    tokens, codes, modalities = trace["tokens"], trace["codes"], trace["modalities"]
    if len(codes) != 8 or not codes[0] or any(len(book) != len(codes[0]) for book in codes):
        return False
    if modalities.count(1) != len(tokens) or modalities.count(3) != len(codes[0]):
        return False
    if any(book[-1] != 2048 for book in codes):
        return False
    text_index = audio_index = 0
    text_done = False
    expected, left = 1, 6
    for modality in modalities:
        if modality != expected:
            return False
        left -= 1
        if modality == 1:
            text_done |= tokens[text_index] == 130
            text_index += 1
            if left == 0 or text_done:
                expected, left = 3, 12
        else:
            terminal = codes[0][audio_index] == 2048
            if terminal and any(book[audio_index] != 2048 for book in codes):
                return False
            audio_index += 1
            if terminal or (left == 0 and not text_done):
                expected, left = 1, 6
    return text_done and modalities[-1] == 3
