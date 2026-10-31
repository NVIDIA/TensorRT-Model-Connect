# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Positive paraphrases and adversarial speech/content controls."""

import pytest

from families.lfm2_audio.tests.semantic_contract import (
    dialogue_content,
    interleaved_schedule,
    red_light_content,
    spoken_content,
)


@pytest.mark.parametrize(
    "actual", ["HELLO THERE! How can I help you today?", "Hello there. How can I help you today."]
)
def test_spoken_greeting(actual):
    assert spoken_content("Hello there. How can I help you today?", actual)["equivalent"]


@pytest.mark.parametrize(
    "actual",
    [
        "Hello there.",
        "How can I help you today?",
        "Hello there. How can I not help you today?",
        "Hello John. How can I help you today?",
        "Hello there. How can I help you tomorrow?",
    ],
)
def test_spoken_content_does_not_accept_partial_or_changed_words(actual):
    assert not spoken_content("Hello there. How can I help you today?", actual)["equivalent"]


def test_acoustic_spelling_and_list_connective():
    assert spoken_content("They're your trophies.", "Their you're trophies.")["equivalent"]
    assert spoken_content(
        'How about "Elegant Chairs" or "Timeless Seats"?',
        "How about elegant chairs, timeless seats?",
        alternatives=True,
    )["equivalent"]
    assert not spoken_content(
        "Bring red or blue chairs.", "Bring red blue chairs.", alternatives=True
    )["equivalent"]
    assert not spoken_content(
        '"Two Chairs" or "Three Seats"', "two chairs three seats", alternatives=False
    )["equivalent"]
    assert not spoken_content(
        '"Two Chairs" or "Three Seats"', "two chairs four seats", alternatives=True
    )["equivalent"]
    assert not spoken_content("Chairs for Alice", "Chairs for Bob")["equivalent"]


@pytest.mark.parametrize(
    "intended,transcript",
    [
        ("-20 degrees", "20 degrees"),
        ("-2.5 volts", "+2.5 volts"),
        ("−20 degrees", "20 degrees"),
        ("20.5 meters", "205 meters"),
    ],
)
def test_spoken_numbers_preserve_sign_and_decimal_value(intended, transcript):
    assert not spoken_content(intended, transcript)["equivalent"]


def test_equivalent_unicode_minus_preserves_negative_value():
    assert spoken_content("−20 degrees", "-20 degrees")["equivalent"]


@pytest.mark.parametrize(
    "text,turn",
    [
        ('How about "Handcrafted Woodworking Excellence"?', 0),
        ('How about "Elegant Chairs Made to Last"?', 1),
        ('"Crafted Seats, Timeless Design"', 1),
    ],
)
def test_dialogue_task_content(text, turn):
    assert dialogue_content(text, turn)


@pytest.mark.parametrize(
    "text,turn",
    [
        ('How about "Elegant Shoes Made to Last"?', 1),
        ('How about "Handcrafted Woodworking Excellence"?', 1),
        ("I cannot make slogans for chairs.", 1),
        ("No chairs: how about crafted tables?", 1),
        ("Here is a weather report.", 0),
        ('How about "Cheap Plastic Toys"?', 0),
    ],
)
def test_dialogue_rejects_wrong_topic_or_refusal(text, turn):
    assert not dialogue_content(text, turn)


@pytest.mark.parametrize(
    "text", ["A red traffic light means stop and wait.", "Stop when the red light is on."]
)
def test_red_signal_meaning(text):
    assert red_light_content(text)


@pytest.mark.parametrize(
    "text",
    [
        "A red traffic light means do not stop.",
        "A red light means drive through without stopping.",
        "Stop at a blue light.",
        "A red light means go.",
        "A red traffic light means you must avoid a stop.",
        "You shouldn't stop at a red traffic light.",
        "You needn't stop at a red traffic light.",
        "At a red traffic light, stopping is not required.",
        "At a red traffic light, it is wrong to stop.",
    ],
)
def test_red_signal_rejects_contradictions(text):
    assert not red_light_content(text)


def test_interleaved_order_and_terminal_frame():
    trace = {
        "tokens": [10] * 6 + [130],
        "codes": [[1] * 12 + [2048] for _ in range(8)],
        "modalities": [1] * 6 + [3] * 12 + [1, 3],
    }
    assert interleaved_schedule(trace)
    assert not interleaved_schedule(trace | {"modalities": [3] + trace["modalities"][1:]})
    assert not interleaved_schedule(trace | {"codes": [[1] * 13 for _ in range(8)]})
    assert not interleaved_schedule(trace | {"tokens": [10] * 7})
