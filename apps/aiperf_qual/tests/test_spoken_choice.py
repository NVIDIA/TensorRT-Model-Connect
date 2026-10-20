# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spoken answer scoring must not reward silence or failed extraction."""

import pytest

from trtmc_aiperf_qual.gold_metrics import spoken_choice

GOLD = {"answer": "C", "prompt": "Predators eat\nA. lions\nB. humans\nC. bunnies\nD. grass\nSelect A, B, C, or D."}


@pytest.mark.parametrize("text", ["Option C, bunnies.", "C.", "The answer is C.", "They eat bunnies."])
def test_recognizes_the_correct_spoken_answer(text):
    assert spoken_choice(GOLD, {"text": text}) == (True, "C")


@pytest.mark.parametrize("text", ["", "Thank you for listening.", "A predator needs food.",
                                  "Option A or option C.", "lions and bunnies"])
def test_missing_or_ambiguous_answers_are_never_guessed(text):
    assert spoken_choice(GOLD, {"text": text}) == (False, "")


def test_wrong_answers_remain_wrong():
    assert spoken_choice(GOLD, {"text": "Option D, grass."}) == (False, "D")


def test_an_explicit_answer_overrides_a_word_in_the_explanation():
    assert spoken_choice(GOLD, {"text": "Bunnies are animals. The answer is A. Okay."}) == (False, "A")


@pytest.mark.parametrize("text", ["I would say C, definitely.", "You would pick C. That's the answer.",
                                  "You would pick C. It could also be a human.", "Yes. The answer is C. Okay.",
                                  "C is the correct answer."])
def test_answer_cues_work_inside_a_spoken_reply(text):
    assert spoken_choice(GOLD, {"text": text}) == (True, "C")


@pytest.mark.parametrize("text", ["The answer is B. C is the correct answer.",
                                  "They don't eat bunnies.", "Not option C."])
def test_conflicting_answers_and_negated_mentions_are_not_correct(text):
    assert spoken_choice(GOLD, {"text": text}) == (False, "")


def test_question_and_answer_labels_are_not_required_by_transcription():
    # The scorer, rather than the ASR stage, owns access to gold labels.
    assert spoken_choice(GOLD, {"text": "bunnies", "audio_artifact": "output.wav"}) == (True, "C")
