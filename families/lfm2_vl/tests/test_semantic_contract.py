# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Positive paraphrases and adversarial controls for each semantic contract."""

import pytest

from .semantic_contract import EXPECTED_CALL, OCR_TEXT, compare_semantics


def compare(name, original, actual, **extra):
    return compare_semantics(name, {"text": original, **extra}, {"text": actual, **extra})


def test_caption_preserves_facts_without_matching_wording():
    reference = "The green Statue of Liberty stands on Liberty Island in New York Harbor. A boat is visible."
    native = "On Liberty Island in New York Harbor, the Statue of Liberty has a green exterior. The city skyline is behind it."
    assert compare("model-card", reference, native)["equivalent"]
    for wrong in [
        "A green tower.",
        native.replace("New York Harbor", "Paris"),
        "Not the Statue of Liberty on Liberty Island in New York Harbor. It is green.",
    ]:
        assert not compare("model-card", reference, wrong)["equivalent"]


@pytest.mark.parametrize(
    "denial",
    [
        "This does not depict the Statue of Liberty.",
        "This is not Lady Liberty.",
        "The statue is not located on Liberty Island.",
        "The statue isn't in New York Harbor.",
        "The statue is not in New York Bay.",
        "New York Harbor is not its location.",
        "The statue is not green.",
        "Green is not the statue's color.",
        "Liberty Island cannot be its location.",
        "New York Harbor can't be its location.",
    ],
)
def test_caption_rejects_direct_denial_even_after_positive_facts(denial):
    caption = "The green Statue of Liberty stands on Liberty Island in New York Harbor."
    assert not compare("model-card", caption, caption + " " + denial)["equivalent"]


def test_caption_ignores_unrelated_negation():
    caption = "The green Statue of Liberty stands on Liberty Island in New York Harbor."
    for detail in ["No boats are visible. ", "The island is not crowded. "]:
        assert compare("model-card", caption, detail + caption)["equivalent"]


def test_ocr_only_whitespace_is_permitted():
    assert compare(
        "image-budget", OCR_TEXT, "CABIN STATUS\n\nTemperature: 22 C\nSeat belts: FASTENED"
    )["equivalent"]
    for wrong in [
        "",
        OCR_TEXT.replace("22", "23"),
        OCR_TEXT.replace("FASTENED", "UNFASTENED"),
        "The cabin is comfortable.",
    ]:
        assert not compare("image-budget", OCR_TEXT, wrong)["equivalent"]


@pytest.mark.parametrize(
    "language,reference,paraphrase,wrong",
    [
        ("en", "The image is entirely red.", "Red.", "The image is blue."),
        ("ar", "اللون هو أحمر.", "الأحمر", "اللون أزرق"),
        ("zh", "红色", "这张图片是红色的。", "蓝色"),
        (
            "fr",
            "La couleur de cette image est le rouge.",
            "Cette image est rouge.",
            "Cette image est bleue.",
        ),
        ("de", "Rot.", "Das Bild ist rot.", "Das Bild ist blau."),
        ("ja", "この画像は単一の色、赤で構成されています。", "赤色です。", "青色です。"),
        ("ko", "이 이미지는 단색 빨간색입니다.", "빨간색입니다.", "파란색입니다."),
        ("es", "Rojo", "La imagen es roja.", "La imagen es azul."),
    ],
)
def test_color_language_and_meaning(language, reference, paraphrase, wrong):
    name = "language-" + language
    assert compare(name, reference, paraphrase)["equivalent"]
    assert not compare(name, reference, wrong)["equivalent"]
    assert not compare(name, reference, "Interesting image.")["equivalent"]


def test_color_negation_and_language_mismatch_fail():
    assert not compare("language-en", "Red", "It is not red.")["equivalent"]
    assert not compare("language-ar", "أحمر", "Red")["equivalent"]
    assert not compare("language-fr", "Rouge", "The image is rouge.")["equivalent"]
    assert not compare("aspect-ratio", "Green", "Red and green")["equivalent"]


@pytest.mark.parametrize(
    "answer",
    [
        "It is not red.",
        "It isn't red.",
        "Red is not its color.",
        "It is red. It is not red.",
        "Red can't be its color.",
        "Red cannot be its color.",
        "Red can’t be its color.",
        "It cannot be red.",
    ],
)
def test_color_rejects_direct_denial(answer):
    assert not compare("language-en", "Red", answer)["equivalent"]


def test_color_ignores_unrelated_negation():
    for answer in [
        "The image is red. It is not a square.",
        "No objects are visible. It is red.",
        "Not blue; it is red.",
        "It is red, not blue.",
        "It cannot be blue; it is red.",
        "Not white; it is red.",
    ]:
        assert compare("language-en", "Red", answer)["equivalent"]
    for answer in ["Maybe red.", "I cannot tell whether the image is red."]:
        assert not compare("language-en", "Red", answer)["equivalent"]
    assert not compare("language-en", "Red", "Red and black, not white.")["equivalent"]


def test_image_order_cannot_be_swapped():
    reference = "The first image is red and the second image is blue."
    assert compare("multi-image", reference, "First: red; second: blue.")["equivalent"]
    assert compare("multi-image", reference, "The second is blue. The first is red.")["equivalent"]
    assert not compare(
        "multi-image", reference, "The first image is blue and the second image is red."
    )["equivalent"]


def test_shared_reference_error_remains_visible():
    text = "The color changed from blue to red."
    result = compare("image-conversation", text, text)
    assert result["equivalent"]
    assert result["reference_task_correct"] is False
    assert result["native_task_correct"] is False
    assert not compare("image-conversation", text, "The color changed from red to blue.")[
        "equivalent"
    ]


def test_arithmetic_and_tool_result_require_values():
    assert compare("text-only", "3 + 5 = 8", "The answer is eight.")["equivalent"]
    assert not compare("text-only", "3 + 5 = 8", "3 + 5 = 9")["equivalent"]
    for wrong in ["-8", "−8", "- 8", "8.5"]:
        assert not compare("text-only", "3 + 5 = 8", wrong)["equivalent"]
    reference = "The current weather in Paris is sunny with a temperature of 20 degrees Celsius."
    assert compare("tool-result", reference, "Paris is sunny at 68 degrees Fahrenheit.")[
        "equivalent"
    ]
    assert not compare("tool-result", reference, "Paris is sunny at 21 degrees Celsius.")[
        "equivalent"
    ]
    for wrong in [
        "-20 degrees Celsius",
        "−20 degrees Celsius",
        "- 20 degrees Celsius",
        "-68 degrees Fahrenheit",
    ]:
        assert not compare("tool-result", reference, f"Paris is sunny at {wrong}.")["equivalent"]


def test_typed_tools_require_name_arguments_state_and_ids():
    original = {"text": "Retrieving the weather.", "calls": [EXPECTED_CALL]}
    native = {"text": "", "calls": [{**EXPECTED_CALL, "call_id": "call_0"}]}
    assert compare_semantics("tool-call", original, native)["equivalent"]
    for change in [
        {"name": "get_time"},
        {"arguments": {"location": "Berlin"}},
        {"state": 2},
        {"call_id": ""},
    ]:
        wrong = {"text": "Retrieving the weather.", "calls": [{**native["calls"][0], **change}]}
        assert not compare_semantics("tool-call", original, wrong)["equivalent"]


def test_context_checks_capacity_and_reference_behavior():
    reference = {"text": "I'm sorry, but I'm not", "input_tokens": 32760, "tokens": [1] * 8}
    native = {"text": "Sorry, I cannot fulfill that request.", "tokens": [2] * 8}
    assert compare_semantics("full-context", reference, native)["equivalent"]
    assert not compare_semantics("full-context", reference, {**native, "tokens": [2] * 7})[
        "equivalent"
    ]
    assert not compare_semantics("full-context", reference, {**native, "text": "OK"})["equivalent"]
