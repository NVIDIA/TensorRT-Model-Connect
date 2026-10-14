# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-specific semantic contracts for the pinned model-card fixtures.

These checks extract concrete answers, not an embedding-similarity score. Unknown
or contradictory answers fail for review. Token/logit comparisons remain separate
diagnostics; the requested reference contract permits equivalent wording.
"""

from __future__ import annotations

import re
import unicodedata

OCR_TEXT = "CABIN STATUS Temperature: 22 C Seat belts: FASTENED"
EXPECTED_CALL = {"name": "get_weather", "arguments": {"location": "Paris"}, "state": 1}

COLOR_WORDS = {
    "en": {
        "red": r"\bred\b",
        "blue": r"\bblue\b",
        "green": r"\bgreen\b",
        "other": r"\b(?:white|black|yellow|purple|orange)\b",
    },
    "ar": {
        "red": r"(?<!\w)(?:ال)?(?:احمر|حمراء)(?!\w)",
        "blue": r"ازرق|زرقاء",
        "green": r"اخضر|خضراء",
        "other": r"ابيض|بيضاء|اسود|سوداء|اصفر|صفراء",
    },
    "zh": {
        "red": r"红色|紅色",
        "blue": r"蓝|藍",
        "green": r"绿|綠",
        "other": r"白色|黑色|黄色|黃色",
    },
    "fr": {
        "red": r"\brouge\b",
        "blue": r"\bbleu\w*\b",
        "green": r"\bvert\w*\b",
        "other": r"\b(?:blanc|noir|jaune)\w*\b",
    },
    "de": {
        "red": r"\brot(?:e|es|en|er)?\b",
        "blue": r"\bblau\w*\b",
        "green": r"\bgrün\w*\b",
        "other": r"\b(?:weiß|schwarz|gelb)\w*\b",
    },
    "ja": {
        "red": r"赤(?:色|い)?(?=で|。|\s|$)",
        "blue": r"青|ブルー",
        "green": r"緑|グリーン",
        "other": r"白色|黒色|黄色",
    },
    "ko": {
        "red": r"빨간색|빨강|붉은색|적색",
        "blue": r"파란색|파랑|청색",
        "green": r"녹색|초록",
        "other": r"흰색|백색|검정|검은색|노란색",
    },
    "es": {
        "red": r"\broj[oa]s?\b",
        "blue": r"\bazul\w*\b",
        "green": r"\bverde\w*\b",
        "other": r"\b(?:blanc|negr|amarill)[oa]s?\b",
    },
}
NEGATION = {
    "en": r"\b(?:not|no|never|unsure|maybe|cannot)\b|n't",
    "ar": r"(?<!\w)(?:ليس|ليست|غير|لا|ربما)(?!\w)",
    "zh": r"不是|并非|並非|不确定|不確定|可能",
    "fr": r"\b(?:pas|non|peut-être)\b",
    "de": r"\b(?:nicht|kein\w*|vielleicht)\b",
    "ja": r"ありません|ではない|わから|かもしれ",
    "ko": r"아니|않|모르",
    "es": r"\b(?:no|quizás|quiza|tal vez)\b",
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def english_fact_denied(text, fact):
    """Recognize direct denials of a fixture fact, not unrelated negated clauses."""
    links = (
        r"the|a|an|its|actually|really|even|entirely|very|longer|"
        r"on|in|at|from|be|been|located|situated|standing|"
        r"show|shows|shown|depict|depicts|depicted|feature|features|featured|"
        r"have|has|iconic"
    )
    before = rf"(?:\b(?:not|no|never|without)\b|n['’]t)\s+(?:(?:{links})\s+){{0,6}}(?:{fact})"
    after = (
        rf"(?:{fact})\s+(?:(?:is|was)\s+(?:not|never)|(?:is|was)n['’]t)\s+"
        r"(?:(?:the|its|a|correct|actual|statue['’]s|image['’]s)\s+){0,3}"
        r"(?:colou?r|location|setting|landmark|subject|where|shown|depicted|pictured|present)\b"
    )
    absent = rf"(?:{fact})\s+(?:is|was)\s+(?:absent|missing)\b"
    return any(re.search(pattern, text) for pattern in (before, after, absent))


def color_answer(text, language):
    value = normalized(text).casefold()
    if language == "ar":
        value = (
            re.sub(r"[\u064b-\u065f\u0670\u0640]", "", value).replace("أ", "ا").replace("إ", "ا")
        )
    if language == "en":
        denied = any(english_fact_denied(value, pattern) for pattern in COLOR_WORDS["en"].values())
        uncertain = re.search(
            r"(?:\b(?:unsure|maybe|cannot)\b|can't)[^.;!?]{0,60}\b(?:red|blue|green)\b", value
        )
        if denied or uncertain:
            return None
    elif re.search(NEGATION[language], value):
        return None
    found = {color for color, pattern in COLOR_WORDS[language].items() if re.search(pattern, value)}
    if len(found) != 1 or "other" in found:
        return None
    # A translated color word inside an English sentence does not demonstrate
    # answering in the requested language. Short color-only answers are valid.
    words = re.sub(r"[^\w\s]", "", value).strip()
    if language in {"ar", "zh", "ja", "ko"} and re.search(r"[a-z]", value):
        return None
    if language == "fr" and words != "rouge" and not re.search(r"couleur|\best\b|\bcette\b", value):
        return None
    if language == "de" and words != "rot" and not re.search(r"\b(?:bild|ist|farbe)\b", value):
        return None
    if (
        language == "es"
        and words not in {"rojo", "roja"}
        and not re.search(r"\b(?:imagen|color|es)\b", value)
    ):
        return None
    return next(iter(found))


def photo_facts(text):
    value = normalized(text).casefold()
    facts = {
        "landmark": r"\b(?:statue of liberty|lady liberty)\b",
        "island": r"\bliberty island\b",
        "location": r"\bnew york (?:harbou?r|bay)\b",
        "green_exterior": r"\bgreen(?:ish)?\b",
    }
    if any(english_fact_denied(value, pattern) for pattern in facts.values()):
        return None
    if re.search(r"eiffel|big ben|\b(?:paris|london|rome)\b", value):
        return None
    required = {name: bool(re.search(pattern, value)) for name, pattern in facts.items()}
    return required if all(required.values()) else None


def ordered_colors(text):
    value = normalized(text).casefold()
    if re.search(NEGATION["en"], value):
        return None
    labels = list(re.finditer(r"\b(first|second)\b", value))
    if len(labels) != 2 or {match[1] for match in labels} != {"first", "second"}:
        return None
    result = {}
    for index, match in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(value)
        colors = set(
            re.findall(
                r"\b(red|blue|green|white|black|yellow|orange|purple)\b", value[match.end() : end]
            )
        )
        if len(colors) != 1:
            return None
        result[match[1]] = next(iter(colors))
    return result


def transition(text):
    value = normalized(text).casefold()
    if re.search(NEGATION["en"], value):
        return None
    found = re.findall(r"\b(red|blue)\s+(?:to|into)\s+(red|blue)\b", value)
    return {"from": found[0][0], "to": found[0][1]} if len(found) == 1 else None


def arithmetic(text):
    value = normalized(text).casefold().replace("−", "-")
    if re.search(NEGATION["en"], value):
        return None
    value = re.sub(r"\beight\b", "8", value)
    value = re.sub(r"([+-])\s+(?=\d)", r"\1", value)
    numbers = [float(number) for number in re.findall(r"[+-]?\d+(?:\.\d+)?", value)]
    if not numbers or numbers[-1] != 8 or not set(numbers) <= {3, 5, 8}:
        return None
    if re.search(r"(?:3\s*[-*/]\s*5|5\s*[-*/]\s*3)", value):
        return None
    return 8


def weather(text):
    value = normalized(text).casefold().replace("−", "-")
    value = re.sub(r"([+-])\s+(?=\d)", r"\1", value)
    if re.search(NEGATION["en"], value):
        return None
    sunny = bool(re.search(r"\bsunny\b|\bsunshine\b", value))
    temperature = re.findall(
        r"(?<![\w.+-])([+-]?\d+(?:\.\d+)?)\s*(?:degrees?\s*|°\s*)?(celsius|c|fahrenheit|f)\b", value
    )
    if "paris" not in value or not sunny or len(temperature) != 1:
        return None
    number, unit = temperature[0]
    celsius = float(number) if unit in {"c", "celsius"} else (float(number) - 32) * 5 / 9
    return {"location": "Paris", "condition": "sunny", "celsius": celsius}


def refusal(text):
    value = normalized(text).casefold()
    return (
        "refusal"
        if re.search(r"\bi(?:'m| am) sorry\b|\bi (?:cannot|can't|am unable)\b", value)
        else None
    )


def compare_semantics(name, reference, native):
    """Return comparable task facts and shared-reference quality limitations."""
    ref_text, text = reference["text"], native["text"]
    expected = None
    if name in {"model-card", "splitting-disabled", "recommended-sampling"}:
        ref, actual = photo_facts(ref_text), photo_facts(text)
        expected = {"landmark": True, "island": True, "location": True, "green_exterior": True}
    elif name in {"ocr", "image-budget"}:
        ref, actual = normalized(ref_text) or None, normalized(text) or None
        expected = OCR_TEXT
    elif name == "multi-image":
        ref, actual = ordered_colors(ref_text), ordered_colors(text)
        expected = {"first": "red", "second": "blue"}
    elif name == "image-conversation":
        ref, actual = transition(ref_text), transition(text)
        expected = {"from": "red", "to": "blue"}
    elif name == "aspect-ratio" or name.startswith("language-"):
        language = "en" if name == "aspect-ratio" else name.removeprefix("language-")
        ref, actual = color_answer(ref_text, language), color_answer(text, language)
        expected = "green" if name == "aspect-ratio" else "red"
    elif name in {"text-only", "reasoning-history"}:
        ref, actual, expected = arithmetic(ref_text), arithmetic(text), 8
    elif name == "tool-call":
        keys = ("name", "arguments", "state")
        ref = [{key: call[key] for key in keys} for call in reference.get("calls", [])]
        calls = native.get("calls", [])
        actual = [{key: call[key] for key in keys} for call in calls]
        call_ids = [call.get("call_id") for call in calls]
        if not all(call_ids) or len(call_ids) != len(set(call_ids)):
            actual = None
        if not ref:
            ref = None
        if not actual:
            actual = None
        expected = [EXPECTED_CALL]
    elif name == "tool-result":
        ref, actual = weather(ref_text), weather(text)
        if reference.get("calls") or native.get("calls"):
            actual = None
        expected = {"location": "Paris", "condition": "sunny", "celsius": 20.0}
    elif name == "full-context":
        ref, actual = refusal(ref_text), refusal(text)
        # This case intentionally exercises the endpoint, not just a large
        # prefix. The original uses all eight continuation slots without EOS.
        if (
            reference.get("input_tokens") != 32760
            or len(reference["tokens"]) != 8
            or len(native["tokens"]) != 8
        ):
            actual = None
    else:
        raise ValueError(f"No semantic contract for {name}")
    equivalent = ref is not None and actual == ref
    # Reference parity and model answer quality are separate facts. For
    # example, a reference direction error must stay visible even when the
    # native model faithfully reproduces it; the port must not "repair" it.
    return {
        "equivalent": equivalent,
        "reference_facts": ref,
        "native_facts": actual,
        "expected_fixture_facts": expected,
        "reference_task_correct": None if expected is None else ref == expected,
        "native_task_correct": None if expected is None else actual == expected,
    }
