# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Case-specific behavior checks against independently parsed original output."""

import ast
from fractions import Fraction
import json
import re
import unicodedata

STARTS_IN_REASONING = True
CONTEXT = 131072
POLICY = "model-card-semantic-v2"
LANGUAGE_ANCHORS = {
    "en": ("paris", "france", "capital"),
    "ar": ("باريس", "فرنسا", "عاصم"),
    "zh": ("巴黎", "法国", "首都"),
    "fr": ("paris", "france", "capitale"),
    "de": ("paris", "frankreich", "hauptstadt"),
    "it": ("parigi", "francia", "capitale"),
    "ja": ("パリ", "フランス", "首都"),
    "ko": ("파리", "프랑스", "수도"),
    "pt": ("paris", "frança", "capital"),
    "es": ("parís", "francia", "capital"),
    "vi": ("paris", "pháp", "thủ đô"),
    "th": ("ปารีส", "ฝรั่งเศส", "เมืองหลวง"),
    "id": ("paris", "prancis", "ibu kota"),
    "hi": ("पेरिस", "फ्रांस", "राजधानी"),
    "ru": ("париж", "франци", "столиц"),
    "pl": ("paryż", "francj", "stolic"),
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def public_text_form(text):
    """Normalize presentation whitespace without changing names or quantities."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def english_fact_denial(text, facts):
    denial = r"(?:\b(?:not|never|no longer)\b|\b(?:is|are|was|were|do|does|did|can|could|should|would|has|have)n['’]t\b)"
    qualifiers = r"(?:a|an|the|any|actually|really|ever|currently|considered|regarded|known|used|be|been|being|as|for|in|to|one|of|biological|research|common|widely|france['’]s)"
    return bool(
        re.search(
            denial + r"(?:\s+" + qualifiers + r"){0,8}\s+(?:" + facts + r")\b", normalized(text)
        )
    )


def capital_denied(text, language):
    value = normalized(text)
    if language == "en":
        return english_fact_denial(value, r"capital|paris")
    patterns = {
        "ar": r"(?:ليست|ليس|غير)\s+(?:هي\s+)?(?:عاصمة|باريس)",
        "zh": r"(?:不是|并非|並非|不算)\s*(?:法国的?|法國的?)?(?:首都|巴黎)",
        "fr": r"\b(?:pas|jamais)\s+(?:(?:la|une|vraiment)\s+)*(?:capitale|paris)\b",
        "de": r"\b(?:nicht|keine)\s+(?:(?:die|eine|wirklich)\s+)*(?:hauptstadt|paris)\b",
        "it": r"\bnon\s+(?:(?:è|la|una|davvero)\s+)*(?:capitale|parigi)\b",
        "ja": r"(?:首都|パリ)[^。.!?]{0,16}(?:ではない|ではありません|じゃない)",
        "ko": r"(?:파리|수도)[^.!?]{0,20}(?:아니|아닙|않)",
        "pt": r"\bn[aã]o\s+(?:(?:é|a|uma|realmente)\s+)*(?:capital|paris)\b",
        "es": r"\bno\s+(?:(?:es|la|una|realmente)\s+)*(?:capital|parís)\b",
        "vi": r"không\s+(?:(?:phải|là)\s+)*(?:thủ đô|paris)",
        "th": r"ไม่(?:ใช่)?\s*(?:เมืองหลวง|ปารีส)",
        "id": r"\b(?:bukan|tidak)\s+(?:merupakan\s+)?(?:ibu kota|paris)\b",
        "hi": r"(?:राजधानी|पेरिस)[^।.!?]{0,25}नहीं",
        "ru": r"\bне\s+(?:является\s+)?(?:столиц\w*|париж\w*)\b",
        "pl": r"\bnie\s+(?:jest\s+)?(?:stolic\w*|paryż\w*)\b",
    }
    return bool(re.search(patterns[language], value))


def parse_calls(text, complete=True):
    """Interpret JSON or literal Pythonic calls without evaluating executable code."""
    if not complete:
        return [{"type": "tool_call", "name": "", "arguments_json": text, "state": 2}]
    try:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            expression = ast.parse(text.strip(), mode="eval").body
            assert isinstance(expression, ast.List) and expression.elts
            value = []
            for call in expression.elts:
                assert isinstance(call, ast.Call) and not call.args
                assert isinstance(call.func, (ast.Name, ast.Attribute))
                name = ast.unparse(call.func)
                assert re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name)
                arguments = {}
                for keyword in call.keywords:
                    assert keyword.arg is not None and keyword.arg not in arguments
                    arguments[keyword.arg] = ast.literal_eval(keyword.value)
                value.append({"name": name, "arguments": arguments})
        values = value if isinstance(value, list) else [value]
        assert values
        result = []
        for item in values:
            function = item.get("function", item)
            name, arguments = function["name"], function["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            assert isinstance(name, str) and name and isinstance(arguments, dict)
            result.append(
                {
                    "type": "tool_call",
                    "name": name,
                    "arguments_json": json.dumps(arguments, allow_nan=False),
                    "state": 1,
                }
            )
        return result
    except (AssertionError, ValueError, SyntaxError, TypeError, KeyError, AttributeError):
        return [{"type": "tool_call", "name": "", "arguments_json": text, "state": 3}]


def parse_reference(decoded, *, reasoning_prefix=False, allow_json_tools=False):
    source = ("<think>" if reasoning_prefix else "") + decoded
    parts = []
    while source:
        if source.startswith("<think>"):
            body, end, source = source[7:].partition("</think>")
            parts.append({"type": "reasoning", "text": body})
            if not end:
                break
        elif source.startswith("<|tool_call_start|>"):
            body, end, source = source[19:].partition("<|tool_call_end|>")
            parts.extend(parse_calls(body, bool(end)))
            if not end:
                break
        else:
            boundaries = [source.find(marker) for marker in ("<think>", "<|tool_call_start|>")]
            boundary = min((index for index in boundaries if index >= 0), default=len(source))
            text, source = source[:boundary], source[boundary:]
            try:
                value = json.loads(text)
                first = value[0] if isinstance(value, list) and value else value
                is_call = isinstance(first, dict) and (
                    "function" in first or {"name", "arguments"} <= first.keys()
                )
            except (ValueError, IndexError):
                is_call = False
            if allow_json_tools and is_call:
                parts.extend(parse_calls(text))
            elif text:
                parts.append({"type": "text", "text": text})
    return parts


def reference_parts(tokenizer, ids, request):
    if "messages" not in request:
        return [{"type": "text", "text": tokenizer.decode(ids, skip_special_tokens=True)}]
    pieces, pending = [], []
    markers = {"<think>", "</think>", "<|tool_call_start|>", "<|tool_call_end|>"}
    for token in ids:
        spelling = tokenizer.convert_ids_to_tokens(token)
        if spelling in markers:
            pieces.append(
                tokenizer.decode(
                    pending, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            )
            pending.clear()
            pieces.append(spelling)
        else:
            pending.append(token)
    pieces.append(
        tokenizer.decode(pending, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    )
    return parse_reference(
        "".join(pieces),
        reasoning_prefix=STARTS_IN_REASONING,
        allow_json_tools=bool(request.get("tools")),
    )


def tool_signature(parts):
    calls = []
    for part in parts:
        if part["type"] != "tool_call":
            continue
        state = part["state"]
        arguments = part["arguments_json"]
        if state == 1:
            arguments = json.loads(arguments)
            assert isinstance(arguments, dict), "complete tool arguments must be an object"
        calls.append({"name": part["name"], "arguments": arguments, "state": state})
    return calls


def typed_arguments(value):
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, dict):
        return (
            "object",
            tuple(sorted((key, typed_arguments(item)) for key, item in value.items())),
        )
    if isinstance(value, list):
        return ("array", tuple(typed_arguments(item) for item in value))
    return (type(value).__name__, value)


def call_meaning(calls):
    return [(call["name"], typed_arguments(call["arguments"]), call["state"]) for call in calls]


def final_text(parts):
    return "".join(part["text"] for part in parts if part["type"] == "text").strip()


def refusal(text):
    value = normalized(text).replace("’", "'")
    return bool(
        re.search(
            r"\bi(?:'m| am)? (?:cannot|can't|(?:unable|not able) to) "
            r"(?:directly )?(?:change|set|adjust|control) "
            r"(?:(?:the|a|your) )?(?:cabin )?temperature\b",
            value,
        )
        or re.search(
            r"\bi (?:don't|do not) have access to (?:a|the|any) (?:tool|system|control) "
            r"that (?:can )?(?:sets?|changes?|adjusts?|controls?) "
            r"(?:the )?(?:cabin )?temperature\b",
            value,
        )
    )


def claims_temperature_action(text):
    """Recognize bounded completed-action claims, excluding conditional advice."""
    value = normalized(text).replace("’", "'")
    patterns = (
        r"\b(?P<claim>i(?:'ve| have)? (?:already |successfully )?(?:set|changed|adjusted|updated) "
        r"(?:(?:the|a|your) )?(?:(?:cabin )?temperature|it)\b)",
        r"(?:^|[.!?;,]|\bbut\b)\s*(?P<claim>(?:(?:the|a|your) )?(?:cabin )?temperature "
        r"(?:has been |was |is now |is )?(?:already |successfully )?"
        r"(?:set|changed|adjusted|updated)\b)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, value):
            start = max(value.rfind(mark, 0, match.start("claim")) for mark in ".!?;") + 1
            if not re.search(r"\b(?:if|unless)\b", value[start : match.start("claim")]):
                return True
    return False


def denies_temperature_action(text):
    """Bind explicit action denials to temperature or its local result pronoun."""
    value = normalized(text).replace("’", "'")
    negative = (
        r"(?:not|never|cannot|can't|hasn't|haven't|isn't|wasn't|weren't|won't|don't|doesn't|didn't)"
    )
    temperature = (
        r"(?:(?:the|a|your) )?(?:(?:requested|desired|target) )?(?:cabin )?temperature"
        r"(?: (?:change|adjustment|setting|update))?"
    )
    cabin = r"(?:(?:the|a|your) )?cabin"
    subject = rf"(?:{temperature}|it|{cabin}(?= (?:has|have|had|is|was|were|will|{negative})\b))"
    target = rf"(?:{temperature}|it|{cabin}(?= (?:to|at)\b|[.!?;,]|$))"
    # Preserve bounded modifiers while stopping at a separate coordinated clause.
    span = r"(?:(?!\b(?:and|but|or|while|whereas)\b)[^.!?;,]){0,25}"
    action = r"(?:set|adjusted|changed|updated|completed|done)"
    return bool(
        re.search(
            rf"\b{subject}\b{span}\b{negative}\b{span}\b{action}\b",
            value,
        )
        or re.search(
            rf"\b{negative}\b{span}\b{action} {target}\b",
            value,
        )
        or re.search(
            rf"(?:^|[.!?;])\s*{negative}\b{span}\b{action}(?: yet)?\s*(?:[.!?;]|$)",
            value,
        )
    )


def temperature_ack(text, temperature, *, short_ack=False):
    from decimal import Decimal

    value = normalized(text).translate(str.maketrans({sign: "-" for sign in "−‐‑‒–—﹣－"}))
    value = re.sub(r"(?<!\w)([+-])\s+(?=\d)", r"\1", value)
    value = re.sub(r"\b(?:minus|negative)\s*(?=\d)", "-", value)
    assert not refusal(value) and not denies_temperature_action(value), (
        "temperature action was denied"
    )
    assert re.search(r"\b(set|adjusted|changed|done|completed)\b", value), (
        "missing action acknowledgement"
    )
    quantities = list(re.finditer(r"(?<![\w.+-])([+-]?\d+(?:\.\d+)?)", value))
    assert not re.search(
        r"\b(?:minus|negative)\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\b",
        value,
    ), "negative word temperature conflicts with requested setting"
    if short_ack and not quantities:
        return
    assert "temperature" in value or "cabin" in value, "missing temperature subject"
    assert quantities, "missing requested temperature"
    expected = Decimal(str(temperature))
    for quantity in quantities:
        number = Decimal(quantity[1])
        unit = re.match(
            r"\s*(?:degrees?\s*|°\s*)?(celsius|fahrenheit|kelvin|c|f|k)\b", value[quantity.end() :]
        )
        if unit and unit[1] in {"f", "fahrenheit"}:
            number = (number - Decimal(32)) * Decimal(5) / Decimal(9)
        elif unit:
            assert unit[1] in {"c", "celsius"}, "unsupported temperature unit"
        assert number == expected, "conflicting temperature value or unit"


def literal_function_intents(text):
    decoder, intents = json.JSONDecoder(), []
    offset = 0
    while offset < len(text):
        index = text.find("{", offset)
        if index < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except ValueError:
            offset = index + 1
            continue
        if isinstance(value, dict) and {"name", "arguments"} <= value.keys():
            intents.append({"name": value["name"], "arguments": value["arguments"]})
        offset = index + consumed
    return intents


def organism_concepts(text, label):
    text = normalized(text).replace("*", "")
    assert not english_fact_denial(
        text, r"caenorhabditis elegans|nematode|roundworm|model organism"
    ), label + " denies a required organism fact"
    assert not re.search(
        r"\b(?:not|never)\s+(?:used|employed|serving)\s+(?:in|for|as)\s+(?:biological\s+)?(?:research|biology|a model)",
        text,
    ), label + " denies research-model use"
    assert "caenorhabditis elegans" in text, label + " lacks organism identification"
    assert "roundworm" in text or "nematode" in text, label + " lacks organism classification"
    assert "model organism" in text and ("biolog" in text or "research" in text), (
        label + " lacks biological research-model use"
    )


# These four English claims are scoped to the model-card organism answer.
# Gene counts exclude cells/base pairs; development excludes lifespan and
# generation turnover. Missing additional facts are not contradictions.
_CLAIM_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:thousand|million|k))?"
_CLAIM_APPROX = r"(?:(?:approximately|approx\.?|about|around|roughly|~)\s*)?"
_CLAIM_DENIAL = r"(?P<denial>(?:(?:does\s+not|doesn['’]t)\s+(?:have|contain)|not)\s+)?"
_CLAIM_DURATION = (
    rf"{_CLAIM_DENIAL}{_CLAIM_APPROX}(?P<low>{_CLAIM_NUMBER})"
    rf"(?:\s*(?:-|to)\s*(?P<high>{_CLAIM_NUMBER}))?"
    r"\s*(?P<unit>hours?|days?|weeks?)\b"
)


def _claim_number(value):
    value = value.replace(",", "").replace(" ", "")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(thousand|million|k)?", value)
    return Fraction(match[1]) * {None: 1, "k": 1000, "thousand": 1000, "million": 1000000}[match[2]]


def _claim_negated(match):
    return bool(
        match.groupdict().get("denial")
        or re.search(
            r"(?:\bnot|n['’]t)\s+(?:fully\s+)?(?:sequenced|published|completed|released|take)\b"
            r"|\bnot\s+(?:in\s+)?\(?\s*\d",
            match[0],
        )
    )


def _other_organism_claim(match):
    """Exclude explicit other-organism subjects, not generic comparisons to them.

    This bounded grammar recognizes named subject/predicate forms and attached
    ``in humans`` qualifiers. It does not attempt general coreference resolution.
    """
    other = r"(?:humans?|homo sapiens|mice|mouse|yeast|drosophila)"
    prefix = match.string[: match.start()]
    suffix = match.string[match.end() :]
    subject = (
        rf"\b{other}(?:['’]s|['’])?\s+"
        r"(?:(?:genomes?|genome sequence)\s+)?"
        r"(?:(?:(?:do(?:es)?\s+not|doesn['’]t|don['’]t)\s+)?"
        r"(?:has|have|contains?|carry|carries|uses?)\s+)?$"
    )
    return bool(
        re.search(subject, prefix)
        or re.match(rf"genomes?\s+(?:of|in)\s+(?:the\s+)?{other}\b", match[0])
        or re.match(rf"\s+(?:in|of)\s+(?:the\s+)?{other}\b", suffix)
    )


def organism_reproduction_claims(text):
    """Read explicit reproduction modes and direct denials, without truth claims.

    Hermaphroditism, self-fertilization, eggs, and missing modes imply nothing.
    Only the bounded reproduction phrases below supply claims.
    """
    value = normalized(text).replace("*", "")
    value = re.sub(r"\bnot\s+(?:only|just|merely)\b", "", value)
    denial = r"(?P<denial>(?:(?:do(?:es)?|did)\s+not|doesn['’]t|don['’]t|didn['’]t|cannot|can['’]t|never|not)\s+)?"
    mode = r"(?:asexually|sexually|asexual|sexual|parthenogenesis)"
    modes = rf"(?P<modes>{mode}(?:\s*(?:,?\s+(?:and|or|but also)\s+){mode})*)"
    restriction = r"(?P<exclusive>only\s+|exclusively\s+)?"
    patterns = (
        rf"\b{denial}reproduc(?:e|es|ing)\s+{restriction}(?:both\s+)?(?:by\s+|via\s+|through\s+)?{modes}\b",
        rf"\breproduc(?:e|es|ing)\s+(?P<denial>not\s+){restriction}(?:by\s+|via\s+|through\s+)?{modes}\b",
        rf"\b(?P<denial>no\s+|not\s+|without\s+|(?:does not|doesn['’]t)\s+(?:use|undergo)\s+)?{restriction}{modes}\s+reproduction\b",
        rf"\breproduction\s+(?:is|occurs|happens|takes place)\s+{denial}{restriction}(?:both\s+)?(?:by\s+|via\s+|through\s+)?{modes}\b",
    )
    claims = set()
    for pattern in patterns:
        for match in re.finditer(pattern, value):
            if _other_organism_claim(match):
                continue
            identified = {
                "asexual" if item.startswith("asexual") or item == "parthenogenesis" else "sexual"
                for item in re.findall(mode, match["modes"])
            }
            denied = bool(match["denial"])
            claims.update((item, denied) for item in identified)
            if match["exclusive"] and not denied:
                claims.update((item, True) for item in {"sexual", "asexual"} - identified)
    return claims


def compare_organism_reproduction_claims(reference, native):
    """Reject opposing explicit modes or denials; a missing mode is not a claim."""
    expected = organism_reproduction_claims(reference)
    actual = organism_reproduction_claims(native)
    if expected and actual:
        for left, right in ((expected, actual), (actual, expected)):
            affirmed = {mode for mode, denied in left if not denied}
            other_denied = {mode for mode, denied in right if denied}
            assert not affirmed & other_denied, (
                "native/reference explicit reproduction modes conflict by denial"
            )
        reference_modes = {mode for mode, denied in expected if not denied}
        native_modes = {mode for mode, denied in actual if not denied}
        if reference_modes and native_modes:
            assert reference_modes & native_modes, (
                "native/reference explicit reproduction modes conflict"
            )
    return {
        "shared_property": bool(expected and actual),
        "reference_claims": [list(claim) for claim in sorted(expected)],
        "native_claims": [list(claim) for claim in sorted(actual)],
        "scope": "Explicit reproduction modes only; missing additional modes are not contradictions.",
    }


def organism_numeric_claims(text):
    """Extract bounded, explicit final-answer claims, without judging their truth."""
    value = (
        normalized(text)
        .replace("*", "")
        .translate(str.maketrans({dash: "-" for dash in "‐‑‒–—﹣－"}))
    )
    facts = {
        "sequencing_year": set(),
        "gene_count": set(),
        "egg_to_adult_days": set(),
        "larval_stage_count": set(),
    }
    year_patterns = (
        r"\bgenomes?\b[^.!?]{0,100}?\bsequenced\s*(?:(?:in|not)\s*){0,2}\(?\s*((?:19|20)\d{2})\b",
        r"\bgenome\s+sequenc(?:e|ing)\b[^.!?]{0,60}?\b(?:published|completed|released)"
        r"\s+(?:(?:in|not)\s+){0,2}((?:19|20)\d{2})\b",
        r"\b((?:19|20)\d{2}),?\s+(?:its\s+|the\s+)?genome\s+(?:was\s+)?(?:not\s+)?(?:fully\s+)?sequenced\b",
    )
    for pattern in year_patterns:
        for match in re.finditer(pattern, value):
            if _other_organism_claim(match):
                continue
            year = Fraction(match[1])
            facts["sequencing_year"].add((year, year, _claim_negated(match)))
    gene_patterns = (
        rf"\b{_CLAIM_DENIAL}{_CLAIM_APPROX}(?P<number>{_CLAIM_NUMBER})\s+(?:protein[- ]coding\s+)?genes\b",
        rf"\bgenes\s*\(\s*(?:e\.g\.,?\s*)?{_CLAIM_DENIAL}{_CLAIM_APPROX}(?P<number>{_CLAIM_NUMBER})\s*\)",
    )
    for pattern in gene_patterns:
        for match in re.finditer(pattern, value):
            if _other_organism_claim(match):
                continue
            number = _claim_number(match["number"])
            facts["gene_count"].add((number, number, _claim_negated(match)))
    egg_to_adult = r"(?:from\s+)?egg[ -]+to[ -]+(?:an?\s+)?adult"
    duration_patterns = (
        rf"{_CLAIM_DURATION}\s+{egg_to_adult}\b",
        rf"\b{egg_to_adult}(?:\s+development)?\s+(?:in|takes|requires|does not take)\s+{_CLAIM_DURATION}",
    )
    for pattern in duration_patterns:
        for match in re.finditer(pattern, value):
            scale = {"hour": Fraction(1, 24), "day": 1, "week": 7}[match["unit"].rstrip("s")]
            low = _claim_number(match["low"]) * scale
            high = _claim_number(match["high"] or match["low"]) * scale
            assert low <= high, "organism duration has a reversed range"
            facts["egg_to_adult_days"].add((low, high, _claim_negated(match)))
    count_words = {
        word: Fraction(number)
        for number, word in enumerate(
            ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
        )
    }
    words = "|".join(count_words)
    numeric_count = rf"[+-]?{_CLAIM_NUMBER}"
    count = rf"{numeric_count}|(?:[a-z]+-)+(?:{words})|{words}"
    for match in re.finditer(
        rf"(?<![\w.,+-])(?P<count>{count})\s+(?:distinct\s+)?larval\s+stages?\b", value
    ):
        if _other_organism_claim(match):
            continue
        token = match["count"]
        if token in count_words:
            number = count_words[token]
        else:
            assert re.fullmatch(numeric_count, token), "unsupported compound larval-stage count"
            number = _claim_number(token.lstrip("+-")) * (-1 if token.startswith("-") else 1)
        denied = bool(
            re.search(
                r"(?:\bnot|\bnever|n['’]t)\s+(?:(?:have|contain|undergo|go(?:es)?\s+through)\s+)?(?:exactly\s+)?$",
                value[: match.start()],
            )
        )
        facts["larval_stage_count"].add((number, number, denied))
    return facts


def compare_organism_numeric_claims(reference, native):
    """Compare shared explicit properties; no implicit numeric tolerance.

    Scalar claims must be equal after unit/format normalization; explicit ranges
    must overlap. Approximation words alone do not invent a wider interval.
    A denied interval conflicts with an affirmative claim wholly inside it.
    """
    expected, actual = organism_numeric_claims(reference), organism_numeric_claims(native)
    shared = [name for name in expected if expected[name] and actual[name]]
    for name in shared:
        for left, right in ((expected[name], actual[name]), (actual[name], expected[name])):
            affirmed = [claim for claim in left if not claim[2]]
            other_affirmed = [claim for claim in right if not claim[2]]
            denied = [claim for claim in right if claim[2]]
            if affirmed and other_affirmed:
                assert all(
                    any(max(a[0], b[0]) <= min(a[1], b[1]) for b in other_affirmed)
                    for a in affirmed
                ), f"native/reference explicit {name} claims conflict"
            assert not any(b[0] <= a[0] <= a[1] <= b[1] for a in affirmed for b in denied), (
                f"native/reference explicit {name} claims conflict by denial"
            )
    return {
        "shared_properties": shared,
        "reference_claims": {
            key: [[str(a), str(b), denied] for a, b, denied in sorted(values)]
            for key, values in expected.items()
        },
        "native_claims": {
            key: [[str(a), str(b), denied] for a, b, denied in sorted(values)]
            for key, values in actual.items()
        },
    }


def evaluate(name, native, expected_parts, reference_ids, eos, request, input_count):
    """Return reviewable checks; token equality is deliberately not a gate."""
    actual_parts = native.get("parts", [{"type": "text", "text": native.get("text", "")}])
    expected_calls, actual_calls = tool_signature(expected_parts), tool_signature(actual_parts)
    assert call_meaning(actual_calls) == call_meaning(expected_calls), (
        "typed tool name/arguments/state differ from reference"
    )
    limit = request["config"]["max_new_tokens"]
    actual_ids = native["token_ids"]
    assert reference_ids and actual_ids, "paired generation must produce tokens"
    for label, ids in (("reference", reference_ids), ("native", actual_ids)):
        assert not any(token in eos for token in ids[:-1]), (
            label + " returned tokens after the first EOS"
        )
    reference_eos = reference_ids[-1] in eos
    actual_eos = actual_ids[-1] in eos
    stochastic = name in {"model-card-sampling", "sampled-answer-completion"}
    assert len(actual_ids) <= limit and len(reference_ids) <= limit, "generation exceeded budget"
    if stochastic:
        for label, ids, ended in (
            ("reference", reference_ids, reference_eos),
            ("native", actual_ids, actual_eos),
        ):
            assert ended or len(ids) == limit, (
                label + " sampled generation stopped before EOS or budget"
            )
        assert actual_eos == reference_eos, "native/reference EOS stop behavior differs"
    else:
        assert actual_eos == reference_eos, "native/reference EOS stop behavior differs"
        if not reference_eos:
            assert len(reference_ids) == len(actual_ids) == limit, (
                "budget stop truncated generation"
            )
    expected_finish = (
        (3 if actual_calls and any(call["state"] == 1 for call in actual_calls) else 1)
        if actual_eos
        else 2
    )
    if "parts" in native:
        assert native["finish_reason"] == expected_finish, (
            "typed finish reason differs from reference behavior"
        )
    if "input_tokens" in native:
        assert native["input_tokens"] == input_count, "native input usage differs from reference"
    if "output_tokens" in native:
        assert native["output_tokens"] == len(actual_ids), "native output usage is inconsistent"
    expected_text, actual_text = final_text(expected_parts), final_text(actual_parts)
    checks = {
        "policy": POLICY,
        "tool_calls": actual_calls,
        "reference_stop": "eos" if reference_eos else "budget",
        "native_stop": "eos" if actual_eos else "budget",
        "reference_final_text": expected_text,
        "native_final_text": actual_text,
    }
    if stochastic:
        checks["generation_config"] = request["config"]
        checks["rng_scope"] = (
            "Equal seeds alone do not guarantee equal generated text; probability distributions and sampling execution policy also affect tokens and EOS/budget outcomes."
        )
    if name == "single-token-continuation":
        assert input_count == 1 and request.get("prompt") == "Hello", "singleton input changed"
        checks["scope"] = (
            "Single-token continuation interface/cache contract; unconstrained text is diagnostic only."
        )
    elif name == "full-context":
        assert len(request["token_ids"]) == input_count and input_count + limit == CONTEXT, (
            "full context admission/capacity changed"
        )
        checks["scope"] = (
            "Full capacity/cache and stop contract; output text is diagnostic, not retrieval-quality proof."
        )
    elif name == "raw-capital-completion":
        for label, text in (("reference", expected_text), ("native", actual_text)):
            assert re.match(r"^\s*\*{0,2}paris\b", normalized(text)), (
                label + " does not complete the capital answer with Paris"
            )
        checks["scope"] = (
            "Paris answer clause and paired stop/interface contract; trailing continuation is retained as a quality diagnostic, not prose-equivalence proof."
        )
    elif name == "chat-prefix-interface":
        for label, parts in (("reference", expected_parts), ("native", actual_parts)):
            reasoning = " ".join(part["text"] for part in parts if part["type"] == "reasoning")
            assert re.search(r"c\.?\s*elegans|caenorhabditis", normalized(reasoning)), (
                label + " reasoning prefix lost the prompt topic"
            )
        checks["scope"] = (
            "Short chat prefix framing/topic and stop contract; not completed-answer proof."
        )
    elif name.startswith("language-"):
        anchors = LANGUAGE_ANCHORS[name.removeprefix("language-")]
        for label, text in (("reference", expected_text), ("native", actual_text)):
            assert not capital_denied(text, name.removeprefix("language-")), (
                label + " denies the capital fact"
            )
            assert all(anchor in normalized(text) for anchor in anchors), (
                label + " does not identify Paris in the requested language"
            )
        checks["required_language_anchors"] = anchors
    elif name == "model-card-sampling" and limit == 512 and not expected_text:
        assert not reference_eos and not actual_eos, (
            "paired reasoning-only card case must stop at budget"
        )
        assert len(reference_ids) == len(actual_ids) == 512, (
            "reasoning-only card case did not consume its budget"
        )
        for label, parts in (("reference", expected_parts), ("native", actual_parts)):
            assert parts and all(part["type"] == "reasoning" for part in parts), (
                label + " did not preserve the reasoning-only typed-part state"
            )
            reasoning = "\n".join(part["text"] for part in parts)
            organism_concepts(reasoning, label)
            checks[label + "_reasoning"] = reasoning
        checks["completion_state"] = "reasoning_only_at_budget"
        checks["scope"] = (
            "Paired topical reasoning-at-budget behavior; NOT completed-answer proof. Incidental reasoning claims are retained for quality review."
        )
    elif name in {"model-card-quickstart", "model-card-sampling", "sampled-answer-completion"}:
        for label, text in (("reference", expected_text), ("native", actual_text)):
            organism_concepts(text, label)
        checks["explicit_factual_claims"] = compare_organism_numeric_claims(
            expected_text, actual_text
        )
        checks["explicit_reproduction_claims"] = compare_organism_reproduction_claims(
            expected_text, actual_text
        )
        if name == "sampled-answer-completion":
            checks["completion_state"] = "final_answer_present"
            checks["scope"] = (
                "Sampled final-answer content at an additional 2048-token cap; not the original 512-token card budget."
            )
    elif name == "positive-temperature-tool":
        assert actual_calls == [
            {"name": "set_temperature", "arguments": {"temperature": 22}, "state": 1}
        ], "explicit positive tool case must call set_temperature(22)"
    elif name in {"model-card-tool-call", "model-card-tool-json"}:
        assert actual_calls == [
            {"name": "get_candidate_status", "arguments": {"candidate_id": "12345"}, "state": 1}
        ], "candidate tool intent differs"
    elif name == "model-card-tool-result":
        for label, text in (("reference", expected_text), ("native", actual_text)):
            text = normalized(text)
            assert all(
                value in text
                for value in ("12345", "interview scheduled", "clinical research associate")
            ), label + " lacks returned candidate facts"
            assert "2023-11-20" in text or ("november 20" in text and "2023" in text), (
                label + " lacks returned interview date"
            )
            candidate_ids = re.findall(r"candidate(?:\s+id)?\s*(?:[:#]\s*)?(\d+)", text)
            assert all(value == "12345" for value in candidate_ids), (
                label + " reports a conflicting candidate ID"
            )
            assert set(re.findall(r"(?<!\d)20\d{2}(?!\d)", text)) <= {"2023"}, (
                label + " reports a conflicting interview year"
            )
            assert not re.search(r"\b(?:not|never|no longer)\b.{0,30}interview scheduled", text), (
                label + " denies the returned status"
            )
    elif name in {"tool-result", "parallel-tools", "strip-prior-reasoning"}:
        for text in (expected_text, actual_text):
            temperature_ack(text, 23 if name == "strip-prior-reasoning" else 22)
    elif name == "preserve-prior-reasoning":
        if STARTS_IN_REASONING:
            for text in (expected_text, actual_text):
                temperature_ack(text, 23, short_ack=True)
        else:
            expected_intent = [{"name": "set_cabin_temperature", "arguments": {"temperature": 23}}]
            assert (
                literal_function_intents(expected_text)
                == literal_function_intents(actual_text)
                == expected_intent
            ), "literal function intent differs"
            assert not actual_calls, "literal reference intent must not become an extra typed call"
            checks["presentation_note"] = (
                "Literal function intent in text; tags and acknowledgement presentation may differ."
            )
    elif name in {"system-tools", "json-tool-request", "single-user"}:
        if not expected_calls:
            for text in (expected_text, actual_text):
                assert refusal(text) and "temperature" in normalized(text), (
                    "reference refusal/no-call behavior was not preserved"
                )
                assert not claims_temperature_action(text), "contradictory temperature action"
        checks["tool_surface_scope"] = (
            "Compare parsed original call behavior; JSON/Pythonic surface spelling is diagnostic."
        )
    else:
        raise AssertionError("no reviewed semantic policy for case " + name)
    return checks
