# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Case-specific behavior checks against independently parsed original output."""

import ast
import json
import re
import unicodedata

STARTS_IN_REASONING = False
CONTEXT = 128000
POLICY = "model-card-semantic-v1"
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
    value = normalized(text).replace("’", "'")
    return bool(
        re.search(
            r"\bi(?:'ve| have)? (?:already |successfully )?(?:set|changed|adjusted|updated) "
            r"(?:(?:the|a|your) )?(?:(?:cabin )?temperature|it)\b",
            value,
        )
        or re.search(
            r"\b(?:cabin )?temperature (?:has been |was |is now |is )?"
            r"(?:already |successfully )?(?:set|changed|adjusted|updated)\b",
            value,
        )
    )


def temperature_ack(text, temperature, *, short_ack=False):
    from decimal import Decimal

    value = normalized(text).translate(str.maketrans({sign: "-" for sign in "−‐‑‒–—﹣－"}))
    value = re.sub(r"(?<!\w)([+-])\s+(?=\d)", r"\1", value)
    value = re.sub(r"\b(?:minus|negative)\s*(?=\d)", "-", value)
    assert not refusal(value) and not re.search(
        r"\b(?:not|never|hasn't|haven't|isn't|wasn't|won't|don't)\b[^.!?]{0,25}\b(?:set|adjusted|changed|completed)\b",
        value,
    ), "temperature action was denied"
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
    stochastic = name == "model-card-sampling"
    assert len(actual_ids) <= limit and len(reference_ids) <= limit, "generation exceeded budget"
    if stochastic:
        for label, ids, ended in (
            ("reference", reference_ids, reference_eos),
            ("native", actual_ids, actual_eos),
        ):
            assert ended or len(ids) == limit, (
                label + " sampled generation stopped before EOS or budget"
            )
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
            "Native and Torch use independent random generators; equal seeds do not imply equal draws, tokens, or EOS/budget outcomes."
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
        assert re.fullmatch(r"(?:france is\s*)+", normalized(expected_text)), (
            "original repetitive behavior needs review"
        )
        assert normalized(actual_text) == normalized(expected_text), (
            "native repetitive behavior differs and needs review"
        )
        checks["quality_failure"] = (
            "The original repeats 'France is' rather than answering Paris; this is paired reference-behavior evidence, not successful factual completion."
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
    elif name in {"model-card-quickstart", "model-card-sampling"}:
        for label, text in (("reference", expected_text), ("native", actual_text)):
            text = normalized(text).replace("*", "")
            assert not english_fact_denial(
                text, r"caenorhabditis elegans|nematode|roundworm|model organism"
            ), label + " denies a required organism fact"
            assert not re.search(
                r"\b(?:not|never)\s+(?:used|employed|serving)\s+(?:in|for|as)\s+(?:biological\s+)?(?:research|biology|a model)",
                text,
            ), label + " denies research-model use"
            assert "caenorhabditis elegans" in text, label + " lacks organism identification"
            assert "roundworm" in text or "nematode" in text, (
                label + " lacks organism classification"
            )
            assert "model organism" in text and ("biolog" in text or "research" in text), (
                label + " lacks biological research-model use"
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
                assert not claims_temperature_action(text), (
                    "refusal contains a contradictory temperature action claim"
                )
        checks["tool_surface_scope"] = (
            "Compare parsed original call behavior; JSON/Pythonic surface spelling is diagnostic."
        )
    else:
        raise AssertionError("no reviewed semantic policy for case " + name)
    return checks
