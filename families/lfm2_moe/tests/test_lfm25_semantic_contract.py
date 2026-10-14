# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks for the explicitly approved behavior acceptance policy."""

import copy
import json

import pytest

from families.lfm2_moe.tests import lfm25_semantic_contract as semantic


def call(name="set_temperature", arguments=None, state=1, call_id="transport"):
    return {
        "type": "tool_call",
        "name": name,
        "arguments_json": json.dumps({"temperature": 22} if arguments is None else arguments),
        "state": state,
        "call_id": call_id,
    }


def text(value):
    return {"type": "text", "text": value}


def check(name, expected, actual, *, reference_ids=None, native_ids=None, limit=8, request=None):
    reference_ids = [1, 9] if reference_ids is None else reference_ids
    native_ids = [2, 3, 9] if native_ids is None else native_ids
    request = request or {
        "messages": [{"role": "user", "content": "fixture"}],
        "config": {"max_new_tokens": limit},
    }
    calls = any(part["type"] == "tool_call" and part["state"] == 1 for part in actual)
    native = {
        "parts": actual,
        "token_ids": native_ids,
        "finish_reason": (3 if calls else 1) if native_ids[-1] == 9 else 2,
        "input_tokens": 5,
        "output_tokens": len(native_ids),
    }
    return semantic.evaluate(name, native, expected, reference_ids, [9], request, 5)


def test_calls_compare_meaning_and_ignore_token_and_transport_identity():
    result = check("positive-temperature-tool", [call(call_id="hf")], [call(call_id="native")])
    assert result["tool_calls"][0]["arguments"] == {"temperature": 22}


@pytest.mark.parametrize(
    "bad",
    [
        call(arguments={"temperature": 23}),
        call(name="wrong"),
        call(state=2),
        call(arguments={"temperature": True}),
    ],
)
def test_wrong_tool_arguments_name_or_state_fail(bad):
    with pytest.raises(AssertionError, match="typed tool"):
        check("positive-temperature-tool", [call()], [bad])


def test_extra_tool_call_and_boolean_number_confusion_fail():
    with pytest.raises(AssertionError, match="typed tool"):
        check("positive-temperature-tool", [call()], [call(), call()])
    with pytest.raises(AssertionError, match="typed tool"):
        check("system-tools", [call(arguments={"value": 1})], [call(arguments={"value": True})])


def test_json_and_pythonic_reference_calls_have_same_meaning():
    pythonic = semantic.parse_reference(
        "<|tool_call_start|>[set_temperature(temperature=22)]<|tool_call_end|>"
    )
    serialized = semantic.parse_reference(
        '[{"name":"set_temperature","arguments":{"temperature":22}}]', allow_json_tools=True
    )
    assert semantic.call_meaning(semantic.tool_signature(pythonic)) == semantic.call_meaning(
        semantic.tool_signature(serialized)
    )
    check("json-tool-request", pythonic, [call()])


def test_reference_parser_never_executes_expressions_and_preserves_states():
    malformed = semantic.parse_calls('[unsafe(value=__import__("os"))]')
    assert malformed[0]["state"] == 3
    incomplete = semantic.parse_reference("<|tool_call_start|>[set_temperature(temperature=")
    assert incomplete[0]["state"] == 2
    assert semantic.parse_reference("<think>unfinished")[0] == {
        "type": "reasoning",
        "text": "unfinished",
    }


def test_weak_tool_fixture_preserves_reference_refusal_and_no_call():
    expected = [text("I don't have access to a tool that sets cabin temperature.")]
    check("system-tools", expected, [text("I am unable to change the cabin temperature directly.")])
    with pytest.raises(AssertionError):
        check("system-tools", expected, [call()])
    with pytest.raises(AssertionError):
        check("system-tools", expected, [text("The cabin temperature is set to 22.")])


@pytest.mark.parametrize(
    "response",
    [
        "I’m not able to directly change the temperature of a cabin for you.",
        "I'm not able to change the cabin temperature directly.",
        "I am not able to adjust cabin temperature.",
        "I’m unable to directly change the cabin temperature.",
    ],
)
def test_single_user_temperature_inability_paraphrases(response):
    expected = [text("I’m unable to change the cabin temperature directly.")]
    check("single-user", expected, [text(response)])
    with pytest.raises(AssertionError, match="temperature action was denied"):
        check("strip-prior-reasoning", [text("Cabin temperature set to 23°C.")], [text(response)])


@pytest.mark.parametrize(
    "response",
    [
        "I’m able to directly change the temperature of a cabin for you.",
        "I am not not able to change the cabin temperature.",
        "I’m not able to change the music. I set the cabin temperature to 22°C.",
        "The cabin temperature is not able to change by itself. I set it to 22°C.",
    ],
)
def test_temperature_refusal_does_not_accept_ability_or_unrelated_inability(response):
    expected = [text("I’m unable to change the cabin temperature directly.")]
    with pytest.raises(AssertionError, match="refusal/no-call"):
        check("single-user", expected, [text(response)])


def test_temperature_inability_text_does_not_hide_an_extra_typed_call():
    expected = [text("I’m unable to change the cabin temperature directly.")]
    response = text("I’m not able to directly change the temperature of a cabin for you.")
    with pytest.raises(AssertionError, match="typed tool"):
        check("single-user", expected, [response, call()])


@pytest.mark.parametrize(
    "response",
    [
        "I cannot change the music. I set cabin temperature to 22°C.",
        "I am unable to adjust the music. The cabin temperature is set to 22°C.",
    ],
)
def test_legacy_refusal_words_must_refer_to_temperature_capability(response):
    expected = [text("I am unable to change the cabin temperature directly.")]
    with pytest.raises(AssertionError, match="refusal/no-call"):
        check("single-user", expected, [text(response)])


@pytest.mark.parametrize(
    "claim",
    [
        "I have set the temperature to 22°C.",
        "The cabin temperature has been set to 22°C.",
        "I’ve adjusted it to 22°C.",
    ],
)
def test_refusal_cannot_be_followed_by_a_claim_of_completed_temperature_change(claim):
    expected = [text("I am unable to change the cabin temperature directly.")]
    response = text("I’m not able to directly change the cabin temperature. " + claim)
    with pytest.raises(AssertionError, match="contradictory temperature action"):
        check("single-user", expected, [response])


def test_original_weak_tool_capability_refusal_remains_accepted():
    response = text(
        "I don't have access to a tool that can set cabin temperature. "
        "The only available tool (get_weather_forecast) is for checking weather conditions, "
        "not controlling vehicle climate systems. Would you like me to check the weather forecast instead?"
    )
    check("system-tools", [response], [response])


def test_quickstart_matches_budget_stop_without_mandatory_eos():
    answer = text(
        "Caenorhabditis elegans is a nematode and a model organism in biological research."
    )
    check(
        "model-card-quickstart",
        [answer],
        [answer],
        reference_ids=[1] * 512,
        native_ids=[2] * 512,
        limit=512,
    )
    with pytest.raises(AssertionError, match="EOS stop"):
        check(
            "model-card-quickstart",
            [answer],
            [answer],
            reference_ids=[1] * 512,
            native_ids=[2, 9],
            limit=512,
        )
    with pytest.raises(AssertionError, match="truncated"):
        check(
            "model-card-quickstart",
            [answer],
            [answer],
            reference_ids=[1] * 512,
            native_ids=[2] * 511,
            limit=512,
        )
    with pytest.raises(AssertionError, match="identification"):
        check("model-card-quickstart", [answer], [text("Here is a nonempty response.")])


@pytest.mark.parametrize("bad", ["The capital of France is Lyon.", "法国的首都是巴黎。", "Paris."])
def test_language_answer_requires_city_and_requested_language(bad):
    with pytest.raises(AssertionError, match="requested language"):
        check("language-en", [text("The capital of France is Paris.")], [text(bad)])


def test_temperature_result_rejects_wrong_value_denial_and_extra_call():
    expected = [text("The cabin temperature has been set to 22 degrees Celsius.")]
    check("tool-result", expected, [text("Cabin temperature successfully adjusted to 22°C.")])
    for actual in (
        [text("Cabin temperature set to 23°C.")],
        [text("Cabin temperature has not been set to 22°C.")],
        [call()],
    ):
        with pytest.raises(AssertionError):
            check("tool-result", expected, actual)


def test_candidate_result_requires_returned_entities_and_date():
    expected = [
        text(
            "Candidate ID 12345: Interview Scheduled, Clinical Research Associate, November 20, 2023."
        )
    ]
    check("model-card-tool-result", expected, expected)
    for value in (
        "12345: Interview Scheduled.",
        "Candidate ID 12345: Interview Scheduled, Clinical Research Associate, November 20, 2024.",
    ):
        with pytest.raises(AssertionError):
            check("model-card-tool-result", expected, [text(value)])


def test_full_context_is_capacity_contract_not_retrieval_waiver(monkeypatch):
    monkeypatch.setattr(semantic, "CONTEXT", 24)
    request = {"token_ids": list(range(16)), "config": {"max_new_tokens": 8}}
    native = {"text": "background", "token_ids": [2] * 8}
    result = semantic.evaluate(
        "full-context", native, [text("more background")], [1] * 8, [9], request, 16
    )
    assert "not retrieval-quality" in result["scope"]
    bad = copy.deepcopy(request)
    bad["token_ids"].pop()
    with pytest.raises(AssertionError, match="admission"):
        semantic.evaluate("full-context", native, [text("background")], [1] * 8, [9], bad, 16)


def test_reviewed_raw_completion_is_case_specific():
    if semantic.STARTS_IN_REASONING:
        expected, actual = " Paris. (A statement of fact)", " Paris. (A) (B)"
    else:
        expected = actual = " France is France is France is"
    result = check(
        "raw-capital-completion",
        [text(expected)],
        [text(actual)],
        reference_ids=[1] * 8,
        native_ids=[2] * 8,
    )
    assert "scope" in result or "quality_failure" in result
    with pytest.raises(AssertionError):
        check(
            "raw-capital-completion",
            [text(expected)],
            [text("Something nonempty")],
            reference_ids=[1] * 8,
            native_ids=[2] * 8,
        )


def test_short_chat_prefix_is_not_completed_answer_proof():
    reasoning = {"type": "reasoning", "text": "The user asks what C. elegans is."}
    result = check(
        "chat-prefix-interface",
        [reasoning],
        [reasoning],
        reference_ids=[1] * 64,
        native_ids=[2] * 64,
        limit=64,
    )
    assert "not completed-answer" in result["scope"]
    with pytest.raises(AssertionError, match="prompt topic"):
        check(
            "chat-prefix-interface",
            [reasoning],
            [text("A nonempty final response")],
            reference_ids=[1] * 64,
            native_ids=[2] * 64,
            limit=64,
        )


def test_preserved_reasoning_follows_reviewed_original_final_behavior():
    if semantic.STARTS_IN_REASONING:
        check("preserve-prior-reasoning", [text("Done.")], [text("Completed.")])
    else:
        expected = [
            text(
                '<tool>{"name":"set_cabin_temperature","arguments":{"temperature":23}}</tool> Set to 23°C.'
            )
        ]
        actual = [text('{"name":"set_cabin_temperature","arguments":{"temperature":23.0}}')]
        result = check("preserve-prior-reasoning", expected, actual)
        assert "presentation_note" in result
        with pytest.raises(AssertionError):
            check(
                "preserve-prior-reasoning",
                expected,
                [call(name="set_cabin_temperature", arguments={"temperature": 23})],
            )


def test_sampling_checks_each_stop_independently_and_requires_final_answer():
    answer = text(
        "Caenorhabditis elegans is a roundworm and model organism for biological research."
    )
    # Independent RNGs may give the reference a budget stop and native an EOS,
    # or the reverse. Both stop reports and both final answers must be valid.
    for reference_ids, native_ids in (([1] * 8, [2, 9]), ([1, 9], [2] * 8)):
        result = check(
            "model-card-sampling",
            [answer],
            [answer],
            reference_ids=reference_ids,
            native_ids=native_ids,
        )
        assert result["reference_stop"] != result["native_stop"]
        assert "independent random" in result["rng_scope"]
    with pytest.raises(AssertionError, match="before EOS or budget"):
        check("model-card-sampling", [answer], [answer], reference_ids=[1] * 8, native_ids=[2] * 7)
    with pytest.raises(AssertionError, match="identification"):
        check(
            "model-card-sampling",
            [answer],
            [{"type": "reasoning", "text": answer["text"]}],
            reference_ids=[1] * 8,
            native_ids=[2] * 8,
        )


@pytest.mark.parametrize(
    "bad",
    [
        "Paris is not the capital of France.",
        "Paris isn't France's capital.",
        "The capital of France is not Paris.",
    ],
)
def test_direct_capital_denials_fail(bad):
    with pytest.raises(AssertionError, match="denies the capital"):
        check("language-en", [text("The capital of France is Paris.")], [text(bad)])


@pytest.mark.parametrize(
    "bad",
    [
        "Caenorhabditis elegans is not a nematode and is not a model organism for biological research.",
        "Caenorhabditis elegans is a roundworm but isn't a model organism in biological research.",
        "Caenorhabditis elegans is a nematode model organism but is not used in biological research.",
    ],
)
def test_direct_required_organism_denials_fail(bad):
    answer = text("Caenorhabditis elegans is a nematode model organism for biological research.")
    with pytest.raises(AssertionError, match="denies"):
        check("model-card-quickstart", [answer], [text(bad)])


def test_unrelated_negation_is_not_a_blanket_rejection():
    answer = text(
        "Caenorhabditis elegans is a nematode, not a plant. It is a model organism for biological research."
    )
    reasoning = {"type": "reasoning", "text": "Do not discuss unrelated topics."}
    check("model-card-quickstart", [reasoning, answer], [reasoning, answer])


@pytest.mark.parametrize(
    "bad",
    [
        "Cabin temperature set to −22°C.",
        "Cabin temperature set to - 22 degrees Celsius.",
        "Cabin temperature set to –22°C.",
        "Cabin temperature set to minus 22°C.",
        "Cabin temperature set to negative twenty-two degrees Celsius.",
        "Cabin temperature set to 22°F.",
        "Cabin temperature set to 22 kelvin.",
        "Cabin temperature set to 22°C and then 23°C.",
        "Cabin temperature hasn't been set to 22°C.",
    ],
)
def test_signed_wrong_unit_and_conflicting_temperature_facts_fail(bad):
    answer = text("The cabin temperature has been set to 22 degrees Celsius.")
    with pytest.raises(AssertionError):
        check("tool-result", [answer], [text(bad)])


def test_equivalent_temperature_units_preserve_the_same_value():
    answer = text("The cabin temperature has been set to 22 degrees Celsius.")
    check("tool-result", [answer], [text("Cabin temperature set to 71.6°F (22°C).")])


@pytest.mark.parametrize("side", ["native", "reference"])
def test_no_tokens_may_follow_first_eos(side):
    answer = text("The capital of France is Paris.")
    ids = {"native_ids": [2, 9], "reference_ids": [1, 9]}
    ids[side + "_ids"] = [9, 2, 9]
    with pytest.raises(AssertionError, match="after the first EOS"):
        check("language-en", [answer], [answer], **ids)
