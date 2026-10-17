# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks for the explicitly approved behavior acceptance policy."""

import copy
import json

import pytest

from families.lfm2.tests import lfm25_semantic_contract as semantic


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


@pytest.mark.parametrize("case", ["system-tools", "json-tool-request", "single-user"])
@pytest.mark.parametrize(
    "response",
    [
        "I cannot change the music. The cabin temperature is set to 22°C.",
        "I am unable to adjust the music. I have set the cabin temperature to 22°C.",
        "I am able to change the cabin temperature.",
    ],
)
def test_refusal_must_describe_temperature_capability(case, response):
    expected = [text("I cannot change the cabin temperature.")]
    with pytest.raises(AssertionError, match="refusal/no-call"):
        check(case, expected, [text(response)])


@pytest.mark.parametrize("case", ["system-tools", "json-tool-request", "single-user"])
@pytest.mark.parametrize(
    "response",
    [
        "I’m not able to directly change the temperature of a cabin for you.",
        "I am unable to adjust the cabin temperature directly.",
        "I don't have access to a tool that can set cabin temperature.",
    ],
)
def test_temperature_capability_refusals_remain_accepted(case, response):
    check(case, [text("I cannot change the cabin temperature.")], [text(response)])


@pytest.mark.parametrize("case", ["system-tools", "json-tool-request", "single-user"])
@pytest.mark.parametrize(
    "claim",
    [
        "I have set the cabin temperature to 22°C.",
        "I’ve adjusted it to 22°C.",
        "The cabin temperature has been set to 22°C.",
        "Cabin temperature set to 22°C.",
        "I have set the cabin temperature to 22°C, if that helps.",
    ],
)
def test_temperature_refusal_cannot_hide_a_completed_action(case, claim):
    refusal = "I cannot change the cabin temperature."
    for response in (refusal + " " + claim, claim + " " + refusal):
        with pytest.raises(AssertionError, match="contradictory temperature action"):
            check(case, [text(refusal)], [text(response)])


@pytest.mark.parametrize("case", ["system-tools", "json-tool-request", "single-user"])
@pytest.mark.parametrize(
    "advice",
    [
        "You can set the cabin temperature to 22°C using the controls.",
        "You can get the temperature set to 22°C using the controls.",
        "If I set the temperature to 22°C, it would be more comfortable.",
        "I have not set the cabin temperature to 22°C.",
    ],
)
def test_temperature_refusal_advice_is_not_a_completed_action(case, advice):
    refusal = "I cannot change the cabin temperature."
    check(case, [text(refusal)], [text(refusal + " " + advice)])


@pytest.mark.parametrize(
    "case,temperature", [("tool-result", 22), ("parallel-tools", 22), ("strip-prior-reasoning", 23)]
)
@pytest.mark.parametrize(
    "unrelated",
    [
        "I cannot change the music.",
        "I have not changed the music.",
        "The music hasn’t changed.",
        "I have not completed the music setup.",
        "I have not changed the cabin music.",
    ],
)
def test_temperature_acknowledgement_ignores_unrelated_refusal_and_denial(
    case, temperature, unrelated
):
    acknowledgement = f"The cabin temperature is set to {temperature}°C."
    for response in (unrelated + " " + acknowledgement, acknowledgement + " " + unrelated):
        check(case, [text(acknowledgement)], [text(response)])
        check(case, [text(response)], [text(acknowledgement)])


@pytest.mark.parametrize(
    "case,temperature", [("tool-result", 22), ("parallel-tools", 22), ("strip-prior-reasoning", 23)]
)
@pytest.mark.parametrize(
    "denial",
    [
        "Cabin temperature hasn't been set to {temperature}°C.",
        "The cabin temperature is {temperature}°C. It hasn't been set.",
        "Cabin temperature set to {temperature}°C. I have not set it.",
        "The cabin temperature is {temperature}°C. I have not set the cabin to {temperature}°C.",
        "Cabin temperature set to {temperature}°C. I can't set it.",
        "The cabin temperature is {temperature}°C. It has not really been set.",
        "The cabin temperature is {temperature}°C. It has not actually yet been set.",
        "The cabin temperature is {temperature}°C. I have not set the requested temperature.",
        "The cabin temperature is {temperature}°C. The temperature has still not been changed.",
        "I have not changed the cabin temperature to {temperature}°C. I have completed the music setup.",
    ],
)
def test_temperature_denial_is_not_hidden_by_an_unrelated_completed_action(
    case, temperature, denial
):
    expected = [text(f"The cabin temperature is set to {temperature}°C.")]
    with pytest.raises(AssertionError, match="temperature action was denied"):
        check(case, expected, [text(denial.format(temperature=temperature))])


def test_short_acknowledgement_scope_preserves_completion():
    check("preserve-prior-reasoning", [text("Done.")], [text("Completed.")])


@pytest.mark.parametrize(
    "denied",
    [
        "Not done.",
        "Not completed.",
        "It hasn't been set.",
        "Not set.",
        "Not adjusted.",
        "Not changed.",
        "Not set yet.",
    ],
)
def test_short_temperature_acknowledgement_denial_is_rejected_on_either_side(denied):
    for reference, native in (("Done.", denied), (denied, "Done.")):
        with pytest.raises(AssertionError, match="temperature action was denied"):
            check("preserve-prior-reasoning", [text(reference)], [text(native)])


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


@pytest.mark.parametrize(
    "case", ["tool-result", "parallel-tools", "strip-prior-reasoning", "preserve-prior-reasoning"]
)
@pytest.mark.parametrize(
    "denial",
    [
        "I didn't set the temperature.",
        "I didn’t set the temperature.",
        "the temperature doesn't get set.",
        "the temperature doesn’t get set.",
    ],
)
def test_contracted_temperature_denial_rejects_contradictory_acknowledgements(case, denial):
    temperature = 23 if case in {"strip-prior-reasoning", "preserve-prior-reasoning"} else 22
    expected = text(f"The cabin temperature is set to {temperature}°C.")
    response = f"I set the cabin temperature to {temperature}°C, but {denial}"
    with pytest.raises(AssertionError, match="temperature action was denied"):
        semantic.temperature_ack(response, temperature)
    with pytest.raises(AssertionError, match="temperature action was denied"):
        check(case, [expected], [text(response)])
    with pytest.raises(AssertionError, match="temperature action was denied"):
        check(case, [text(response)], [expected])


@pytest.mark.parametrize(
    "case", ["tool-result", "parallel-tools", "strip-prior-reasoning", "preserve-prior-reasoning"]
)
@pytest.mark.parametrize(
    "unrelated",
    [
        "I didn't set the music.",
        "I didn’t set the cabin music.",
        "the music doesn't get changed.",
        "the cabin music doesn’t get changed.",
    ],
)
def test_contracted_music_denial_preserves_temperature_acknowledgement(case, unrelated):
    temperature = 23 if case in {"strip-prior-reasoning", "preserve-prior-reasoning"} else 22
    expected = text(f"The cabin temperature is set to {temperature}°C.")
    response = f"I set the cabin temperature to {temperature}°C, but {unrelated}"
    semantic.temperature_ack(response, temperature)
    check(case, [expected], [text(response)])
    check(case, [text(response)], [expected])


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


@pytest.mark.parametrize("name", ["model-card-sampling", "sampled-answer-completion"])
def test_sampling_matches_reference_stop_and_requires_final_answer(name):
    answer = text(
        "Caenorhabditis elegans is a roundworm and model organism for biological research."
    )
    # Different token IDs and EOS lengths are allowed; the stop reason must match.
    for reference_ids, native_ids in (([1, 9], [2, 3, 9]), ([1] * 8, [2] * 8)):
        result = check(
            name,
            [answer],
            [answer],
            reference_ids=reference_ids,
            native_ids=native_ids,
        )
        assert result["reference_stop"] == result["native_stop"]
        assert "probability distributions" in result["rng_scope"]
    for reference_ids, native_ids in (([1] * 8, [2, 9]), ([1, 9], [2] * 8)):
        with pytest.raises(AssertionError, match="EOS stop behavior differs"):
            check(
                name,
                [answer],
                [answer],
                reference_ids=reference_ids,
                native_ids=native_ids,
            )
    with pytest.raises(AssertionError, match="before EOS or budget"):
        check(name, [answer], [answer], reference_ids=[1] * 8, native_ids=[2] * 7)
    with pytest.raises(AssertionError, match="identification"):
        check(
            name,
            [answer],
            [{"type": "reasoning", "text": answer["text"]}],
            reference_ids=[1] * 8,
            native_ids=[2] * 8,
        )


def test_dense_card_budget_matches_paired_reasoning_state_without_claiming_answer():
    reasoning = {
        "type": "reasoning",
        "text": "Caenorhabditis elegans is a nematode model organism used in biological research.",
    }
    result = check(
        "model-card-sampling",
        [reasoning],
        [reasoning],
        reference_ids=[1] * 512,
        native_ids=[2] * 512,
        limit=512,
    )
    assert result["completion_state"] == "reasoning_only_at_budget"
    assert "NOT completed-answer" in result["scope"]
    with pytest.raises(AssertionError, match="reasoning-only typed-part"):
        check(
            "model-card-sampling",
            [reasoning],
            [text(reasoning["text"])],
            reference_ids=[1] * 512,
            native_ids=[2] * 512,
            limit=512,
        )
    with pytest.raises(AssertionError, match="identification"):
        check(
            "model-card-sampling",
            [reasoning],
            [{"type": "reasoning", "text": "Unrelated thoughts."}],
            reference_ids=[1] * 512,
            native_ids=[2] * 512,
            limit=512,
        )
    with pytest.raises(AssertionError, match="identification"):
        check(
            "sampled-answer-completion",
            [reasoning],
            [reasoning],
            reference_ids=[1] * 2048,
            native_ids=[2] * 2048,
            limit=2048,
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


ORGANISM_CORE = "Caenorhabditis elegans is a roundworm and a model organism in biological research."
# Minimal final-answer excerpts from the public Dev full-card response pair.
PUBLIC_FACT_CONFLICTS = (
    (
        "sequencing_year",
        "One of the first multicellular genomes sequenced (2002), providing a reference for understanding gene function.",
        "One of the first nematode genomes sequenced (2003), with a compact, well-annotated genome (~100 million base pairs).",
    ),
    (
        "gene_count",
        "Approximately 1,000 genes, with many conserved pathways shared with humans.",
        "Fewer genes (e.g., ~20,000) compared to humans, yet highly conserved pathways with mammals.",
    ),
    (
        "egg_to_adult_days",
        "Approximately 2–3 weeks from egg to adult, enabling rapid generation turnover.",
        "Short lifespan (~2-3 weeks), rapid development (from egg to adult in ~3 days), and self-fertilizing hermaphrodite.",
    ),
)


@pytest.mark.parametrize(
    "case", ("model-card-quickstart", "model-card-sampling", "sampled-answer-completion")
)
@pytest.mark.parametrize("property_name,reference,native", PUBLIC_FACT_CONFLICTS)
def test_model_card_rejects_public_shared_fact_conflicts(case, property_name, reference, native):
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match=property_name + " claims conflict"):
            check(case, [text(ORGANISM_CORE + " " + left)], [text(ORGANISM_CORE + " " + right)])


@pytest.mark.parametrize(
    "reference,native,shared",
    [
        (
            "Its genome was fully sequenced in 2002.",
            "Its genome was sequenced (2002).",
            ["sequencing_year"],
        ),
        (
            "The complete genome sequence was published in 1998.",
            "Its genome was fully sequenced in 1998.",
            ["sequencing_year"],
        ),
        (
            "In 1998, its genome was fully sequenced.",
            "Its genome was sequenced in 1998.",
            ["sequencing_year"],
        ),
        ("Approximately 20,000 genes.", "About 20 thousand genes.", ["gene_count"]),
        ("Genes (e.g., ~20,000).", "20k genes.", ["gene_count"]),
        (
            "~3 days from egg to adult.",
            "Egg-to-adult development takes 72 hours.",
            ["egg_to_adult_days"],
        ),
        (
            "2–3 weeks from egg to adult.",
            "From egg to adult in 14 to 21 days.",
            ["egg_to_adult_days"],
        ),
        ("2–3 days from egg to adult.", "From egg to adult in 3 days.", ["egg_to_adult_days"]),
        ("~100 million base pairs and ~1,000 cells.", "~1000 genes.", []),
        ("Short lifespan: 2-3 weeks.", "From egg to adult in 3 days.", []),
        ("Generation turnover every 3 hours.", "From egg to adult in 3 days.", []),
        ("The genome is sequenced.", "The genome was sequenced in 1998.", []),
        ("", "It has 20,000 genes and develops from egg to adult in 3 days.", []),
    ],
)
def test_model_card_fact_formatting_units_and_nonoverlapping_details(reference, native, shared):
    for left, right in ((reference, native), (native, reference)):
        result = semantic.compare_organism_numeric_claims(left, right)
        assert result["shared_properties"] == shared


def test_development_does_not_consume_lifespan_or_generation_turnover():
    result = semantic.organism_numeric_claims(
        "Short lifespan (~2-3 weeks), rapid development (from egg to adult in ~3 days). "
        "Completes development from egg to adult in ~3 days at 20°C, with a generational turnover every 3 hours."
    )
    assert result["egg_to_adult_days"] == {(3, 3, False)}


def test_gene_count_does_not_consume_cells_or_base_pairs():
    result = semantic.organism_numeric_claims(
        "Contains ~1,000 cells; genome size ~100 million base pairs; 302 neurons; approximately 20,000 genes."
    )
    assert result["gene_count"] == {(20000, 20000, False)}


def test_sequencing_year_does_not_consume_unrelated_award_year():
    result = semantic.organism_numeric_claims(
        "The genome was sequenced in 1998. Pioneered RNAi technology (Fire and Mello, 2006 Nobel Prize)."
    )
    assert result["sequencing_year"] == {(1998, 1998, False)}


def test_extra_conflicting_value_for_shared_property_cannot_hide_behind_matching_one():
    reference = "It has 1,000 genes."
    native = "It has 1,000 genes. It has 20,000 genes."
    with pytest.raises(AssertionError, match="gene_count claims conflict"):
        semantic.compare_organism_numeric_claims(reference, native)


def test_fact_guard_is_additional_to_existing_organism_anchors():
    with pytest.raises(AssertionError, match="organism identification"):
        check(
            "model-card-quickstart",
            [text(ORGANISM_CORE)],
            [text("Its genome was sequenced in 1998.")],
        )


@pytest.mark.parametrize(
    "reference,native,property_name",
    [
        (
            "The genome was sequenced in 2002.",
            "The genome was sequenced not in 2002.",
            "sequencing_year",
        ),
        (
            "The genome was sequenced in 2002.",
            "The genome was not sequenced in 2002.",
            "sequencing_year",
        ),
        (
            "The genome sequence was published in 2002.",
            "The genome sequence was not published in 2002.",
            "sequencing_year",
        ),
        ("It has 20,000 genes.", "It does not have 20,000 genes.", "gene_count"),
        ("It has 20,000 genes.", "Not 20,000 genes.", "gene_count"),
        (
            "3 days from egg to adult.",
            "Egg-to-adult development does not take 3 days.",
            "egg_to_adult_days",
        ),
    ],
)
def test_shared_numeric_claim_denial_fails(reference, native, property_name):
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match=property_name + " claims conflict by denial"):
            semantic.compare_organism_numeric_claims(left, right)


def test_denial_of_different_value_is_not_a_conflict():
    result = semantic.compare_organism_numeric_claims(
        "The genome was sequenced in 2002.", "The genome was not sequenced in 2003."
    )
    assert result["shared_properties"] == ["sequencing_year"]


def test_approximation_does_not_add_undocumented_numeric_tolerance():
    with pytest.raises(AssertionError, match="gene_count claims conflict"):
        semantic.compare_organism_numeric_claims("About 20,000 genes.", "About 21,000 genes.")


@pytest.mark.parametrize(
    "other_clause",
    [
        "The human genome was sequenced in 2003.",
        "The genome of humans was sequenced in 2003.",
        "The Homo sapiens genome was sequenced in 2003.",
        "The mouse genome was sequenced in 2003.",
        "Humans have approximately 20,000 genes.",
        "The human genome contains 20,000 genes.",
        "Human genes (e.g., ~20,000) are studied too.",
        "Mice have 20,000 genes.",
        "Yeast has 6,000 genes.",
        "There are approximately 20,000 genes in humans.",
        "Humans do not have 1,000 genes.",
    ],
)
def test_explicit_other_organism_claims_do_not_change_target_facts(other_clause):
    base = "Its genome was sequenced in 1998. It has 1,000 genes."
    expected = semantic.organism_numeric_claims(base)
    for value in (base + " " + other_clause, other_clause + " " + base):
        assert semantic.organism_numeric_claims(value) == expected
        semantic.compare_organism_numeric_claims(base, value)
        semantic.compare_organism_numeric_claims(value, base)


def test_attached_other_organism_count_preserves_target_count():
    result = semantic.organism_numeric_claims(
        "It has 1,000 genes compared with 20,000 genes in humans."
    )
    assert result["gene_count"] == {(1000, 1000, False)}


@pytest.mark.parametrize(
    "value",
    [
        "Fewer genes (e.g., ~20,000) compared to humans, yet highly conserved pathways with mammals.",
        "Unlike humans, C. elegans has 20,000 genes.",
        "Compared with humans, it has 20,000 genes.",
        "The human genome contains 30,000 genes; C. elegans has 20,000 genes.",
        "Humans have 30,000 genes, while C. elegans has 20,000 genes.",
    ],
)
def test_human_comparison_does_not_hide_target_conflicting_count(value):
    assert semantic.organism_numeric_claims(value)["gene_count"] == {(20000, 20000, False)}
    with pytest.raises(AssertionError, match="gene_count claims conflict"):
        semantic.compare_organism_numeric_claims("It has 1,000 genes.", value)


def test_other_genome_year_does_not_hide_target_conflicting_year():
    value = "The human genome was sequenced in 2003. The C. elegans genome was sequenced in 2002."
    assert semantic.organism_numeric_claims(value)["sequencing_year"] == {(2002, 2002, False)}
    with pytest.raises(AssertionError, match="sequencing_year claims conflict"):
        semantic.compare_organism_numeric_claims("Its genome was sequenced in 1998.", value)


SAMPLED_REPRODUCTION_REFERENCE = (
    "It reproduces via parthenogenesis (asexual reproduction without fertilization), "
    "meaning females can produce offspring on their own."
)
SAMPLED_REPRODUCTION_NATIVE = (
    "They reproduce sexually, with females laying up to 500 eggs over their lifetime."
)


def test_retained_sampled_completion_reproduction_mechanisms_conflict():
    for left, right in (
        (SAMPLED_REPRODUCTION_REFERENCE, SAMPLED_REPRODUCTION_NATIVE),
        (SAMPLED_REPRODUCTION_NATIVE, SAMPLED_REPRODUCTION_REFERENCE),
    ):
        with pytest.raises(AssertionError, match="reproduction modes conflict"):
            semantic.compare_organism_reproduction_claims(left, right)
        with pytest.raises(AssertionError, match="reproduction modes conflict"):
            check(
                "model-card-sampling",
                [text(ORGANISM_CORE + " " + left)],
                [text(ORGANISM_CORE + " " + right)],
            )


@pytest.mark.parametrize(
    "reference,native",
    [
        ("It reproduces asexually.", "It reproduces via parthenogenesis."),
        ("It reproduces sexually.", "Sexual reproduction occurs."),
        ("Its reproduction is asexual.", "Asexual reproduction occurs."),
        ("It reproduces sexually.", "It reproduces both sexually and asexually."),
        ("It reproduces asexually.", "It uses sexual and asexual reproduction."),
        (
            "It reproduces asexually.",
            "It reproduces not only sexually but also asexually.",
        ),
        (
            "It reproduces sexually.",
            "It reproduces not merely sexually but also asexually.",
        ),
        ("It does not reproduce sexually.", "It reproduces asexually."),
        ("No asexual reproduction occurs.", "It reproduces sexually."),
        ("", "It reproduces sexually."),
        ("It is a hermaphrodite.", "It reproduces sexually."),
        ("It is self-fertilizing.", "It reproduces asexually."),
        ("Females can lay eggs.", "It reproduces sexually."),
        ("Researchers study parthenogenesis.", "It reproduces sexually."),
        ("Its genome is not large. It reproduces sexually.", "It reproduces sexually."),
    ],
)
def test_compatible_reproduction_and_omitted_mode_claims(reference, native):
    for left, right in ((reference, native), (native, reference)):
        semantic.compare_organism_reproduction_claims(left, right)


@pytest.mark.parametrize(
    "value",
    [
        "It is a hermaphrodite.",
        "It is self-fertilizing.",
        "Females can lay eggs without fertilization.",
        "Researchers study parthenogenesis.",
        "Its generation turnover is every 3 hours.",
    ],
)
def test_no_reproductive_mode_inferred_from_other_facts(value):
    assert semantic.organism_reproduction_claims(value) == set()


@pytest.mark.parametrize(
    "reference,native",
    [
        ("It reproduces sexually.", "It does not reproduce sexually."),
        ("It reproduces sexually.", "It doesn't reproduce sexually."),
        ("It reproduces sexually.", "They do not reproduce sexually."),
        ("It reproduces sexually.", "It reproduces not sexually."),
        ("It reproduces sexually.", "No sexual reproduction occurs."),
        ("It reproduces sexually.", "Its reproduction is not sexual."),
        ("It reproduces sexually.", "It does not use sexual reproduction."),
        ("It reproduces asexually.", "It does not reproduce via parthenogenesis."),
        ("It reproduces asexually.", "It reproduces only sexually."),
        (
            "It reproduces sexually and asexually.",
            "It reproduces exclusively sexually.",
        ),
        ("It reproduces sexually.", "It never reproduces sexually."),
    ],
)
def test_direct_reproduction_denials_conflict(reference, native):
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match="reproduction modes conflict by denial"):
            semantic.compare_organism_reproduction_claims(left, right)


def test_reproduction_modes_do_not_overlap_by_substring():
    assert semantic.organism_reproduction_claims("It reproduces asexually.") == {("asexual", False)}
    assert semantic.organism_reproduction_claims("It reproduces sexually.") == {("sexual", False)}


@pytest.mark.parametrize(
    "other_clause",
    ["Humans reproduce sexually.", "Humans use sexual reproduction."],
)
def test_other_organism_reproduction_does_not_change_target_mode(other_clause):
    target = "It reproduces asexually."
    expected = {("asexual", False)}
    for value in (target + " " + other_clause, other_clause + " " + target):
        assert semantic.organism_reproduction_claims(value) == expected
        semantic.compare_organism_reproduction_claims(target, value)


def test_generation_turnover_extra_claim_remains_separate_from_shared_development():
    reference = "~3 days from egg to adult, enabling rapid generation turnover."
    native = (
        "Completes development from egg to adult in ~3 days at 20°C, "
        "with a generational turnover every 3 hours, accelerating experimental timelines."
    )
    result = semantic.compare_organism_numeric_claims(reference, native)
    assert result["shared_properties"] == ["egg_to_adult_days"]
    assert result["reference_claims"]["egg_to_adult_days"] == [["3", "3", False]]
    assert result["native_claims"]["egg_to_adult_days"] == [["3", "3", False]]
    assert not semantic.compare_organism_reproduction_claims(reference, native)["shared_property"]


@pytest.mark.parametrize(
    "case", ("model-card-quickstart", "model-card-sampling", "sampled-answer-completion")
)
@pytest.mark.parametrize(
    "reference,native",
    [
        (
            "It has four larval stages (L1, L2, L3, L4) before reaching adulthood.",
            "The worm goes through three distinct larval stages (L1, L2, L3) before becoming an adult.",
        ),
        ("4 larval stages.", "3 larval stages."),
        ("Four larval stages.", "Not four larval stages."),
        ("Four larval stages.", "It does not undergo four larval stages."),
        ("Four larval stages.", "It does not have exactly four larval stages."),
        ("Four larval stages.", "It doesn't go through four distinct larval stages."),
        ("Four larval stages.", "It doesn’t have four larval stages."),
    ],
)
def test_model_card_rejects_shared_larval_stage_conflicts(case, reference, native):
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match="larval_stage_count claims conflict"):
            check(case, [text(ORGANISM_CORE + " " + left)], [text(ORGANISM_CORE + " " + right)])


@pytest.mark.parametrize(
    "left,right,count",
    [
        ("Four larval stages.", "4 distinct larval stages (L1 through L4).", 4),
        ("Four larval stages (L1–L4).", "4 larval stages (L1, L2, L3, L4).", 4),
        ("Three larval stages.", "3 distinct larval stages.", 3),
        ("One larval stage.", "1 larval stage.", 1),
    ],
)
def test_larval_stage_count_spelling_and_labels(left, right, count):
    for value in (left, right):
        assert semantic.organism_numeric_claims(value)["larval_stage_count"] == {
            (count, count, False)
        }
    for reference, native in ((left, right), (right, left)):
        assert semantic.compare_organism_numeric_claims(reference, native)["shared_properties"] == [
            "larval_stage_count"
        ]


@pytest.mark.parametrize(
    "value",
    [
        "Not three larval stages but four larval stages.",
        "Not three but four larval stages.",
        "It does not undergo three larval stages.",
        "Not only four larval stages but also adulthood.",
        "Adult lifespan is not three weeks. It has four larval stages.",
    ],
)
def test_larval_stage_count_compatible_negation(value):
    for reference, native in (("Four larval stages.", value), (value, "Four larval stages.")):
        semantic.compare_organism_numeric_claims(reference, native)


@pytest.mark.parametrize(
    "other",
    [
        "Adult lifespan is two to three weeks.",
        "From egg to adult in 3 days.",
        "It has 302 neurons and 1000 cells.",
        "The L4 larval stage lasts 3 days.",
        "A four-day larval stage.",
        "Humans have three larval stages.",
        "Drosophila has three distinct larval stages.",
    ],
)
def test_larval_stage_count_ignores_other_quantities_and_supported_subjects(other):
    assert semantic.organism_numeric_claims(other)["larval_stage_count"] == set()
    for value in ("Four larval stages. " + other, other + " Four larval stages."):
        assert semantic.organism_numeric_claims(value)["larval_stage_count"] == {(4, 4, False)}


@pytest.mark.parametrize(
    "value",
    [
        "It has larval stages, including L1 and L2.",
        "The larval stages are L1, L2, L3, L4.",
        "The larval stages run from L1 through L4.",
    ],
)
def test_enumerated_only_larval_stages_do_not_infer_a_cardinal(value):
    assert semantic.organism_numeric_claims(value)["larval_stage_count"] == set()
    result = semantic.compare_organism_numeric_claims("Four larval stages.", value)
    assert result["shared_properties"] == []


def test_matching_larval_count_cannot_hide_an_extra_conflicting_count():
    with pytest.raises(AssertionError, match="larval_stage_count claims conflict"):
        semantic.compare_organism_numeric_claims(
            "Four larval stages.", "Four larval stages. Three larval stages."
        )


@pytest.mark.parametrize(
    "spelling,count", [("1,004", 1004), ("4,000", 4000), ("-1,004", -1004), ("+4.0", 4)]
)
def test_larval_stage_count_reads_complete_numeric_spelling(spelling, count):
    value = f"It has {spelling} larval stages."
    assert semantic.organism_numeric_claims(value)["larval_stage_count"] == {(count, count, False)}
    if count != 4:
        for reference, native in (("Four larval stages.", value), (value, "Four larval stages.")):
            with pytest.raises(AssertionError, match="larval_stage_count claims conflict"):
                semantic.compare_organism_numeric_claims(reference, native)


@pytest.mark.parametrize("spelling", ["twenty-four", "twenty‑four"])
def test_unsupported_compound_larval_count_does_not_become_its_suffix(spelling):
    value = f"It has {spelling} larval stages."
    for reference, native in (("Four larval stages.", value), (value, "Four larval stages.")):
        with pytest.raises(AssertionError, match="unsupported compound larval-stage count"):
            semantic.compare_organism_numeric_claims(reference, native)


def test_larval_stage_exactly_modifier_preserves_scoped_denial():
    assert semantic.organism_numeric_claims("It does not have exactly four larval stages.")[
        "larval_stage_count"
    ] == {(4, 4, True)}
    assert semantic.organism_numeric_claims("It has exactly four larval stages.")[
        "larval_stage_count"
    ] == {(4, 4, False)}


@pytest.mark.parametrize(
    "case", ("model-card-quickstart", "model-card-sampling", "sampled-answer-completion")
)
def test_visible_germ_cell_contradiction_is_rejected_even_at_budget(case):
    reference = "An adult has exactly 959 somatic cells (plus a few germ cells)."
    native = (
        "Exactly 959 somatic cells in a fully mature adult (no germline cells are formed; "
        "the germline is derived from the same cells that become somatic)."
    )
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match="germ-cell presence claims conflict"):
            check(
                case,
                [text(ORGANISM_CORE + " " + left)],
                [text(ORGANISM_CORE + " " + right)],
                reference_ids=[1] * 512,
                native_ids=[2] * 512,
                limit=512,
            )


@pytest.mark.parametrize(
    "value",
    [
        "Plus a few germ cells.",
        "It has germline cells.",
        "Germ cells are present.",
        "Germ cells are formed.",
        "There are some germ cells.",
        "Germ cells are not absent.",
        "Unlike humans, C. elegans has germ cells.",
        "C. elegans has germ cells, humans have no germ cells.",
        "No neurons are formed. It has germ cells.",
    ],
)
def test_explicit_germ_cell_presence_paraphrases(value):
    assert semantic.organism_germ_cell_claims(value) == {"present"}
    semantic.compare_organism_germ_cell_claims("Plus a few germ cells.", value)
    for left, right in (("No germline cells are formed.", value), (value, "It has no germ cells.")):
        with pytest.raises(AssertionError, match="germ-cell presence claims conflict"):
            semantic.compare_organism_germ_cell_claims(left, right)


@pytest.mark.parametrize(
    "value",
    [
        "No germline cells are formed.",
        "It has no germ cells.",
        "It lacks germ cells.",
        "Germ cells are absent.",
        "Germ cells are not present.",
        "It doesn't have germ cells.",
        "It cannot have germ cells.",
        "It never has germ cells.",
        "Humans have germ cells, C. elegans has no germ cells.",
    ],
)
def test_explicit_germ_cell_absence_paraphrases(value):
    assert semantic.organism_germ_cell_claims(value) == {"absent"}
    semantic.compare_organism_germ_cell_claims("No germline cells are formed.", value)
    with pytest.raises(AssertionError, match="germ-cell presence claims conflict"):
        semantic.compare_organism_germ_cell_claims("Plus a few germ cells.", value)


@pytest.mark.parametrize(
    "value",
    [
        "If the worm is sterile, no germ cells are formed.",
        "Germ cells may be absent.",
        "No germ cells are formed in mutants.",
        "No germ cells are formed in somatic tissue.",
        "No germ cells are formed yet.",
        "Not all germ cells are present.",
        "Germ cells are not always present.",
        "The claim that no germ cells are formed is false.",
        "It is not true that no germ cells are formed.",
        "It does not lack germ cells.",
        "It does not have no germ cells.",
        "No germ cells are lost.",
        "No germ cells are missing.",
        "Some germ cells are absent.",
        "Humans have no germ cells.",
        "No germ cells are formed in mice.",
        "Human germ cells are absent.",
        "Other organisms have germ cells.",
        "It has germ-cell tumors.",
        "Does C. elegans have germ cells?",
        "The worm is transparent.",
    ],
)
def test_germ_cell_omissions_qualifiers_denials_and_other_subjects_imply_nothing(value):
    assert semantic.organism_germ_cell_claims(value) == set()
    for target in ("Plus a few germ cells.", "No germ cells are formed."):
        for left, right in ((target, value), (value, target)):
            assert not semantic.compare_organism_germ_cell_claims(left, right)["shared_property"]


def test_germ_cell_claims_do_not_apply_to_reasoning_only_sampling_budget():
    result = check(
        "model-card-sampling",
        [{"type": "reasoning", "text": ORGANISM_CORE + " Plus a few germ cells."}],
        [{"type": "reasoning", "text": ORGANISM_CORE + " No germ cells are formed."}],
        reference_ids=[1] * 512,
        native_ids=[2] * 512,
        limit=512,
    )
    assert result["completion_state"] == "reasoning_only_at_budget"
    assert "explicit_germ_cell_claims" not in result


@pytest.mark.parametrize(
    "case",
    ("model-card-quickstart", "model-card-sampling", "sampled-answer-completion"),
)
@pytest.mark.parametrize("budget", [False, True])
def test_visible_cellularity_contradiction_is_rejected(case, budget):
    reference = ORGANISM_CORE + " The worm has exactly 302 neurons."
    native = ORGANISM_CORE + (
        " Discovered in 1892 by William Howard, it is a single-celled organism "
        "(in the sense that it is an animal, not a cell) belonging to the phylum Nematoda."
        " It contains exactly 302 neurons."
    )
    args = dict(reference_ids=[1] * 512, native_ids=[2] * 512, limit=512) if budget else {}
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match="cellularity claims conflict"):
            check(case, [text(left)], [text(right)], **args)


@pytest.mark.parametrize(
    "value",
    [
        "C. elegans is single-celled.",
        "Caenorhabditis elegans is a single celled organism.",
        "The worm is unicellular.",
        "This nematode is a single‑celled animal.",
        "C. elegans is a worm. It is unicellular.",
        "Bacteria are multicellular, whereas C. elegans is single-celled.",
        "C. elegans is unicellular, whereas bacteria are multicellular.",
    ],
)
def test_affirmed_target_single_cell_classification(value):
    assert semantic.organism_cellularity_claims(value) == {"single"}
    with pytest.raises(AssertionError, match="cellularity claims conflict"):
        semantic.compare_organism_cellularity_claims("The worm has 302 neurons.", value)


@pytest.mark.parametrize(
    "value",
    [
        "C. elegans is multicellular.",
        "The worm contains exactly 302 neurons.",
        "C. elegans is a worm. This organism has 959 somatic cells.",
        "C. elegans comprises two cells.",
        "C. elegans consists of four cells.",
        "C. elegans is a worm. It has roughly 1,000 cells.",
        "Bacteria are single-celled, whereas C. elegans is multicellular.",
        "C. elegans is multicellular, whereas bacteria are single-celled.",
    ],
)
def test_target_multiple_cell_evidence_without_exact_count_policy(value):
    assert semantic.organism_cellularity_claims(value) == {"multiple"}
    semantic.compare_organism_cellularity_claims("The worm has 302 neurons.", value)
    with pytest.raises(AssertionError, match="cellularity claims conflict"):
        semantic.compare_organism_cellularity_claims("The worm is unicellular.", value)


@pytest.mark.parametrize(
    "value",
    [
        "C. elegans is not single-celled.",
        "The worm isn't unicellular.",
        "It is not true that C. elegans is single-celled.",
        'The claim "C. elegans is single-celled" is false.',
        "A textbook incorrectly calls C. elegans 'unicellular'.",
        'A source reports: "C. elegans is unicellular."',
        "A source wrote ‘C. elegans is unicellular.’",
        "It is a misconception that C. elegans is single-celled.",
        "Some people believe C. elegans is a single-celled organism.",
        "A source states that C. elegans is unicellular.",
        "It was falsely claimed that C. elegans is unicellular.",
        "C. elegans is single-celled. Correction: that description is wrong; it is multicellular.",
        "C. elegans is unicellular. That classification is incorrect; it is not single-celled.",
        "C. elegans is single-celled, but that is incorrect; it is multicellular.",
        "A biologist denies that C. elegans is single-celled.",
        "It is not the case that C. elegans is unicellular.",
        "C. elegans develops from a single cell.",
        "The worm begins life as a single fertilized egg.",
        "C. elegans is unicellular during embryonic development.",
        "C. elegans may be unicellular.",
        "Is C. elegans single-celled?",
        "Bacteria are single-celled, whereas C. elegans is transparent.",
        "C. elegans is transparent, whereas bacteria are single-celled.",
        "C. elegans is a worm. An amoeba is another organism. It is unicellular.",
        "C. elegans is a roundworm. Yeast is a model organism. This organism is unicellular.",
        "C. elegans is a roundworm. Yeast is a model organism. The organism is unicellular.",
        "Humans have 302 neurons.",
        "C. elegans has one cell.",
        "C. elegans has 1 cell.",
        "C. elegans has 1.0 cells.",
        "The worm has no cells.",
        "The worm has 302 neuronal genes.",
        "C. elegans is a single-celled organism's predator.",
        "The worm is transparent.",
    ],
)
def test_cellularity_qualified_retracted_other_subject_and_origin_claims_imply_nothing(
    value,
):
    assert semantic.organism_cellularity_claims(value) == set()
    for target in ("The worm is unicellular.", "The worm has 302 neurons."):
        for left, right in ((target, value), (value, target)):
            assert not semantic.compare_organism_cellularity_claims(left, right)["shared_property"]


def test_negated_single_cell_statement_can_affirm_multicellularity():
    value = "C. elegans is not unicellular; it is multicellular."
    assert semantic.organism_cellularity_claims(value) == {"multiple"}
    semantic.compare_organism_cellularity_claims("The worm has 302 neurons.", value)


def test_cellularity_is_not_checked_in_reasoning_only_sampling_budget():
    result = check(
        "model-card-sampling",
        [{"type": "reasoning", "text": ORGANISM_CORE + " The worm has 302 neurons."}],
        [{"type": "reasoning", "text": ORGANISM_CORE + " The worm is unicellular."}],
        reference_ids=[1] * 512,
        native_ids=[2] * 512,
        limit=512,
    )
    assert result["completion_state"] == "reasoning_only_at_budget"
    assert "explicit_cellularity_claims" not in result


@pytest.mark.parametrize("case", ("model-card-quickstart", "model-card-sampling"))
def test_complete_larval_sequence_rejects_the_retained_three_stage_contradiction(case):
    reference = "Complete but very short: egg → larva (L1–L4) → adult."
    native = "Egg → first‑larva (L1) → second‑larva (L2) → third‑larva (L3) → adult."
    for left, right in ((reference, native), (native, reference)):
        with pytest.raises(AssertionError, match="larval_stage_count claims conflict"):
            check(case, [text(ORGANISM_CORE + " " + left)], [text(ORGANISM_CORE + " " + right)])


@pytest.mark.parametrize(
    "value,count",
    [
        ("Egg -> L1 -> L2 -> L3 -> adult.", 3),
        ("Egg → L1 → L2 → L3 → L4 → adult.", 4),
        ("Egg ⇒ larval stages (L1 through L4) ⇒ adulthood.", 4),
        ("Egg ⟶ larvae (L1, L2, L3, and L4) ⟶ adult.", 4),
        ("Egg → larva (L1–L4) → adult. The remaining answer is cut off", 4),
        ("Egg → L1 → adult.", 1),
        ("Drosophila has a different life cycle. C. elegans: egg → L1–L4 → adult.", 4),
        ("Drosophila: egg → L1 → L2 → L3 → adult; C. elegans: egg → L1–L4 → adult.", 4),
        ("C. elegans: egg → L1–L4 → adult; Drosophila: egg → L1 → L2 → L3 → adult.", 4),
    ],
)
def test_complete_larval_sequence_has_a_bounded_explicit_count(value, count):
    assert semantic.organism_numeric_claims(value)["larval_stage_count"] == {(count, count, False)}
    comparison = semantic.compare_organism_numeric_claims(f"{count} larval stages.", value)
    assert comparison["shared_properties"] == ["larval_stage_count"]


@pytest.mark.parametrize(
    "value",
    [
        "Egg → L1 → L2 → L3",
        "Egg → L1 → L2 → L3 →",
        "Egg → L1 → L2 → L3 → adul",
        "Egg → L1 → L2 → ... → adult.",
        "Egg → L1 → L2 → … → adult.",
        "L1 → L2 → L3 → adult.",
        "Egg → L2 → L3 → adult.",
        "Egg → L1 → L3 → adult.",
        "Egg → L1 → L2 → L2 → adult.",
        "Egg → L4–L1 → adult.",
        "Egg → L1–L40 → adult.",
        "Egg → L1 → dauer → L4 → adult.",
        "Egg → L1/L2/L3 → adult.",
        "Egg → L1 → L2 → L3 → adult?",
        'Someone wrote "egg → L1 → L2 → L3 → adult".',
        "Someone wrote ‘egg → L1 → L2 → L3 → adult’.",
        "It does not follow egg → L1 → L2 → L3 → adult.",
        "It doesn’t follow egg → L1 → L2 → L3 → adult.",
        "If C. elegans followed egg → L1 → L2 → L3 → adult, this would be different.",
        "C. elegans may follow egg → L1 → L2 → L3 → adult.",
        "A source claimed egg → L1 → L2 → L3 → adult.",
        "Some people believe egg → L1 → L2 → L3 → adult.",
        "This simplified diagram shows egg → L1 → L2 → L3 → adult.",
        "Drosophila has a life cycle: egg → L1 → L2 → L3 → adult.",
        "A fly has a life cycle: egg → L1 → L2 → L3 → adult.",
        "A butterfly has a life cycle: egg → L1 → L2 → L3 → adult.",
        "Drosophila, unlike C. elegans, has a life cycle: egg → L1 → L2 → L3 → adult.",
        "C. elegans is a nematode. Drosophila is a fly. It develops egg → L1 → L2 → L3 → adult.",
        "Egg → L1 → L2 → L3 → adult, but that is incorrect.",
        "Egg → L1 → L2 → L3 → adult. That is incorrect.",
        "Egg → L1 → L2 → L3 → adult. That’s wrong.",
        "Egg → L1 → L2 → L3 → adult. That is not correct.",
        "Egg → L1 → L2 → L3 → adult. This sequence is incomplete.",
        "Egg → L1 → L2 → L3 → adult, but this sequence is incomplete.",
        "Rather than egg → L1 → L2 → L3 → adult, use the complete sequence.",
    ],
)
def test_open_or_unendorsed_larval_sequences_do_not_invent_a_total(value):
    assert semantic.organism_numeric_claims(value)["larval_stage_count"] == set()
    assert (
        semantic.compare_organism_numeric_claims("Four larval stages.", value)["shared_properties"]
        == []
    )


def test_complete_sequence_conflict_is_not_hidden_by_an_additional_matching_count():
    with pytest.raises(AssertionError, match="larval_stage_count claims conflict"):
        semantic.compare_organism_numeric_claims(
            "Four larval stages.",
            "Four larval stages. Egg → L1 → L2 → L3 → adult.",
        )
