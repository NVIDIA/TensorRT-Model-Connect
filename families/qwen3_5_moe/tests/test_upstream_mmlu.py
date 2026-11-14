# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral full-workload guards; synthetic fixtures are not model quality evidence."""

from copy import deepcopy

import pytest

from families.qwen3_5_moe.tests.upstream_mmlu import (
    CONTROLS,
    EDGE_REVISION,
    ERROR_TEXT,
    REQUEST_COUNT,
    checked_predictions,
    public_requests,
    raw_prompt,
    score_report,
    validate_dataset,
)


@pytest.fixture
def dataset():
    """Provide a full-count synthetic dataset without changing production acceptance constants."""
    row = {
        "messages": [
            {"role": "system", "content": " leading\n"},
            {"role": "user", "content": " question?  "},
        ],
        "answer": "A",
        "subject": "synthetic",
    }
    return {**CONTROLS, "requests": [row] * REQUEST_COUNT}


@pytest.fixture
def public_report():
    """Provide one synthetic token per ordered request and the actual driver's required headers."""
    return {
        "passed": True,
        "backend": "edge_llm",
        "edge_revision": EDGE_REVISION,
        "task": "text_generation",
        "results": [
            {"id": str(index), "text": "A", "token_ids": [65]} for index in range(REQUEST_COUNT)
        ],
    }


def test_prepare_preserves_every_request_and_original_whitespace(dataset):
    before = deepcopy(dataset)
    requests = public_requests(dataset)
    assert len(requests) == REQUEST_COUNT
    assert requests[0]["prompt"] == " leading\n question?  "
    assert requests[-1]["id"] == str(REQUEST_COUNT - 1)
    assert all(row["config"]["max_new_tokens"] == 1 for row in requests)
    assert all(row["config"]["use_chat_template"] is False for row in requests)
    assert dataset == before
    # Edge 0.11 exports the same raw framing in one user message.
    current = deepcopy(dataset)
    current["requests"][0]["messages"] = [{"role": "user", "content": " leading\n question?  "}]
    assert public_requests(current) == requests
    assert current["requests"][0]["messages"][0]["content"] == " leading\n question?  "


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", 8),
        ("max_generate_length", 32),
        ("temperature", 1.0),
        ("apply_chat_template", True),
        ("top_k", True),
        ("top_p", 0.9),
    ],
)
def test_changed_controls_rejected(dataset, key, value):
    dataset[key] = value
    with pytest.raises(ValueError, match="control"):
        validate_dataset(dataset)


def test_subset_or_unknown_controls_rejected(dataset):
    dataset["requests"] = dataset["requests"][:-1]
    with pytest.raises(ValueError, match="all"):
        validate_dataset(dataset)
    dataset["seed"] = 42
    with pytest.raises(ValueError, match="controls"):
        validate_dataset(dataset)


def test_unsupported_media_roles_and_row_overrides_rejected(dataset):
    row = deepcopy(dataset["requests"][0])
    row["messages"][1]["content"] = [{"type": "image", "image": "fixture.png"}]
    with pytest.raises(ValueError, match="plain-text"):
        raw_prompt(row)
    row["messages"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        raw_prompt(row)
    row["max_generate_length"] = 12
    with pytest.raises(ValueError, match="fields"):
        raw_prompt(row)


def test_complete_public_output_keeps_full_scorer_denominator(dataset, public_report):
    for row, text in zip(public_report["results"], ("", " ", "\n\n")):
        row["text"] = text
    result = checked_predictions(dataset, public_report, "public")
    assert [r["output_text"] for r in result["responses"][:3]] == ["", " ", "\n\n"]
    assert len(result["responses"]) == REQUEST_COUNT
    assert result["responses"][-1]["request_idx"] == REQUEST_COUNT - 1
    assert result["responses"][0]["messages"] == dataset["requests"][0]["messages"]


@pytest.mark.parametrize(
    "fault", ["short", "extra", "reordered", "duplicate", "error", "invalid_text", "budget"]
)
def test_public_execution_gaps_rejected(dataset, public_report, fault):
    rows = public_report["results"]
    if fault == "short":
        rows.pop()
    elif fault == "extra":
        rows.append(dict(rows[0]))
    elif fault == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif fault == "error":
        rows[0]["text"] = ERROR_TEXT
    elif fault == "invalid_text":
        rows[0]["text"] = None
    else:
        rows[0]["token_ids"] = [65, 66]
    with pytest.raises(ValueError):
        checked_predictions(dataset, public_report, "public")


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("backend", "native"),
        ("edge_revision", "wrong"),
        ("task", "vision_language_generation"),
    ],
)
def test_public_execution_identity_required(dataset, public_report, field, value):
    public_report[field] = value
    with pytest.raises(ValueError, match="public Edge"):
        checked_predictions(dataset, public_report, "public")


def native_report(dataset):
    """Build synthetic pinned native serialization with exact original prompt identity."""
    row = dataset["requests"][0]
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in row["messages"]
    ]
    return {
        "responses": [
            {
                "request_idx": index,
                "batch_idx": 0,
                "messages": messages,
                "formatted_complete_request": raw_prompt(row),
                "finish_reason": "max-length",
                "output_text": "A",
            }
            for index in range(REQUEST_COUNT)
        ]
    }


def test_complete_native_serialization_is_checked(dataset):
    report = native_report(dataset)
    report["responses"][0]["output_text"] = "\n\n"
    result = checked_predictions(dataset, report, "edge")
    assert result["responses"][0]["output_text"] == "\n\n"
    assert len(result["responses"]) == REQUEST_COUNT


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_idx", 1),
        ("request_idx", False),
        ("batch_idx", 1),
        ("messages", []),
        ("formatted_complete_request", "trimmed"),
        ("finish_reason", "error"),
    ],
)
def test_native_order_input_and_completion_gaps_rejected(dataset, field, value):
    report = native_report(dataset)
    report["responses"][0][field] = value
    with pytest.raises(ValueError, match="Native"):
        checked_predictions(dataset, report, "edge")


def test_scorer_uses_exact_fraction_not_rounded_display():
    below = score_report(f"Overall Accuracy: 0.3500 (35.00%) - 4914/{REQUEST_COUNT} correct")
    above = score_report(f"Overall Accuracy: 0.3500 (35.00%) - 4915/{REQUEST_COUNT} correct")
    assert below["passed"] is False
    assert above["passed"] is True
    assert above["listed_batch8_qualification"] is False


@pytest.mark.parametrize(
    "text",
    [
        "Overall Accuracy: 1.0 (100%) - 4/4 correct",
        f"Skipped 1/{REQUEST_COUNT} entries\nOverall Accuracy: 1.0 - {REQUEST_COUNT}/{REQUEST_COUNT} correct",
        f"Overall Accuracy: 1.0 - {REQUEST_COUNT + 1}/{REQUEST_COUNT} correct",
        "No valid predictions to evaluate (all entries were errors)",
    ],
)
def test_partial_filtered_or_invalid_scorer_results_rejected(text):
    with pytest.raises(ValueError):
        score_report(text)


@pytest.mark.parametrize("token", [-1, True, "65"])
def test_invalid_public_token_rejected(dataset, public_report, token):
    """Malformed generated IDs never constitute a valid one-token answer."""
    public_report["results"][0]["token_ids"] = [token]
    with pytest.raises(ValueError, match="output-budget"):
        checked_predictions(dataset, public_report, "public")


def test_native_eos_label_matches_pinned_serializer(dataset):
    """Accept actual terminal EOS and reject enum-derived shorthand."""
    report = native_report(dataset)
    report["responses"][0]["finish_reason"] = "end-of-sequence"
    assert len(checked_predictions(dataset, report, "edge")["responses"]) == REQUEST_COUNT
    report["responses"][0]["finish_reason"] = "eos"
    with pytest.raises(ValueError):
        checked_predictions(dataset, report, "edge")
