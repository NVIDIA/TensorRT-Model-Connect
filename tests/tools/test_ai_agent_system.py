from tools import ai_agent_system


def test_task_body_contains_required_sections() -> None:
    body = ai_agent_system.task_body(
        scope="docs only",
        change="remove stale text",
        acceptance=["stale text is gone"],
        verification=["rg stale docs || true"],
        non_goals=["no runtime edits"],
        risk="Low.",
    )

    assert ai_agent_system.validate_task_description(body) == []
    assert "## Scope" in body
    assert "- stale text is gone" in body


def test_validate_task_description_reports_missing_sections() -> None:
    missing = ai_agent_system.validate_task_description(
        "## Scope\nOne file.\n\n## Change\nDo it.\n"
    )

    assert missing == ["Acceptance Criteria", "Verification", "Non-goals"]


def test_project_path_preserves_github_slug() -> None:
    assert ai_agent_system.encoded_project("NVIDIA/TensorRT-Model-Connect") == "NVIDIA/TensorRT-Model-Connect"
    assert ai_agent_system.encoded_project("12345") == "12345"


def test_is_ai_promotion_schedule_accepts_explicit_variable() -> None:
    schedule = {
        "description": "Periodic maintenance",
        "variables": [{"key": "AI_STAGING_PROMOTE", "value": "1"}],
    }

    assert ai_agent_system.is_ai_promotion_schedule(schedule)


def test_is_ai_promotion_schedule_accepts_descriptive_name() -> None:
    schedule = {"description": "AI staging promotion PR", "variables": []}

    assert ai_agent_system.is_ai_promotion_schedule(schedule)


def test_is_ai_promotion_schedule_rejects_unrelated_schedule() -> None:
    schedule = {
        "description": "Nightly",
        "variables": [{"key": "AI_STAGING_PROMOTE", "value": "0"}],
    }

    assert not ai_agent_system.is_ai_promotion_schedule(schedule)


def test_has_any_prefix_empty_prefix_tuple_never_matches() -> None:
    # str.startswith(()) returns False for any input; guards routing logic
    # against accidentally matching every branch when no prefixes are configured.
    assert ai_agent_system.has_any_prefix("ai-task-26", ()) is False
    assert ai_agent_system.has_any_prefix("", ()) is False


def test_has_any_prefix_no_match_and_match_at_start() -> None:
    prefixes = ("ai-task-", "ai-promotion-")

    # Substring-but-not-prefix must not count as a match.
    assert ai_agent_system.has_any_prefix("feat/ai-task-26", prefixes) is False
    assert ai_agent_system.has_any_prefix("main", prefixes) is False

    # Matches must anchor at the start of the branch name.
    assert ai_agent_system.has_any_prefix("ai-task-26-cover-pure-helpers", prefixes) is True
    assert ai_agent_system.has_any_prefix("ai-promotion-2026-04-22", prefixes) is True


def test_csv_labels_empty_single_and_preserves_order() -> None:
    # An empty label list must join to the empty string (not a stray comma).
    assert ai_agent_system.csv_labels([]) == ""

    # Single label round-trips with no separator.
    assert ai_agent_system.csv_labels(["ai:task"]) == "ai:task"

    # Order from the caller is preserved; we intentionally pass a non-sorted list
    # to confirm csv_labels does not re-order labels.
    labels = ["ai:sanity-pending", "ai-generated", "ai:staging-pr"]
    assert ai_agent_system.csv_labels(labels) == "ai:sanity-pending,ai-generated,ai:staging-pr"


def test_task_issue_labels_include_ai_tags() -> None:
    labels = ai_agent_system.task_issue_labels(["priority:high"])

    assert labels == ["AI", "ai-generated", "ai:task", "ai:ready", "priority:high"]


def test_task_issue_labels_deduplicate_ai_tags() -> None:
    labels = ai_agent_system.task_issue_labels(["AI", "ai-generated", "ai:needs-human"])

    assert labels == ["AI", "ai-generated", "ai:task", "ai:ready", "ai:needs-human"]


def test_schedule_variables_handles_missing_key_and_coerces_values() -> None:
    # Schedule with no "variables" field at all -> empty dict, no KeyError.
    assert ai_agent_system.schedule_variables({}) == {}

    # "variables" present but None -> still empty dict (the `or []` fallback).
    assert ai_agent_system.schedule_variables({"variables": None}) == {}

    # Entries with key=None must be skipped; other values must be coerced to str
    # so downstream `.get("AI_STAGING_PROMOTE") == "1"` comparisons work even
    # when GitHub returns numeric/boolean values.
    schedule = {
        "variables": [
            {"key": None, "value": "ignored"},
            {"key": "AI_STAGING_PROMOTE", "value": 1},
            {"key": "ENABLED", "value": True},
            {"key": "RATIO", "value": 0.5},
        ]
    }

    result = ai_agent_system.schedule_variables(schedule)

    assert "ignored" not in result.values()
    assert result["AI_STAGING_PROMOTE"] == "1"
    assert result["ENABLED"] == "True"
    assert result["RATIO"] == "0.5"


def test_validate_task_description_matches_headings_case_insensitively() -> None:
    # TASK_REQUIRED_HEADINGS uses Title Case, but validate_task_description
    # lowercases both sides so bodies written as '## scope' / '## CHANGE'
    # must still validate; this protects against LLM-authored descriptions
    # that don't preserve exact casing.
    description = (
        "## scope\nOne file.\n\n"
        "## CHANGE\nDo it.\n\n"
        "## Acceptance criteria\n- item\n\n"
        "## VERIFICATION\n- pytest\n\n"
        "## non-goals\n- nothing\n"
    )

    assert ai_agent_system.validate_task_description(description) == []


def test_project_path_is_not_percent_encoded_for_github_routes() -> None:
    # GitHub repository routes use owner/repo path segments directly.
    assert ai_agent_system.encoded_project("987654") == "987654"

    encoded = ai_agent_system.encoded_project("grp/subgrp/proj")
    assert encoded == "grp/subgrp/proj"


def test_task_body_with_empty_bullet_lists_still_validates() -> None:
    # If acceptance/verification/non_goals are empty, the heading must still
    # render (so validate_task_description finds it) but no '- ' bullet lines
    # should be emitted. This mirrors sparse task descriptions operators may
    # draft before filling in details.
    body = ai_agent_system.task_body(
        scope="docs only",
        change="remove stale text",
        acceptance=[],
        verification=[],
        non_goals=[],
        risk="Low.",
    )

    # Required headings are all present, so validation passes.
    assert ai_agent_system.validate_task_description(body) == []

    # No bullet lines are rendered for the empty sections.
    assert "\n- " not in body
    for line in body.splitlines():
        assert not line.startswith("- ")

    # And the headings themselves are still present in order.
    for heading in ai_agent_system.TASK_REQUIRED_HEADINGS:
        assert f"## {heading}" in body
