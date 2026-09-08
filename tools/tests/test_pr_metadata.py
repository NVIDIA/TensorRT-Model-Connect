# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pull-request evidence validation."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from tools import pr_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWED_HEAD = "d" * 40


def _complete_body() -> str:
    return """\
## Background

Fix model selection ambiguity reported in #123.

## Exit Criteria

- Qwen selection is unambiguous.
- Existing runtime math is unchanged.

## Implementation

Update the Qwen builder and its model-owned tests. There are no public API,
ABI, bundle, or dependency changes.

### Change categories

- [x] Model or runtime behavior
- [ ] Public API
- [ ] ABI
- [ ] Bundle or artifact format
- [ ] Dependencies
- [ ] Documentation only
- [ ] CI or developer tooling

## Validation

### Commands and Results

`python3 -m pytest -q core/builder/tests/test_build.py`: 18 passed.

### Hardware, Environment, and Revisions

Repository head `def456`; Qwen revision `abc123`; CPU-only Ubuntu 24.04.

### Not Run / Remaining Gaps

GPU execution was not run because runtime math is unchanged.

## Contributor Self-Review

### Method

`$review-trtmc-pr` on draft PR #123.

### Reviewed Head

`{REVIEWED_HEAD}`

### Result

PASS

### Findings and Resolution

PASS; no blocking or high-severity findings remain.

## Notes For Future Readers

Family-owned support resolution replaces the previous selector. Existing
bundles must be rebuilt; there is no compatibility path. Review the unique
ownership check before the model-owned regression.

### Risk level

- [x] Low
- [ ] Medium
- [ ] High

The change is isolated to model selection.

""".format(REVIEWED_HEAD=REVIEWED_HEAD)


def _without_self_review(body: str) -> str:
    start = body.index("## Contributor Self-Review")
    end = body.index("## Notes For Future Readers")
    return body[:start] + body[end:]


def test_complete_pull_request_body_passes() -> None:
    assert pr_metadata.validate_body(_complete_body()) == []


def test_comments_and_headings_do_not_satisfy_required_evidence() -> None:
    body = _complete_body().replace(
        "Fix model selection ambiguity reported in #123.",
        "<!-- contributor left the Background section empty -->",
    )

    assert "Required section is empty: Background" in pr_metadata.validate_body(body)


def test_validation_requires_commands_results_environment_revisions_and_gaps() -> None:
    body = _complete_body().replace(
        "Repository head `def456`; Qwen revision `abc123`; CPU-only Ubuntu 24.04.",
        "<!-- no revisions supplied -->",
    )

    assert (
        "Required subsection is empty: Validation / Hardware, Environment, and Revisions"
        in pr_metadata.validate_body(body)
    )


def test_self_review_requires_method_head_result_and_findings() -> None:
    evidence_by_subsection = {
        "Method": "`$review-trtmc-pr` on draft PR #123.",
        "Reviewed Head": f"`{REVIEWED_HEAD}`",
        "Result": "PASS",
        "Findings and Resolution": "PASS; no blocking or high-severity findings remain.",
    }

    for title, evidence in evidence_by_subsection.items():
        body = _complete_body().replace(evidence, "<!-- required self-review evidence omitted -->")

        assert (
            f"Required subsection is empty: Contributor Self-Review / {title}"
            in pr_metadata.validate_body(body)
        )


def test_self_review_head_must_be_full_and_match_current_pr_head() -> None:
    short_head = _complete_body().replace(REVIEWED_HEAD, REVIEWED_HEAD[:12])
    wrong_head = "e" * 40

    assert (
        "Contributor Self-Review / Reviewed Head must contain a full Git commit SHA"
        in pr_metadata.validate_body(short_head)
    )
    assert (
        "Contributor Self-Review / Reviewed Head does not match the current PR head"
        in pr_metadata.validate_body(_complete_body(), expected_head_sha=wrong_head)
    )
    assert pr_metadata.validate_body(_complete_body(), expected_head_sha=REVIEWED_HEAD) == []


def test_self_review_result_uses_a_supported_verdict() -> None:
    body = _complete_body().replace("\nPASS\n\n### Findings", "\nREADY\n\n### Findings")

    assert (
        "Contributor Self-Review / Result must be exactly one of: "
        "PASS, BLOCK, HUMAN REVIEW REQUIRED"
    ) in pr_metadata.validate_body(body)


def test_event_validation_compares_self_review_with_current_pr_head(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event = {
        "pull_request": {
            "body": _complete_body(),
            "draft": False,
            "head": {"sha": REVIEWED_HEAD},
        }
    }
    event_path.write_text(json.dumps(event), encoding="utf-8")

    assert pr_metadata.main(["validate", "--event", str(event_path)]) == 0

    event["pull_request"]["head"]["sha"] = "e" * 40
    event_path.write_text(json.dumps(event), encoding="utf-8")

    assert pr_metadata.main(["validate", "--event", str(event_path)]) == 1


def test_draft_event_allows_self_review_to_remain_pending(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    event = {
        "pull_request": {
            "body": _without_self_review(_complete_body()),
            "draft": True,
            "head": {"sha": REVIEWED_HEAD},
        }
    }
    event_path.write_text(json.dumps(event), encoding="utf-8")

    assert pr_metadata.main(["validate", "--event", str(event_path)]) == 0


def test_change_category_and_risk_choices_are_enforced() -> None:
    body = (
        _complete_body()
        .replace("- [x] Model or runtime behavior", "- [ ] Model or runtime behavior")
        .replace("- [ ] High", "- [x] High")
    )

    errors = pr_metadata.validate_body(body)
    assert "Select at least one Change categories option" in errors
    assert "Select exactly one Risk level option" in errors


def test_hidden_comments_cannot_satisfy_checkboxes_or_sections() -> None:
    body = (
        _complete_body()
        .replace(
            "Fix model selection ambiguity reported in #123.",
            "<!--\nHidden text.\n## Background\nStill hidden.\n-->",
        )
        .replace("- [x] Low", "- [ ] Low\n<!--\n- [x] Low\n-->")
    )

    errors = pr_metadata.validate_body(body)

    assert "Required section is empty: Background" in errors
    assert "Select exactly one Risk level option" in errors


def test_choices_outside_required_subsections_are_ignored() -> None:
    body = (
        _complete_body()
        .replace("- [x] Model or runtime behavior", "- [ ] Model or runtime behavior")
        .replace(
            "### Change categories",
            "- [x] Public API\n\n### Change categories",
        )
        .replace("- [x] Low", "- [ ] Low")
        .replace(
            "### Risk level",
            "- [x] High\n\n### Risk level",
        )
    )

    errors = pr_metadata.validate_body(body)

    assert "Select at least one Change categories option" in errors
    assert "Select exactly one Risk level option" in errors


def test_risk_level_requires_a_rationale_beyond_the_checkbox() -> None:
    body = _complete_body().replace(
        "The change is isolated to model selection.",
        "<!-- no risk rationale -->",
    )

    assert (
        "Required subsection is empty: Notes For Future Readers / Risk level"
        in pr_metadata.validate_body(body)
    )


def test_template_and_validator_share_the_same_contract() -> None:
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for title in pr_metadata.REQUIRED_SECTIONS:
        assert f"## {title}" in template
    for title in pr_metadata.VALIDATION_SUBSECTIONS:
        assert f"### {title}" in template
    for title in pr_metadata.SELF_REVIEW_SUBSECTIONS:
        assert f"### {title}" in template
    for option in (*pr_metadata.CHANGE_CATEGORIES, *pr_metadata.RISK_LEVELS):
        assert f"- [ ] {option}" in template


def test_metadata_workflow_runs_unprivileged_exact_merge_check() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "pr-metadata.yml"
    source = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    job = workflow["jobs"]["required"]

    assert "pull_request:" in source
    assert "types: [opened, edited, synchronize, reopened, ready_for_review]" in source
    assert "pull_request_target:" not in source
    assert workflow["permissions"] == {}
    assert job["name"] == "PR Metadata / Required"
    assert job["permissions"] == {"contents": "read"}
    assert "ref: ${{ github.sha }}" in source
    assert "persist-credentials: false" in source
    assert "tools.pr_metadata validate" in source
    assert "pull-requests: write" not in source
    assert "issues: write" not in source
    assert "secrets." not in source
