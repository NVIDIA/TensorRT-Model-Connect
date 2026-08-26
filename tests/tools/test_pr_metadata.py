# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pull-request evidence validation and automatic classification."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools import pr_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]


def _complete_body() -> str:
    return """\
## Background

Fix model selection ambiguity reported in #123.

## Exit Criteria

- Qwen selection is unambiguous.
- Existing runtime math is unchanged.

## Implementation

Update the Qwen builder and its model-owned tests. There are no public API,
ABI, bundle, dependency, migration, or rollout changes.

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

`python3 -m pytest -q tests/builder/test_qwen.py`: 12 passed.

### Hardware, Environment, and Revisions

Repository head `def456`; Qwen revision `abc123`; CPU-only Ubuntu 24.04.

### Not Run / Remaining Gaps

GPU execution was not run because runtime math is unchanged.

## Notes For Future Readers

Backward compatible; no artifact rebuild is required. Review the selection
logic before the model-owned regression.

### Risk level

- [x] Low
- [ ] Medium
- [ ] High

The change is isolated to model selection.

"""


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
    labels = pr_metadata.labels_for_impact(
        {"mode": "models", "affected_models": [], "changes": []}, body
    )

    assert "Select at least one Change categories option" in errors
    assert "Select exactly one Risk level option" in errors
    assert "change:api" not in labels
    assert "risk:high" not in labels


def test_risk_level_requires_a_rationale_beyond_the_checkbox() -> None:
    body = _complete_body().replace(
        "The change is isolated to model selection.",
        "<!-- no risk rationale -->",
    )

    assert (
        "Required subsection is empty: Notes For Future Readers / Risk level"
        in pr_metadata.validate_body(body)
    )


def test_labels_combine_declared_risk_with_repository_impact() -> None:
    impact = {
        "mode": "models",
        "affected_models": ["qwen"],
        "changes": [
            {
                "classifications": [
                    {"path": "src/runtime/models/qwen/model.cpp", "kind": "model"},
                    {"path": "website/docs/qwen.md", "kind": "legal_docs"},
                ]
            }
        ],
    }
    body = _complete_body().replace("- [ ] Public API", "- [x] Public API")

    assert pr_metadata.labels_for_impact(impact, body) == {
        "change:api",
        "component:docs",
        "component:model",
        "model:qwen",
        "risk:low",
    }


def test_broad_change_uses_one_all_models_label() -> None:
    impact = {
        "mode": "all",
        "affected_models": ["albert", "qwen", "whisper"],
        "changes": [{"classifications": [{"path": "CMakeLists.txt", "kind": "platform"}]}],
    }

    labels = pr_metadata.labels_for_impact(impact, _complete_body())

    assert "impact:all-models" in labels
    assert "component:platform" in labels
    assert not {label for label in labels if label.startswith("model:")}


def test_managed_label_sync_preserves_unmanaged_labels() -> None:
    class RecordingLabels:
        def __init__(self) -> None:
            self.ensured: list[str] = []
            self.added: list[str] = []
            self.removed: list[str] = []

        def issue_labels(self, _issue_number: int) -> set[str]:
            return {"bug", "model:old", "risk:high", "component:model"}

        def ensure_label(self, label: str) -> None:
            self.ensured.append(label)

        def add_labels(self, _issue_number: int, labels: list[str]) -> None:
            self.added.extend(labels)

        def remove_label(self, _issue_number: int, label: str) -> None:
            self.removed.append(label)

    client = RecordingLabels()
    desired = {"component:model", "model:qwen", "risk:low"}

    pr_metadata.sync_managed_labels(client, 123, desired)  # type: ignore[arg-type]

    assert client.ensured == ["component:model", "model:qwen", "risk:low"]
    assert client.added == ["model:qwen", "risk:low"]
    assert client.removed == ["model:old", "risk:high"]


def test_template_and_validator_share_the_same_contract() -> None:
    template = (REPO_ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    for title in pr_metadata.REQUIRED_SECTIONS:
        assert f"## {title}" in template
    for title in pr_metadata.VALIDATION_SUBSECTIONS:
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


def test_triage_workflow_never_checks_out_or_executes_pull_request_code() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "pr-triage.yml"
    source = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    job = workflow["jobs"]["classify"]

    assert "pull_request_target:" in source
    assert workflow["permissions"] == {}
    assert job["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
    }
    assert "ref: ${{ github.event.pull_request.base.sha }}" in source
    assert "persist-credentials: false" in source
    assert "refs/remotes/trtmc/pr-head" in source
    assert "git checkout" not in source
    assert "github.event.pull_request.head.ref" not in source
    assert "tools.pr_metadata triage" in source
    assert "secrets." not in source
