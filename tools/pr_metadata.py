#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate pull-request evidence and synchronize repository-derived labels."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.model_ci import ModelCIError, calculate_impact


REQUIRED_SECTIONS = (
    "Background",
    "Exit Criteria",
    "Implementation",
    "Validation",
    "Notes For Future Readers",
)
VALIDATION_SUBSECTIONS = (
    "Commands and Results",
    "Hardware, Environment, and Revisions",
    "Not Run / Remaining Gaps",
)
CHANGE_CATEGORIES = {
    "Model or runtime behavior": None,
    "Public API": "change:api",
    "ABI": "change:abi",
    "Bundle or artifact format": "change:bundle-format",
    "Dependencies": "change:dependencies",
    "Documentation only": "change:documentation-only",
    "CI or developer tooling": "change:ci-tooling",
}
RISK_LEVELS = {"Low": "risk:low", "Medium": "risk:medium", "High": "risk:high"}
MANAGED_PREFIXES = ("model:", "component:", "risk:", "change:", "impact:")
COMPONENT_BY_CLASSIFICATION = {
    "model": "component:model",
    "unit_builder": "component:builder",
    "unit_cli": "component:cli",
    "unit_tests": "component:tests",
    "legal_docs": "component:docs",
    "platform": "component:platform",
    "ci_tooling": "component:ci",
    "unknown": "component:unknown",
}
LABEL_COLORS = {
    "model": "7057ff",
    "component": "1d76db",
    "risk:low": "2da44e",
    "risk:medium": "bf8700",
    "risk:high": "cf222e",
    "change": "d4c5f9",
    "impact": "8250df",
}
_HEADING_RE = re.compile(r"^(?P<level>#{2,3})[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\] (?P<label>.+?)[ \t]*$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class MetadataError(RuntimeError):
    """A contributor-actionable pull-request metadata failure."""


def _heading_blocks(text: str, level: int) -> dict[str, str]:
    headings = [match for match in _HEADING_RE.finditer(text) if len(match.group("level")) == level]
    blocks: dict[str, str] = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        blocks[match.group("title").strip().casefold()] = text[match.end() : end]
    return blocks


def _meaningful(text: str) -> bool:
    without_comments = _HTML_COMMENT_RE.sub("", text)
    without_checkboxes = _CHECKBOX_RE.sub("", without_comments)
    without_headings = _HEADING_RE.sub("", without_checkboxes)
    return bool(without_headings.strip())


def checked_options(body: str, options: Mapping[str, object]) -> list[str]:
    body = _HTML_COMMENT_RE.sub("", body)
    checked = {
        match.group("label").strip()
        for match in _CHECKBOX_RE.finditer(body)
        if match.group("mark").strip()
    }
    return [option for option in options if option in checked]


def validate_body(body: str) -> list[str]:
    errors: list[str] = []
    visible_body = _HTML_COMMENT_RE.sub("", body)
    sections = _heading_blocks(visible_body, 2)
    for title in REQUIRED_SECTIONS:
        content = sections.get(title.casefold())
        if content is None:
            errors.append(f"Missing required section: {title}")
        elif title != "Validation" and not _meaningful(content):
            errors.append(f"Required section is empty: {title}")

    nested_requirements = {
        "Implementation": ("Change categories",),
        "Validation": VALIDATION_SUBSECTIONS,
        "Notes For Future Readers": ("Risk level",),
    }
    for section_title, subsection_titles in nested_requirements.items():
        content = sections.get(section_title.casefold(), "")
        subsections = _heading_blocks(content, 3)
        for title in subsection_titles:
            subsection = subsections.get(title.casefold())
            if subsection is None:
                errors.append(f"Missing required subsection: {section_title} / {title}")
            elif title != "Change categories" and not _meaningful(subsection):
                errors.append(f"Required subsection is empty: {section_title} / {title}")

    implementation_subsections = _heading_blocks(sections.get("implementation", ""), 3)
    change_categories = implementation_subsections.get("change categories", "")
    selected_categories = checked_options(change_categories, CHANGE_CATEGORIES)
    if not selected_categories:
        errors.append("Select at least one Change categories option")
    note_subsections = _heading_blocks(sections.get("notes for future readers", ""), 3)
    risk_level = note_subsections.get("risk level", "")
    selected_risks = checked_options(risk_level, RISK_LEVELS)
    if len(selected_risks) != 1:
        errors.append("Select exactly one Risk level option")
    return errors


def labels_for_impact(impact: Mapping[str, object], body: str) -> set[str]:
    labels: set[str] = set()
    for change in impact.get("changes", []):
        if not isinstance(change, Mapping):
            continue
        for classification in change.get("classifications", []):
            if not isinstance(classification, Mapping):
                continue
            component = COMPONENT_BY_CLASSIFICATION.get(str(classification.get("kind", "")))
            if component:
                labels.add(component)

    if impact.get("mode") == "all":
        labels.add("impact:all-models")
    else:
        for model in impact.get("affected_models", []):
            labels.add(f"model:{model}")

    sections = _heading_blocks(_HTML_COMMENT_RE.sub("", body), 2)
    implementation_subsections = _heading_blocks(sections.get("implementation", ""), 3)
    change_categories = implementation_subsections.get("change categories", "")
    for option in checked_options(change_categories, CHANGE_CATEGORIES):
        label = CHANGE_CATEGORIES[option]
        if label:
            labels.add(label)
    note_subsections = _heading_blocks(sections.get("notes for future readers", ""), 3)
    risk_level = note_subsections.get("risk level", "")
    selected_risks = checked_options(risk_level, RISK_LEVELS)
    if len(selected_risks) == 1:
        labels.add(RISK_LEVELS[selected_risks[0]])
    return labels


def _label_spec(name: str) -> tuple[str, str]:
    if name.startswith("risk:"):
        color = LABEL_COLORS[name]
    else:
        prefix = name.split(":", maxsplit=1)[0]
        color = LABEL_COLORS[prefix]
    description = {
        "model": "Model impact derived from repository ownership metadata",
        "component": "Component impact derived from changed paths",
        "risk": "Risk declared in the pull-request template",
        "change": "Compatibility change declared in the pull-request template",
        "impact": "Broad impact derived from repository ownership metadata",
    }[name.split(":", maxsplit=1)[0]]
    return color, description


class GitHubLabels:
    """Minimal GitHub Issues labels client."""

    def __init__(
        self,
        repository: str,
        token: str,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        self.repository = repository
        self.token = token
        self.opener = opener

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        allowed_statuses: Sequence[int] = (),
    ) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{self.repository}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": "trtmc-pr-triage",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with self.opener(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if error.code in allowed_statuses:
                return {}
            detail = error.read().decode("utf-8", errors="replace")
            raise MetadataError(
                f"GitHub API {method} {path} failed: {error.code} {detail}"
            ) from error
        return json.loads(raw) if raw else {}

    def issue_labels(self, issue_number: int) -> set[str]:
        payload = self._request("GET", f"/issues/{issue_number}/labels?per_page=100")
        if not isinstance(payload, list):
            raise MetadataError("GitHub API returned an invalid issue-label response")
        return {str(item["name"]) for item in payload if isinstance(item, Mapping)}

    def ensure_label(self, name: str) -> None:
        color, description = _label_spec(name)
        self._request(
            "POST",
            "/labels",
            {"name": name, "color": color, "description": description},
            allowed_statuses=(422,),
        )

    def add_labels(self, issue_number: int, labels: Sequence[str]) -> None:
        if labels:
            self._request("POST", f"/issues/{issue_number}/labels", {"labels": list(labels)})

    def remove_label(self, issue_number: int, label: str) -> None:
        encoded = urllib.parse.quote(label, safe="")
        self._request(
            "DELETE",
            f"/issues/{issue_number}/labels/{encoded}",
            allowed_statuses=(404,),
        )


def sync_managed_labels(client: GitHubLabels, issue_number: int, desired: set[str]) -> None:
    current = client.issue_labels(issue_number)
    managed = {label for label in current if label.startswith(MANAGED_PREFIXES)}
    for label in sorted(desired):
        client.ensure_label(label)
    client.add_labels(issue_number, sorted(desired - current))
    for label in sorted(managed - desired):
        client.remove_label(issue_number, label)


def _load_event(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("pull_request"), dict):
        raise MetadataError("GitHub event does not contain pull_request metadata")
    return payload


def _pull_request_body(event: Mapping[str, object]) -> str:
    pull_request = event["pull_request"]
    assert isinstance(pull_request, Mapping)
    body = pull_request.get("body")
    return body if isinstance(body, str) else ""


def _validate(event_path: Path) -> None:
    errors = validate_body(_pull_request_body(_load_event(event_path)))
    if errors:
        for error in errors:
            print(f"::error title=PR metadata::{error}")
        raise MetadataError(f"Pull-request metadata has {len(errors)} error(s)")
    print("Pull-request metadata is complete.")


def _triage(event_path: Path, base: str, head: str) -> None:
    event = _load_event(event_path)
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    issue_number = event.get("number")
    if not repository or not token or not isinstance(issue_number, int):
        raise MetadataError("Triage requires GITHUB_REPOSITORY, GH_TOKEN, and an issue number")
    impact = calculate_impact(Path.cwd(), base, head, platform_change_policy="all")
    desired = labels_for_impact(impact, _pull_request_body(event))
    sync_managed_labels(GitHubLabels(repository, token), issue_number, desired)
    print(json.dumps({"labels": sorted(desired)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--event", type=Path, required=True)
    triage = commands.add_parser("triage")
    triage.add_argument("--event", type=Path, required=True)
    triage.add_argument("--base", required=True)
    triage.add_argument("--head", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            _validate(arguments.event)
        elif arguments.command == "triage":
            _triage(arguments.event, arguments.base, arguments.head)
    except (MetadataError, ModelCIError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
