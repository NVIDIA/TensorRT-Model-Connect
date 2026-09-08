#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate required pull-request evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


REQUIRED_SECTIONS = (
    "Background",
    "Exit Criteria",
    "Implementation",
    "Validation",
    "Contributor Self-Review",
    "Notes For Future Readers",
)
VALIDATION_SUBSECTIONS = (
    "Commands and Results",
    "Hardware, Environment, and Revisions",
    "Not Run / Remaining Gaps",
)
SELF_REVIEW_SUBSECTIONS = (
    "Method",
    "Reviewed Head",
    "Result",
    "Findings and Resolution",
)
SELF_REVIEW_RESULTS = ("PASS", "BLOCK", "HUMAN REVIEW REQUIRED")
CHANGE_CATEGORIES = (
    "Model or runtime behavior",
    "Public API",
    "ABI",
    "Bundle or artifact format",
    "Dependencies",
    "Documentation only",
    "CI or developer tooling",
)
RISK_LEVELS = ("Low", "Medium", "High")
_HEADING_RE = re.compile(r"^(?P<level>#{2,3})[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\] (?P<label>.+?)[ \t]*$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FULL_GIT_SHA_RE = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{40}|[0-9a-f]{64})(?![0-9a-f])", re.IGNORECASE)


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


def checked_options(body: str, options: Sequence[str]) -> list[str]:
    body = _HTML_COMMENT_RE.sub("", body)
    checked = {
        match.group("label").strip()
        for match in _CHECKBOX_RE.finditer(body)
        if match.group("mark").strip()
    }
    return [option for option in options if option in checked]


def validate_body(
    body: str,
    *,
    expected_head_sha: str | None = None,
    require_self_review: bool = True,
) -> list[str]:
    errors: list[str] = []
    visible_body = _HTML_COMMENT_RE.sub("", body)
    sections = _heading_blocks(visible_body, 2)
    required_sections = tuple(
        title for title in REQUIRED_SECTIONS if require_self_review or title != "Contributor Self-Review"
    )
    for title in required_sections:
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
    if require_self_review:
        nested_requirements["Contributor Self-Review"] = SELF_REVIEW_SUBSECTIONS
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
    if require_self_review:
        self_review_subsections = _heading_blocks(sections.get("contributor self-review", ""), 3)
        reviewed_head = self_review_subsections.get("reviewed head")
        if reviewed_head is not None and _meaningful(reviewed_head):
            match = _FULL_GIT_SHA_RE.search(_HTML_COMMENT_RE.sub("", reviewed_head))
            if match is None:
                errors.append("Contributor Self-Review / Reviewed Head must contain a full Git commit SHA")
            elif expected_head_sha is not None and match.group(0).casefold() != expected_head_sha.casefold():
                errors.append("Contributor Self-Review / Reviewed Head does not match the current PR head")
        result = self_review_subsections.get("result")
        if result is not None and _meaningful(result):
            normalized_result = _HTML_COMMENT_RE.sub("", result).strip().strip("`").strip().upper()
            if normalized_result not in SELF_REVIEW_RESULTS:
                choices = ", ".join(SELF_REVIEW_RESULTS)
                errors.append(f"Contributor Self-Review / Result must be exactly one of: {choices}")
    return errors


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


def _pull_request_head_sha(event: Mapping[str, object]) -> str:
    pull_request = event["pull_request"]
    assert isinstance(pull_request, Mapping)
    head = pull_request.get("head")
    if not isinstance(head, Mapping):
        raise MetadataError("GitHub event does not contain pull_request head SHA")
    sha = head.get("sha")
    if not isinstance(sha, str):
        raise MetadataError("GitHub event does not contain pull_request head SHA")
    return sha


def _pull_request_is_draft(event: Mapping[str, object]) -> bool:
    pull_request = event["pull_request"]
    assert isinstance(pull_request, Mapping)
    draft = pull_request.get("draft")
    if not isinstance(draft, bool):
        raise MetadataError("GitHub event does not contain pull_request draft state")
    return draft


def _validate(event_path: Path) -> None:
    event = _load_event(event_path)
    errors = validate_body(
        _pull_request_body(event),
        expected_head_sha=_pull_request_head_sha(event),
        require_self_review=not _pull_request_is_draft(event),
    )
    if errors:
        for error in errors:
            print(f"::error title=PR metadata::{error}")
        raise MetadataError(f"Pull-request metadata has {len(errors)} error(s)")
    print("Pull-request metadata is complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--event", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        _validate(arguments.event)
    except (MetadataError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
