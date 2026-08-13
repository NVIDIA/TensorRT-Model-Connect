# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPOSITORY = Path(__file__).resolve().parents[2]
TEMPLATE_DIRECTORY = REPOSITORY / ".github" / "ISSUE_TEMPLATE"
EXPECTED_FORMS = {
    "bug_report.yml": ("bug", "Bug: "),
    "documentation_request.yml": ("documentation", "Docs: "),
    "feature_request.yml": ("enhancement", "Feature: "),
    "question.yml": ("question", "Question: "),
}
SUPPORTED_BODY_TYPES = {"checkboxes", "dropdown", "input", "markdown", "textarea"}
FORBIDDEN_TEMPLATE_TEXT = (
    "__PROJECT__",
    "__THIS PROJECT__",
    "__PROJECT_NAME__",
    "___PROJECT___",
    "jarmak-nv",
    "rapids-repo-template",
    "PLC-OSS-Template",
    "print_env.sh",
)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"{path} must contain a YAML object"
    return value


def test_issue_template_inventory_is_intentional() -> None:
    filenames = {path.name for path in TEMPLATE_DIRECTORY.iterdir() if path.is_file()}
    assert filenames == {*EXPECTED_FORMS, "config.yml"}
    assert not {name for name in filenames if "security" in name.lower()}


@pytest.mark.parametrize(("filename", "expected"), EXPECTED_FORMS.items())
def test_issue_form_structure(filename: str, expected: tuple[str, str]) -> None:
    expected_label, expected_title = expected
    form = _load_yaml(TEMPLATE_DIRECTORY / filename)

    assert isinstance(form.get("name"), str) and form["name"]
    assert isinstance(form.get("description"), str) and form["description"]
    assert form.get("title") == expected_title
    assert form.get("labels") == [expected_label]

    body = form.get("body")
    assert isinstance(body, list) and body
    assert any(item.get("type") != "markdown" for item in body)
    ids: list[str] = []
    for item in body:
        assert isinstance(item, dict)
        item_type = item.get("type")
        assert item_type in SUPPORTED_BODY_TYPES
        attributes = item.get("attributes")
        assert isinstance(attributes, dict)
        if item_type == "markdown":
            assert isinstance(attributes.get("value"), str) and attributes["value"]
            continue

        item_id = item.get("id")
        assert isinstance(item_id, str) and item_id
        ids.append(item_id)
        assert isinstance(attributes.get("label"), str) and attributes["label"]

        validations = item.get("validations", {})
        assert isinstance(validations, dict)
        if "required" in validations:
            assert isinstance(validations["required"], bool)

    assert len(ids) == len(set(ids)), f"{filename} contains duplicate field IDs"
    assert "security" not in form["labels"]

    source = (TEMPLATE_DIRECTORY / filename).read_text(encoding="utf-8")
    assert "https://www.nvidia.com/en-us/security/report-vulnerability/" in source
    assert "internal CI logs or artifacts" in source
    assert "restricted model assets" in source


def test_issue_chooser_routes_help_and_security_reports() -> None:
    config = _load_yaml(TEMPLATE_DIRECTORY / "config.yml")
    assert config.get("blank_issues_enabled") is False

    links = config.get("contact_links")
    assert isinstance(links, list)
    urls = {link["url"] for link in links}
    assert "https://nvidia.github.io/TensorRT-Model-Connect/release-support/get-help" in urls
    assert "https://www.nvidia.com/en-us/security/report-vulnerability/" in urls


def test_issue_templates_have_no_upstream_placeholders() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(TEMPLATE_DIRECTORY.glob("*.yml"))
    )
    for forbidden in FORBIDDEN_TEMPLATE_TEXT:
        assert forbidden not in combined
