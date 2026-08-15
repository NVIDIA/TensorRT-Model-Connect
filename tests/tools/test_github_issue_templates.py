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


def _body_items_by_id(form: dict) -> dict[str, dict]:
    return {
        item["id"]: item
        for item in form["body"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _checkbox_labels(item: dict) -> list[str]:
    return [option["label"] for option in item["attributes"]["options"]]


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


def test_issue_forms_preserve_reviewed_intake_contracts() -> None:
    bug_path = TEMPLATE_DIRECTORY / "bug_report.yml"
    bug = _load_yaml(bug_path)
    bug_items = _body_items_by_id(bug)
    bug_source = bug_path.read_text(encoding="utf-8")

    assert (
        "https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new"
        "?template=documentation_request.yml"
    ) in bug_source
    assert "compute capability" not in str(
        bug_items["environment"]["attributes"]
    ).lower()
    assert "behavior" not in bug_items
    for item_id in ("observed_behavior", "expected_behavior"):
        assert bug_items[item_id]["validations"]["required"] is True

    documentation = _load_yaml(TEMPLATE_DIRECTORY / "documentation_request.yml")
    documentation_items = _body_items_by_id(documentation)
    assert "priority" not in documentation_items
    assert not any(
        "runtime or hardware checks" in label
        for label in _checkbox_labels(documentation_items["terms"])
    )

    feature_path = TEMPLATE_DIRECTORY / "feature_request.yml"
    feature = _load_yaml(feature_path)
    feature_items = _body_items_by_id(feature)
    feature_source = feature_path.read_text(encoding="utf-8")
    assert "priority" not in feature_items
    assert "evidence boundary" not in feature_source
    assert not any(
        "which claims require" in label
        for label in _checkbox_labels(feature_items["terms"])
    )
