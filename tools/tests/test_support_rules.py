# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared support-rule conformance cases and the support.json schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from tensorrt_model_connect.model_support import (
    ModelMetadata,
    decide_support,
    support_from_json,
)


REPO = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO / "tools/support_rules.schema.json").read_text(encoding="utf-8"))
CONFORMANCE = json.loads(
    (REPO / "tools/support_rules_conformance.json").read_text(encoding="utf-8")
)


def _family(tmp_path: Path, name: str, document: dict) -> object:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "support.json").write_text(json.dumps(document), encoding="utf-8")
    return support_from_json(folder / "support.py")


def _metadata(model: dict) -> ModelMetadata:
    return ModelMetadata(
        config=model.get("config", {}),
        model_index=model.get("model_index", {}),
        files=tuple(model.get("files", ())),
        adapter_config=model.get("adapter_config", {}),
        repo_id=model.get("repo_id", ""),
        revision=model.get("revision", ""),
    )


def test_schema_is_a_valid_json_schema() -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("case", CONFORMANCE["cases"], ids=lambda case: case["name"])
def test_conformance_case(tmp_path: Path, case: dict) -> None:
    for document in case["families"].values():
        jsonschema.validate(document, SCHEMA)
    families = {name: _family(tmp_path, name, doc) for name, doc in case["families"].items()}

    decision = decide_support(_metadata(case["model"]), families)

    expected = case["expected"]
    assert decision.status == expected["status"]
    assert decision.family == expected.get("family")
    if "support" in expected:
        support = expected["support"]
        assert decision.support is not None
        assert list(decision.support.tasks) == support["tasks"]
        assert decision.support.default_task == support["default_task"]
        assert decision.support.default_precision == support.get("default_precision", "fp32")


@pytest.mark.parametrize("case", CONFORMANCE["invalid"], ids=lambda case: case["name"])
def test_invalid_rule_files_fail_schema_and_parser(tmp_path: Path, case: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(case["document"], SCHEMA)
    with pytest.raises(ValueError):
        _family(tmp_path, "invalid", case["document"])


def test_every_family_support_json_matches_the_schema() -> None:
    documents = sorted((REPO / "families").glob("*/support.json"))
    for path in documents:
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), SCHEMA)
        support_from_json(path.with_suffix(".py"))
