# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared support-rule conformance cases and the support.json schema."""

from __future__ import annotations

import ast
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


# Families whose support cannot yet be written as plain support.json rules. A family
# listed here must say why; a family that only declares model identities and tasks
# with one family_support(...) call belongs in support.json, not in this list.
HAND_WRITTEN_SUPPORT = {
    "clef": "Hand-written support logic; not yet expressed as support.json rules.",
    "cosmos3": "Excludes the cosmos3_edge model type from the Cosmos3 pipeline match.",
    "eagle_vlm": "Hand-written support logic; not yet expressed as support.json rules.",
    "elf_flow": "Hand-written support logic; not yet expressed as support.json rules.",
    "gemma": "Derives the task list from which vision and audio configs are present.",
    "k2_horizon_uno": "Hand-written support logic; not yet expressed as support.json rules.",
    "lance": "Hand-written support logic; not yet expressed as support.json rules.",
    "laya": "Hand-written support logic; not yet expressed as support.json rules.",
    "lerobot_act": "Hand-written support logic; not yet expressed as support.json rules.",
    "lerobot_groot": "Hand-written support logic; not yet expressed as support.json rules.",
    "lerobot_pi05": "Hand-written support logic; not yet expressed as support.json rules.",
    "moge": "Hand-written support logic; not yet expressed as support.json rules.",
    "molmo2": "Hand-written support logic; not yet expressed as support.json rules.",
    "musicgen": "Hand-written support logic; not yet expressed as support.json rules.",
    "nemotron_voicechat": "Hand-written support logic; not yet expressed as support.json rules.",
    "qwen": "Hand-written support logic; not yet expressed as support.json rules.",
    "qwen3_5": "Hand-written support logic; not yet expressed as support.json rules.",
    "qwen3_8": "Hand-written support logic; not yet expressed as support.json rules.",
    "qwen3_vl_moe": "Hand-written support logic; not yet expressed as support.json rules.",
    "qwen_image": "Hand-written support logic; not yet expressed as support.json rules.",
    "t5": "Hand-written support logic; not yet expressed as support.json rules.",
    "yolov10": (
        "Matches the exact root JSON shape, including a model-name prefix and a non-empty names map."
    ),
}


def _is_plain_family_support(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "family_support"
    ]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return len(calls) == 1 and not functions


def test_every_family_declares_support_as_data_or_is_listed_with_a_reason() -> None:
    families = sorted(path.parent.name for path in (REPO / "families").glob("*/support.py"))
    without_rules = {
        name for name in families if not (REPO / "families" / name / "support.json").exists()
    }
    unlisted = sorted(without_rules - HAND_WRITTEN_SUPPORT.keys())
    assert not unlisted, (
        f"families without support.json: {unlisted}. Declare their support in "
        "families/<family>/support.json so the Hugging Face Hub can evaluate it."
    )
    stale = sorted(HAND_WRITTEN_SUPPORT.keys() - without_rules)
    assert not stale, f"remove families that now have support.json from the list: {stale}"
    plain = sorted(
        name
        for name in HAND_WRITTEN_SUPPORT
        if _is_plain_family_support(REPO / "families" / name / "support.py")
    )
    assert not plain, f"plain family_support(...) families must use support.json: {plain}"
    assert all(HAND_WRITTEN_SUPPORT.values())
