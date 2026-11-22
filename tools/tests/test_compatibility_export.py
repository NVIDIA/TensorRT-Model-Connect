# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import compatibility_export


RULES = {
    "schema_version": 1,
    "rules": [
        {
            "if": {"/config/model_type": {"in": ["alpha"]}},
            "then": {"tasks": ["generation"], "default_task": "generation"},
        }
    ],
}


def _repo(tmp_path: Path) -> Path:
    for name, text in (("alpha", json.dumps(RULES, indent=2) + "\n"), ("beta", json.dumps(RULES))):
        folder = tmp_path / "families" / name
        folder.mkdir(parents=True)
        (folder / "support.json").write_text(text, encoding="utf-8")
    (tmp_path / "families/legacy").mkdir()
    (tmp_path / "families/legacy/support.py").write_text("", encoding="utf-8")
    return tmp_path


def test_export_copies_every_family_rule_file_verbatim_with_versions(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    exported = compatibility_export.build_export(repo, commit="abc123")

    alpha_bytes = (repo / "families/alpha/support.json").read_bytes()
    assert exported["schema_version"] == 1
    assert list(exported["families"]) == ["alpha", "beta"]
    assert exported["families"]["alpha"] == {
        "sha256": hashlib.sha256(alpha_bytes).hexdigest(),
        "rules": RULES,
    }
    assert exported["rules_version"]["commit"] == "abc123"
    assert len(exported["rules_version"]["sha256"]) == 64


def test_rules_version_changes_only_when_a_rule_file_changes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    first = compatibility_export.build_export(repo, commit="abc123")
    same = compatibility_export.build_export(repo, commit="def456")
    (repo / "families/beta/support.json").write_text(json.dumps(RULES) + " ", encoding="utf-8")
    changed = compatibility_export.build_export(repo, commit="def456")

    assert same["rules_version"]["sha256"] == first["rules_version"]["sha256"]
    assert changed["rules_version"]["sha256"] != first["rules_version"]["sha256"]
    assert changed["families"]["alpha"] == first["families"]["alpha"]
    assert changed["families"]["beta"]["sha256"] != first["families"]["beta"]["sha256"]


def test_cli_writes_the_export(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    output = tmp_path / "out/compatibility.json"

    assert compatibility_export.main(
        ["--repository", str(repo), "--commit", "abc123", "--output", str(output)]
    ) == 0

    assert json.loads(output.read_text(encoding="utf-8"))["families"]["beta"]["rules"] == RULES
