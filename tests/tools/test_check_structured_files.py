# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from tools import check_structured_files


def test_valid_structured_files_parse(tmp_path: Path) -> None:
    files = {
        "value.json": '{"name": "demo", "values": [1, 2]}\n',
        "value.toml": 'name = "demo"\nvalues = [1, 2]\n',
        "value.yaml": "name: demo\nvalues:\n  - 1\n  - 2\n",
        "multi.yml": "name: first\n---\nname: second\n",
    }

    for name, content in files.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        assert check_structured_files.validate_file(path) is None


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"name": "first", "name": "second"}\n', encoding="utf-8")

    failure = check_structured_files.validate_file(path)

    assert failure is not None
    assert "duplicate JSON key: 'name'" in failure


def test_malformed_toml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text('name = "unterminated\n', encoding="utf-8")

    failure = check_structured_files.validate_file(path)

    assert failure is not None
    assert str(path) in failure


def test_malformed_yaml_reports_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("value: [1, 2\n", encoding="utf-8")

    failure = check_structured_files.validate_file(path)

    assert failure is not None
    assert str(path) in failure
