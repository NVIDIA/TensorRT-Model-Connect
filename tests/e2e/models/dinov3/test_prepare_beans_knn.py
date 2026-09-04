# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.models.dinov3.prepare_beans_knn import CLASS_NAMES, prepare_records


def _rows(count: int) -> list[dict]:
    return [
        {"image_bytes": f"jpeg-{index}".encode(), "label": index % len(CLASS_NAMES)}
        for index in range(count)
    ]


def test_prepare_records_emits_manifest_relative_complete_split(tmp_path: Path) -> None:
    output = tmp_path / "beans"
    entrypoint = prepare_records(
        _rows(21),
        _rows(19),
        output,
        require_official_counts=False,
    )

    dataset = json.loads(entrypoint.read_text(encoding="utf-8"))
    assert dataset["train_count"] == 21
    assert dataset["test_count"] == 19
    assert len(dataset["requests"]) == 2
    assert [request["sample_id"] for request in dataset["requests"]] == [
        "beans-test-000-015",
        "beans-test-016-018",
    ]
    bank = json.loads((output / "bank.json").read_text(encoding="utf-8"))
    query = json.loads((output / "queries/test-000.json").read_text(encoding="utf-8"))
    assert len(bank["samples"]) == 21
    assert len(query["samples"]) == 16
    assert bank["samples"][0]["image"] == "images/train/000000.jpg"
    assert query["samples"][0]["image"] == "../images/test/000000.jpg"
    manifest = json.loads((output / "manifest.sha256.json").read_text(encoding="utf-8"))
    paths = {entry["path"] for entry in manifest["entries"]}
    assert "dataset.json" in paths
    assert "bank.json" in paths
    assert "queries/test-001.json" in paths
    assert "manifest.sha256.json" not in paths
    assert len(manifest["tree_sha256"]) == 64


def test_prepare_records_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "beans"
    output.mkdir()
    (output / "owned.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_records(_rows(1), _rows(1), output, require_official_counts=False)
