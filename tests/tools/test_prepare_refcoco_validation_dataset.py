# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from tools import prepare_refcoco_validation_dataset as converter


def test_convert_refcoco_uses_official_prompt_and_pinned_source(monkeypatch, tmp_path) -> None:
    row = {
        "answer": "person in red",
        "question_id": "42",
        "image": {"bytes": b"jpeg"},
        "file_name": "image-42.jpg",
        "image_width": 640,
        "image_height": 480,
        "bbox": [0.1, 0.2, 0.7, 0.8],
    }
    monkeypatch.setattr(converter, "_iter_rows", lambda _root, _split: iter([row]))

    output = converter.convert_refcoco_rec(
        source_root=tmp_path / "source",
        output_dir=tmp_path / "unified",
        splits=["testA"],
    )

    dataset = json.loads(output.read_text(encoding="utf-8"))
    request = dataset["requests"][0]
    assert dataset["source_revision"] == converter.SOURCE_REVISION
    assert dataset["source_license_status"] == "not-declared-by-source-card"
    assert request["messages"][0]["content"][1]["text"] == (
        "Locate a single instance that matches the following description: person in red."
    )
    assert request["metadata"]["bbox_1000"] == [100.0, 200.0, 700.0, 800.0]
    assert (output.parent / request["messages"][0]["content"][0]["image"]).is_file()


def test_bbox_normalization_rejects_degenerate_box() -> None:
    try:
        converter._bbox_1000([0.2, 0.2, 0.2, 0.7])
    except ValueError as error:
        assert "no area" in str(error)
    else:
        raise AssertionError("expected degenerate RefCOCO box to be rejected")
