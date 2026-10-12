# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from trtmc_benchmark.operations import operation_for_name
from trtmc_benchmark.task_adapters import resolve_task_case
from trtmc_benchmark.types import BenchmarkError


def test_document_and_media_order_are_preserved(tmp_path):
    document = '{"questions":{"z":{},"a":{}},"state":"café"}'
    (tmp_path / "record.json").write_text(document)
    for name in ("first.png", "second.png"):
        (tmp_path / name).write_bytes(b"fixture")
    resolved = resolve_task_case(
        "structured_decision",
        {
            "inputs": {
                "document_path": "record.json",
                "image_paths": ["second.png", "first.png"],
                "video_frame_paths": [["first.png", "second.png", "first.png"]],
                "max_state_tokens": 17,
            }
        },
        tmp_path,
    )
    assert resolved.operation == "decide"
    assert resolved.request["document"] == document
    assert resolved.request["image_paths"] == [
        str(tmp_path / "second.png"),
        str(tmp_path / "first.png"),
    ]
    assert resolved.request["video_frame_paths"] == [
        [str(tmp_path / p) for p in ("first.png", "second.png", "first.png")]
    ]
    assert resolved.request["max_state_tokens"] == 17
    assert resolved.measurement.timing_scope == "public_task_call_wall"
    assert operation_for_name("decide").rate_metrics[0].observation_field == "questions"


@pytest.mark.parametrize(
    "inputs",
    [
        {},
        {"document": {}},
        {"document": "{}", "document_path": "record.json"},
        {"document": "{}", "image_paths": "file.png"},
        {"document": "{}", "video_frame_paths": [[]]},
    ],
)
def test_invalid_decision_request_fails_before_execution(tmp_path, inputs):
    with pytest.raises(BenchmarkError):
        resolve_task_case("structured_decision", {"inputs": inputs}, tmp_path)
