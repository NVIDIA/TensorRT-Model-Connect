# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

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


def test_independent_batch_requests(tmp_path):
    items = [{"document": '{"state":"first"}'}, {"document": '{"state":"second"}'}]
    result = resolve_task_case("structured_decision", {"inputs": {"requests": items}}, tmp_path)
    assert [item["document"] for item in result.request["requests"]] == [item["document"] for item in items]
    assert result.operation == "decide"


@pytest.mark.parametrize("inputs", [
    {"requests": []}, {"requests": ["{}"]},
    {"requests": [{"requests": [{"document": "{}"}]}]},
    {"requests": [{"document": "{}"}], "document": "{}"},
])
def test_reject_invalid_batch(tmp_path, inputs):
    with pytest.raises(BenchmarkError):
        resolve_task_case("structured_decision", {"inputs": inputs}, tmp_path)


def test_batch_request_throughput_counts_requests():
    from trtmc_benchmark.metrics import reduce_metrics
    observations = [
        {"runtime_e2e_wall_ms": 20.0, "requests": 3, "questions": 5, "options": 10},
        {"runtime_e2e_wall_ms": 30.0, "requests": 2, "questions": 4, "options": 8},
    ]
    metrics = reduce_metrics("decide", observations)
    assert metrics["sample_count"] == 2
    assert metrics["request_throughput_per_s"] == pytest.approx(100)
    assert metrics["requests_per_s"] == pytest.approx(100)


@pytest.mark.parametrize("profile", ["edge", "lite", "nano", "mini", "flash"])
def test_d3_text_image_manifest_passes_image_to_worker(profile):
    tests = Path(__file__).resolve().parents[4] / "families/d3/tests"
    manifest = json.loads((tests / f"manifests/d3-{profile}.json").read_text())
    case = manifest["testcases"][0]
    request = resolve_task_case("structured_decision", case, tests).request["requests"][0]
    assert request["image_paths"] == [str(tests / "fixtures/red.png")]
