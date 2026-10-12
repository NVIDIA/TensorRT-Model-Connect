# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from trtmc_aiperf_qual.generation import _answered
from trtmc_aiperf_qual.suites import Suite, request_sha


def saved_generation(out, order=(1, 2, 0), failed=None, duplicate_request=False):
    requests = [{"prompt": name} for name in ("toilet", "apple", "oranges")]
    if duplicate_request:
        requests[1] = requests[0]
    samples = [{"request": request, "request_sha": request_sha(request)} for request in requests]
    suite = Suite("geneval", "key", samples, {})
    (out / "aiperf").mkdir()
    (out / "aiperf/inputs.json").write_text(json.dumps({"data": [
        {"session_id": f"session_{index:06d}"} for index in range(len(requests))]}))
    records, raw = [], []
    for index, phase in [(0, "warmup"), *[(i, "profiling") for i in order]]:
        identifier = f"{phase}-{index}"
        records.append({"route": "/v1/tasks/generate_image", "request_id": identifier})
        raw.append({"metadata": {"benchmark_phase": phase, "conversation_id": f"session_{index:06d}",
                                 "x_request_id": identifier}, "status": 500 if index == failed else 200,
                    "payload": {"request": requests[index]}})
    (out / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (out / "aiperf/profile_export_raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    return suite


def test_generated_files_follow_requests_after_warmup_and_export_reordering(tmp_path):
    suite = saved_generation(tmp_path)
    outputs = _answered(tmp_path, suite)
    assert [record["request_id"] for _, record in outputs] == ["profiling-0", "profiling-1", "profiling-2"]
    assert [path.name for path, _ in outputs] == ["profiling-0", "profiling-1", "profiling-2"]


def test_duplicate_requests_preserve_sample_identity_after_export_reordering(tmp_path):
    suite = saved_generation(tmp_path, duplicate_request=True)
    outputs = _answered(tmp_path, suite)
    assert [record["request_id"] for _, record in outputs] == ["profiling-0", "profiling-1", "profiling-2"]


@pytest.mark.parametrize("field", ["conversation", "request"])
def test_generation_rejects_a_mismatched_conversation_or_request(tmp_path, field):
    suite = saved_generation(tmp_path)
    path = tmp_path / "aiperf/profile_export_raw.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if field == "conversation":
        rows[1]["metadata"]["conversation_id"] = "unknown"
    else:
        rows[1]["payload"]["request"] = {"prompt": "another request"}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _answered(tmp_path, suite) is None


@pytest.mark.parametrize("order, failed", [((1, 2), None), ((1, 1, 0), None), ((1, 2, 0), 1)])
def test_incomplete_duplicate_or_failed_generation_is_not_reused(tmp_path, order, failed):
    suite = saved_generation(tmp_path, order, failed)
    assert _answered(tmp_path, suite) is None
