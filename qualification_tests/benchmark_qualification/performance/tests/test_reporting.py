# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from qualification_tests.benchmark_qualification.performance import matrix as perf_matrix


@pytest.mark.parametrize(
    "schema, row_key, reference_key",
    [("v1", "cases", "baseline"), ("v2", "rows", "reference")],
)
def test_performance_report_preserves_historical_measurements(
    tmp_path, schema, row_key, reference_key
):
    scope = "model_call_wall" if schema == "v1" else "public_task_call_wall"
    results = {
        "schema_version": "trtmc.perf-matrix/" + schema,
        "status": "completed",
        "selected_entry_ids": ["example.generate"],
        row_key: [
            {
                "id": "example.generate",
                "model": "example",
                "operation": "generate",
                "status": "green",
                "candidate": {"samples_ms": [8.0, 10.0, 12.0], "timing_scope": scope},
                reference_key: {
                    "samples_ms": [12.0, 14.0, 16.0],
                    "measurement_policy": {"timing_scope": "public_operation_call_wall"},
                },
                "commands": {
                    "candidate": {"argv": ["trtmc-bench", "run", "--model", "example"]}
                },
            }
        ],
    }
    original = json.dumps(results)
    (tmp_path / "results.json").write_text(original)
    assert perf_matrix.main(["report", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    document = (tmp_path / "report.html").read_text()
    assert report["source_schema_version"] == results["schema_version"]
    assert report["rows"][0][reference_key] == results[row_key][0][reference_key]
    assert "p50: 10.000 ms" in document
    assert "p95: 11.800 ms" in document
    assert scope in document
    assert "public_operation_call_wall" in document
    assert "trtmc-bench" in document
    assert 'id="filter"' in document
    assert (tmp_path / "results.json").read_text() == original
    if schema == "v1":
        with pytest.raises(perf_matrix.PerfMatrixError, match="unsupported results schema"):
            perf_matrix._load_results(tmp_path)
