# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from html.parser import HTMLParser

import pytest

from trtmc_aiperf_qual import aiperf_metrics, campaign, execution, judge, report, report_html
from trtmc_aiperf_qual.aiperf_runner import RAW_EXPORT, AiperfRun


def record(session, side, summary, index=1):
    directory = session.out / side / f"run_{index:02d}"
    directory.mkdir(parents=True)
    (directory / "profile_export_aiperf.json").write_text(json.dumps(summary))
    (directory / RAW_EXPORT).write_text(json.dumps({
        "metadata": {"benchmark_phase": "profiling", "session_num": 0}, "status": 200,
        "payload": {"request": {"prompt": "hello"}}, "responses": [{"text": json.dumps({
            "trtmc_timing": {"model_call_ms": 2}, "trtmc_observation": {}})}]}) + "\n")
    session.record(AiperfRun(directory, 0, []), {
        "name": "task <example>|dataset", "role": "both", "gpu_busy_percent": 0, "warmup": 1,
        "expected_requests": 1, "identity": {"side": side, "mode": "eager", "precision": "fp16", "concurrency": 1}})


def test_native_exports_reach_json_markdown_and_campaign_html_without_changing_gates(tmp_path):
    out = tmp_path / "model"
    session = execution.Session(out, {}, lambda: 0, lambda observation: (), lambda mine, theirs: True)
    summary = {"request_latency": {"unit": "ms", "p50": 12.5, "p99": 30.0, "std": 1.0},
               "request_throughput": {"unit": "requests/sec", "avg": 25.0},
               "output_token_throughput": {"unit": "tokens/sec", "avg": 100.0},
               "request_error_rate": {"unit": "%", "avg": 100.0},
               "trtmc_model_call_time": {"unit": "ms", "avg": 2.0},
               "warmup_metrics": {"request_latency": {"unit": "ms", "p50": 999.0}}}
    record(session, "reference", summary)
    record(session, "candidate", {**summary, "request_error_rate": {"unit": "%", "avg": 0.0}})
    result = {"model": "model", "provenance": {}, "accuracy_source": "none", "accuracy": [],
              "performance": [{"reference_mode": "eager", "light": "green", "request": "catalog",
                               "speedup": 2.5, "speedup_interval90": [2.4, 2.6]},
                              {"reference_mode": "eager", "light": "white", "request": "dataset",
                               "kind": "natural_dataset", "gate": False, "comparable": True,
                               "pairs": 1, "matched_pairs": 1, "speedup": 2.5,
                               "natural_task_speedup": 2.5, "speedup_interval90": [2.4, 2.6]}]}
    result["verdict"] = judge.verdict(result, expected_suites=[], expected_modes=1)
    result["aiperf_metrics"] = aiperf_metrics.entries(session.batches)
    assert judge.verdict(result, expected_suites=[], expected_modes=1) == result["verdict"]
    assert all(item["gate"] is False for item in result["aiperf_metrics"])
    assert result["aiperf_metrics"][0]["metrics"] == {key: summary[key] for key in aiperf_metrics.METRICS}
    evidence = [json.loads(line) for line in (out / "execution.jsonl").read_text().splitlines()]
    assert evidence[0]["aiperf_metrics"]["metrics"]["request_latency"]["p50"] == 12.5
    report.write_report(out, result)
    saved = json.loads((out / "report.json").read_text())
    assert saved["aiperf_metrics"] == result["aiperf_metrics"] and saved["verdict"] == result["verdict"]
    assert saved["performance"] == result["performance"]
    markdown = (out / "report.md").read_text()
    assert "informational; no gate" in markdown and "12.500 ms" in markdown
    assert "task <example>\\|dataset" in markdown and "0.000 %" in markdown and "999.000" not in markdown
    assert "speedup" not in markdown.lower() and "2.50x" not in markdown and "total-time ratio" not in markdown
    rows, counts, rank = campaign.collect([tmp_path])
    summary_text, _ = campaign.summary([tmp_path])
    assert "AIPerf native client metrics (informational; no gate)" in summary_text and "25.000 requests/sec" in summary_text
    page = report_html.render(rows, counts, rank, tmp_path / "report.html").read_text()
    assert "task &lt;example&gt;|dataset" in page and "25.000 requests/sec" in page
    assert "Native eager fp16" in page and "TRTMC fp16" in page and "informational; no gate" in page
    assert "speedup" not in page.lower() and "2.50x" not in page and "total-time ratio" not in page

    class VisibleText(HTMLParser):
        def __init__(self):
            super().__init__()
            self.details = 0
            self.text = []

        def handle_starttag(self, tag, attrs):
            if tag == "details":
                self.details += 1

        def handle_endtag(self, tag):
            if tag == "details":
                self.details -= 1

        def handle_data(self, data):
            if not self.details:
                self.text.append(data)

    visible = VisibleText()
    visible.feed(page)
    text = " ".join(visible.text)
    assert all(value in text for value in ("12.500 ms", "30.000 ms", "25.000 requests/sec", "100.000 tokens/sec",
                                          "100.000 %", "0.000 %", "Native eager fp16", "TRTMC fp16"))


def test_run_percentiles_and_precisions_are_kept_separate_and_failed_attempts_are_excluded(tmp_path):
    session = execution.Session(tmp_path, {}, lambda: 0, lambda observation: (), lambda mine, theirs: True)
    record(session, "reference", {"request_latency": {"unit": "ms", "p99": 100.0}})
    with execution.session(session):
        execution.supersede(0)
    record(session, "reference", {"request_latency": {"unit": "ms", "p99": 10.0}}, index=2)
    session.batches[-1]["identity"]["precision"] = "fp32"
    record(session, "candidate", {"request_latency": {"unit": "ms", "p99": 2.0}})
    entries = aiperf_metrics.entries(session.batches)
    assert len(entries) == 2
    assert sorted(item["metrics"]["request_latency"]["p99"] for item in entries) == [2.0, 10.0]
    assert {item["precision"] for item in entries} == {"fp16", "fp32"}
    assert {item["source"] for item in entries} == {
        str(tmp_path / "reference/run_02/profile_export_aiperf.json"),
        str(tmp_path / "candidate/run_01/profile_export_aiperf.json")}


@pytest.mark.parametrize("contents", [None, "{broken", "[]"])
def test_unavailable_native_summary_stays_missing_instead_of_becoming_zero_or_a_gate(tmp_path, contents):
    path = tmp_path / "profile_export_aiperf.json"
    if contents is not None:
        path.write_text(contents)
    captured = aiperf_metrics.capture(AiperfRun(tmp_path, 0, []))
    assert captured["metrics"] == {}
    assert aiperf_metrics.cells(captured)[-len(aiperf_metrics.COLUMNS):] == ["—"] * len(aiperf_metrics.COLUMNS)


def test_nonfinite_native_metrics_are_unavailable_and_zero_error_rate_is_retained():
    extracted = aiperf_metrics.extract({"request_latency": {"unit": "ms", "p50": float("nan"), "p99": float("inf")},
                                       "request_error_rate": {"unit": "%", "avg": 0.0}})
    assert "request_latency" not in extracted
    assert extracted["request_error_rate"] == {"unit": "%", "avg": 0.0}
