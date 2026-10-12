# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from html.parser import HTMLParser

import pytest

from trtmc_aiperf_qual import aiperf_metrics, campaign, execution, judge, report, report_html
from trtmc_aiperf_qual.aiperf_runner import RAW_EXPORT, AiperfRun


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.details = 0
        self.text = []
        self.rows = 0

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            self.details += 1
        if tag == "tr" and not self.details:
            self.rows += 1

    def handle_endtag(self, tag):
        if tag == "details":
            self.details -= 1

    def handle_data(self, data):
        if not self.details:
            self.text.append(data)


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
    assert "Native mode: eager" in page and "<th>Precision</th>" in page and "informational; no gate" in page
    assert "speedup" not in page.lower() and "2.50x" not in page and "total-time ratio" not in page

    visible = VisibleText()
    visible.feed(page)
    text = " ".join(visible.text)
    assert all(value in text for value in ("12.500 ms", "30.000 ms", "25.000 requests/sec", "100.000 tokens/sec",
                                          "100.000 %", "0.000 %", "Native", "TRTMC", "fp16"))


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
    grouped = aiperf_metrics.summaries([captured])[0]
    assert aiperf_metrics.cells(grouped, range(len(aiperf_metrics.COLUMNS)))[-len(aiperf_metrics.COLUMNS):] == ["—"] * len(aiperf_metrics.COLUMNS)


def test_nonfinite_native_metrics_are_unavailable_and_zero_error_rate_is_retained():
    extracted = aiperf_metrics.extract({"request_latency": {"unit": "ms", "p50": float("nan"), "p99": float("inf")},
                                       "request_error_rate": {"unit": "%", "avg": 0.0}})
    assert "request_latency" not in extracted
    assert extracted["request_error_rate"] == {"unit": "%", "avg": 0.0}


def metric_run(side, latency, index, **settings):
    return {"workload": "example-catalog", "role": "performance", "concurrency": 1,
            "mode": "eager", "precision": "fp16", "side": side, "batch_id": index,
            "requests": 3, "aiperf_exit": 0, "metrics": {
                "request_latency": {"unit": "ms", "p50": latency, "p99": latency + 10},
                "request_throughput": {"unit": "requests/sec", "avg": 1 / latency},
                "request_error_rate": {"unit": "%", "avg": 0.0}}, **settings}


def test_repetitions_become_one_native_trtmc_pair_with_medians_and_totals():
    runs = [metric_run(side, value, index) for side, values in
            (("reference", [10, 100, 20]), ("candidate", [5, 8, 7])) for index, value in enumerate(values)]
    original = json.dumps(runs, sort_keys=True)
    panel, = aiperf_metrics.panels(runs, "example")
    native, trtmc = panel["main"]
    assert panel["label"] == "Catalog" and not panel["extras"]
    assert native["runs"] == trtmc["runs"] == 3 and native["requests"] == trtmc["requests"] == 9
    assert native["values"][0]["value"] == 20 and native["values"][1]["value"] == 30
    assert trtmc["values"][0]["value"] == 7 and trtmc["values"][1]["value"] == 17
    assert "Output token throughput" not in aiperf_metrics.headers(panel)
    page = report_html._native_metrics(runs, "example")
    visible = VisibleText()
    visible.feed(page)
    assert visible.rows == 3  # one header plus the Native/TRTMC rows
    assert "20.000 ms" in page and "Across 3/3 runs: 10.000–100.000 ms" in page
    assert "Side / mode / precision" not in page and "<th>Run</th>" not in page
    assert "medians of run p50/p99" in aiperf_metrics.NOTE
    assert json.dumps(runs, sort_keys=True) == original


def test_wan_precision_mismatch_is_explicit_and_fp32_compile_references_stay_separate():
    runs = [metric_run("reference", 900, 2, precision="fp32"),
            metric_run("candidate", 30, 1), metric_run("reference", 40, 0, precision="bf16"),
            metric_run("reference", 20, 3, precision="bf16", mode="compile")]
    panel, = aiperf_metrics.panels(runs, native_precision="bf16")
    assert [(item["side"], item["precision"]) for item in panel["main"]] == [("reference", "bf16"), ("candidate", "fp16")]
    assert len(panel["extras"]) == 2 and panel["main"][0]["values"][0]["value"] == 40
    assert "Precision differs" in aiperf_metrics.panel_note(panel)
    page = report_html._native_metrics(runs, native_precision="bf16")
    visible = VisibleText()
    visible.feed(page)
    text = " ".join(visible.text)
    assert visible.rows == 3 and "bf16" in text and "fp16" in text
    assert "fp32" not in text and "900.000 ms" not in text and "compile" not in text
    assert "fp32" in page and "900.000 ms" in page and "Native mode: compile" in page
    markdown = "\n".join(aiperf_metrics.markdown(runs, native_precision="bf16"))
    assert "Additional native settings" in markdown and "900.000 ms" in markdown


def test_different_workload_role_mode_precision_and_concurrency_never_merge():
    base = metric_run("reference", 10, 0)
    alternatives = [{"workload": "other"}, {"role": "both"}, {"mode": "compile"},
                    {"precision": "fp32"}, {"concurrency": 4}, {"side": "candidate"}]
    runs = [base, *(metric_run("reference", 100, i + 1, **changed) for i, changed in enumerate(alternatives)
                    if "side" not in changed), metric_run("candidate", 100, 6)]
    groups = aiperf_metrics.summaries(runs)
    assert len(groups) == 7 and all(item["runs"] == 1 for item in groups)
    assert sorted(item["values"][0]["value"] for item in groups) == [10, 100, 100, 100, 100, 100, 100]


def test_partial_missing_units_and_failed_runs_remain_visible_in_summaries():
    first, missing = metric_run("reference", 10, 0), metric_run("reference", 100, 1, metrics={}, aiperf_exit=1)
    panel, = aiperf_metrics.panels([first, missing])
    cells = aiperf_metrics.cells(panel["main"][0], panel["columns"])
    assert "failed runs: 1/2" in cells[0] and "10.000 ms (1/2 runs)" in cells
    assert "0.000 % (1/2 runs)" in cells
    conflict = metric_run("reference", 1, 2, metrics={"request_latency": {"unit": "s", "p50": 1}})
    panel, = aiperf_metrics.panels([first, conflict])
    assert "— (units differ)" in aiperf_metrics.cells(panel["main"][0], panel["columns"])
    missing_native, = aiperf_metrics.panels([metric_run("candidate", 5, 0)])
    assert aiperf_metrics.cells(missing_native["main"][0], missing_native["columns"])[0] == "Native"
    assert set(aiperf_metrics.cells(missing_native["main"][0], missing_native["columns"])[1:]) == {"—"}
