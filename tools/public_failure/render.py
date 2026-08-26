# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render validated public-failure-v1 data as one self-contained HTML file."""

from __future__ import annotations

import html
from typing import Mapping

from .contract import validate_public_failure


STYLES = """
:root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #202020; background: #fff; font-size: 14px; line-height: 1.45; }
main { width: min(1120px, calc(100% - 40px)); margin: 0 auto; padding: 32px 0; }
.report-header { padding: 18px 0 16px; border-top: 5px solid #c62828;
  border-bottom: 1px solid #8c8c8c; }
.product { margin: 0 0 7px; color: #555; font-size: 12px; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase; }
.title-row { display: flex; gap: 18px; align-items: center; justify-content: space-between; }
h1 { margin: 0; font-size: 27px; font-weight: 600; line-height: 1.2; }
.status { display: inline-block; padding: 5px 10px; border: 1px solid #9f1f1f;
  color: #8f1818; background: #fff; font-size: 12px; font-weight: 700;
  letter-spacing: .06em; }
.notice { margin: 18px 0; padding: 10px 12px; border-left: 3px solid #686868;
  color: #4d4d4d; background: #f5f5f5; }
h2 { margin: 26px 0 8px; font-size: 17px; font-weight: 600; }
table { width: 100%; border-collapse: collapse; }
.run-meta { border-top: 1px solid #b4b4b4; }
.run-meta th, .run-meta td { padding: 8px 10px; border-bottom: 1px solid #d4d4d4;
  text-align: left; vertical-align: top; }
.run-meta th { width: 180px; color: #4b4b4b; background: #f3f3f3; font-weight: 600; }
.table-wrap { width: 100%; overflow-x: auto; border-top: 2px solid #555; }
.failure-table { min-width: 900px; }
.failure-table th, .failure-table td { padding: 9px 10px; border-right: 1px solid #d4d4d4;
  border-bottom: 1px solid #bdbdbd; text-align: left; vertical-align: top; }
.failure-table th:last-child, .failure-table td:last-child { border-right: 0; }
.failure-table th { color: #333; background: #ededed; font-size: 12px; font-weight: 700; }
.failure-table tbody tr:nth-child(even) { background: #fafafa; }
.failure-index { width: 44px; text-align: center !important; }
.classification { width: 185px; }
.location { width: 175px; }
.test-id { min-width: 270px; overflow-wrap: anywhere; }
.evidence { min-width: 220px; }
.secondary { display: block; margin-top: 3px; color: #5f5f5f; font-size: 12px; }
.withheld { color: #5f5f5f; font-style: italic; }
.omitted { margin: 9px 0 0; color: #555; }
footer { margin-top: 28px; padding-top: 10px; border-top: 1px solid #aaa;
  color: #555; font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
@media (max-width: 600px) {
  main { width: min(100% - 24px, 1120px); padding-top: 18px; }
  .title-row { display: block; }
  .status { margin-top: 12px; }
  .run-meta th { width: 120px; }
}
""".strip()


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_number(value: object) -> str:
    return format(float(value), ".8g")


def _failure_row(failure: Mapping[str, object], index: int) -> str:
    metric = failure.get("metric")
    if isinstance(metric, Mapping):
        evidence = (
            f"<code>{_escape(metric['name'])}</code>"
            f'<span class="secondary">Observed: {_format_number(metric["observed"])}; '
            f"Requirement: {_escape(metric['operator'])} "
            f"{_format_number(metric['threshold'])}</span>"
        )
    else:
        evidence = '<span class="withheld">Details withheld</span>'
    return (
        "<tr>"
        f'<td class="failure-index">{index}</td>'
        '<td class="classification">'
        f"<strong>{_escape(failure['failure_class'])}</strong>"
        f'<span class="secondary">{_escape(failure["reason_code"])}</span></td>'
        '<td class="location">'
        f"{_escape(failure['public_stage'])}"
        f'<span class="secondary">{_escape(failure["model"])} · '
        f"{_escape(failure['backend'])} · {_escape(failure['gpu_type'])}</span></td>"
        f'<td class="test-id"><code>{_escape(failure["test_id"])}</code></td>'
        f'<td class="evidence">{evidence}</td>'
        "</tr>"
    )


def render_failure_report(report: Mapping[str, object]) -> bytes:
    """Validate and render one deterministic, script-free HTML document."""
    validate_public_failure(report)
    failures = report["failures"]
    rows = "".join(_failure_row(failure, index) for index, failure in enumerate(failures, start=1))
    if not rows:
        rows = (
            '<tr><td colspan="5" class="withheld">No structured failure details were '
            "safe to disclose.</td></tr>"
        )
    omitted = int(report["omitted_failure_count"])
    omitted_note = (
        f'<p class="omitted">{omitted} additional failure(s) were omitted.</p>' if omitted else ""
    )
    status = "FAILED" if report["result"] == "failure" else "ERROR"
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; script-src 'none'; connect-src 'none'; font-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'">
  <title>TRTMC Protected CI failure</title>
  <style>{STYLES}</style>
</head>
<body>
<main>
  <header class="report-header">
    <p class="product">TensorRT Model Connect / Protected CI</p>
    <div class="title-row">
      <h1>TRTMC Protected CI failure report</h1>
      <span class="status">{status}</span>
    </div>
  </header>
  <p class="notice">This report contains only approved structured fields. Raw logs and internal diagnostics are not included.</p>
  <h2>Run identification</h2>
  <table class="run-meta">
    <tbody>
      <tr><th scope="row">Repository</th><td>{_escape(report["repository"])}</td></tr>
      <tr><th scope="row">Pull request</th><td>#{_escape(report["pr_number"])}</td></tr>
      <tr><th scope="row">Head commit</th><td><code>{_escape(report["head_sha"])}</code></td></tr>
      <tr><th scope="row">Tested revision</th><td><code>{_escape(report["tested_revision"])}</code> ({_escape(report["tested_revision_kind"])})</td></tr>
      <tr><th scope="row">Run attempt</th><td>{_escape(report["run_attempt"])}</td></tr>
      <tr><th scope="row">Generated at</th><td>{_escape(report["generated_at"])}</td></tr>
      <tr><th scope="row">Disclosure policy</th><td>{_escape(report["policy_version"])}</td></tr>
    </tbody>
  </table>
  <h2>Failure summary</h2>
  <div class="table-wrap">
    <table class="failure-table">
      <thead><tr><th class="failure-index">#</th><th>Classification</th><th>Location</th><th>Test</th><th>Evidence</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  {omitted_note}
  <footer>Report ID: <code>{_escape(report["report_id"])}</code></footer>
</main>
</body>
</html>
"""
    return document.encode("utf-8")
