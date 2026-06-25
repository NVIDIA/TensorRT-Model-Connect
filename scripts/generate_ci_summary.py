#!/usr/bin/env python3
"""Generate a GitHub Actions Markdown summary for CI artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


_STATUS_ORDER = ("fail", "error", "skip", "pass")
_PASS_STATUSES = {"pass", "passed", "success", "succeeded"}
_PYTEST_TO_RESULT_STATUS = {
    "PASSED": "pass",
    "XPASS": "pass",
    "FAILED": "fail",
    "ERROR": "error",
    "SKIPPED": "skip",
    "XFAIL": "skip",
}
_TEST_CASE_RE = re.compile(r"(?:test_e2e|test_model_e2e)\[([^\]]+)\]")
_CONSOLE_OUTCOME_RE = re.compile(
    r"(?:tests/test_e2e\.py::test_e2e|"
    r"tests/e2e/models/[^\s:]+::test_model_e2e)\[([^\]]+)\]\s+"
    r"(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b(.*)"
)
_METRIC_PRIORITY = (
    "logit_cosine_p5",
    "token_agreement_rate",
    "miou",
    "psnr",
    "ssim",
    "mel_distance",
    "pixel_accuracy",
    "normalized_text_edit_distance",
)


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", r"\|")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _load_results(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for result_path in sorted(artifacts_dir.rglob("result.json")):
        try:
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: skipping {result_path}: {exc}", file=sys.stderr)
    return results


def _e2e_root_from_artifacts_dir(artifacts_dir: Path) -> Path:
    if artifacts_dir.name == "artifacts":
        return artifacts_dir.parent
    return artifacts_dir


def _extract_case_name(text: str) -> str:
    match = _TEST_CASE_RE.search(text)
    return match.group(1) if match else ""


def _clean_pytest_reason(reason: str) -> str:
    text = reason.strip()
    while text.startswith("(") and text.endswith(")") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _junit_files(e2e_root: Path) -> list[Path]:
    worker_files = sorted(e2e_root.glob("junit-gpu*.xml"))
    if worker_files:
        return worker_files
    merged = e2e_root / "junit.xml"
    return [merged] if merged.is_file() else []


def _load_pytest_outcomes(e2e_root: Path) -> dict[str, dict[str, str]]:
    outcomes: dict[str, dict[str, str]] = {}

    for xml_path in _junit_files(e2e_root):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            print(f"WARNING: skipping {xml_path}: {exc}", file=sys.stderr)
            continue
        for testcase in root.iter("testcase"):
            case_name = _extract_case_name(
                " ".join(
                    str(testcase.attrib.get(key, ""))
                    for key in ("classname", "name")
                )
            )
            if not case_name:
                continue
            status = "PASSED"
            reason = ""
            source = xml_path.name
            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")
            if error is not None:
                status = "ERROR"
                reason = error.attrib.get("message", "") or (error.text or "")
            elif failure is not None:
                status = "FAILED"
                reason = failure.attrib.get("message", "") or (failure.text or "")
            elif skipped is not None:
                skip_type = skipped.attrib.get("type", "")
                status = "XFAIL" if skip_type == "pytest.xfail" else "SKIPPED"
                reason = skipped.attrib.get("message", "") or (skipped.text or "")
            outcomes[case_name] = {
                "pytest_status": status,
                "reason": _clean_pytest_reason(reason),
                "source": source,
            }

    for log_path in sorted(e2e_root.glob("console-*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"WARNING: skipping {log_path}: {exc}", file=sys.stderr)
            continue
        for line in lines:
            match = _CONSOLE_OUTCOME_RE.search(line)
            if not match:
                continue
            case_name, status, rest = match.groups()
            reason = _clean_pytest_reason(rest.split("[", 1)[0])
            if status in {"XPASS", "XFAIL"} or case_name not in outcomes:
                outcomes[case_name] = {
                    "pytest_status": status,
                    "reason": reason,
                    "source": log_path.name,
                }

    return outcomes


def _merge_pytest_outcomes(
    results: list[dict[str, Any]],
    outcomes: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        item = dict(result)
        case_name = str(item.get("case_name") or "")
        if case_name:
            seen.add(case_name)
        if case_name in outcomes:
            item["_pytest_outcome"] = outcomes[case_name]
        merged.append(item)

    for case_name, outcome in sorted(outcomes.items()):
        if case_name in seen:
            continue
        status = _PYTEST_TO_RESULT_STATUS.get(
            outcome.get("pytest_status", ""), "error")
        merged.append(
            {
                "case_name": case_name,
                "status": status,
                "failure_type": "pytest_failed"
                if status in {"fail", "error"} else None,
                "case_config": {},
                "stages": {
                    "pytest": {
                        "status": status,
                        "message": outcome.get("reason", ""),
                        "metrics": {},
                    }
                },
                "_summary_only": True,
                "_pytest_outcome": outcome,
            }
        )
    return merged


def _status(result: dict[str, Any]) -> str:
    outcome = result.get("_pytest_outcome")
    if isinstance(outcome, dict):
        pytest_status = str(outcome.get("pytest_status") or "")
        if pytest_status in {"XFAIL", "XPASS"}:
            return _PYTEST_TO_RESULT_STATUS[pytest_status]
    return str(result.get("status") or "error").lower()


def _key_metric(result: dict[str, Any]) -> str:
    stages = result.get("stages", {}) or {}
    for stage_data in stages.values():
        metrics = stage_data.get("metrics", {}) if isinstance(stage_data, dict) else {}
        for key in _METRIC_PRIORITY:
            if key in metrics:
                metric = metrics[key]
                value = metric.get("value", metric) if isinstance(metric, dict) else metric
                return f"{key}={_format_value(value)}"
    return ""


def _total_time_seconds(result: dict[str, Any]) -> float | None:
    timing = result.get("timing", {}) or {}
    if not isinstance(timing, dict) or not timing:
        return None
    total = 0.0
    saw_value = False
    for value in timing.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
        saw_value = True
    return total if saw_value else None


def _total_time(result: dict[str, Any]) -> str:
    total = _total_time_seconds(result)
    return "" if total is None else f"{total:.1f}s"


def _failure_note(result: dict[str, Any]) -> str:
    pytest_outcome = result.get("_pytest_outcome")
    if result.get("_summary_only") and isinstance(pytest_outcome, dict):
        status = pytest_outcome.get("pytest_status", "pytest")
        reason = pytest_outcome.get("reason", "")
        return f"{status}: {reason}" if reason else str(status)
    failure_type = result.get("failure_type")
    if failure_type:
        return str(failure_type)
    for stage_name, stage_data in (result.get("stages", {}) or {}).items():
        if not isinstance(stage_data, dict):
            continue
        status = str(stage_data.get("status", "")).lower()
        if status and status not in _PASS_STATUSES:
            message = stage_data.get("message")
            if message:
                return f"{stage_name}: {message}"
            return str(stage_name)
    return ""


def _case_row(result: dict[str, Any], include_failure: bool = False) -> str:
    config = result.get("case_config", {}) or {}
    cols = [
        _md(result.get("case_name", "unknown")),
        _md(config.get("family", "")),
        _md(config.get("task_strategy", "")),
        _md(_status(result)),
        _md(_key_metric(result)),
        _md(_total_time(result)),
    ]
    if include_failure:
        cols.insert(4, _md(_failure_note(result)))
    return "| " + " | ".join(cols) + " |"


def _pytest_status(result: dict[str, Any]) -> str:
    outcome = result.get("_pytest_outcome")
    if not isinstance(outcome, dict):
        return ""
    return str(outcome.get("pytest_status") or "")


def _pytest_reason(result: dict[str, Any]) -> str:
    outcome = result.get("_pytest_outcome")
    if not isinstance(outcome, dict):
        return ""
    return str(outcome.get("reason") or "")


def _pytest_row(result: dict[str, Any]) -> str:
    cols = [
        _md(result.get("case_name", "unknown")),
        _md(_pytest_status(result)),
        _md(_status(result)),
        _md(_pytest_reason(result)),
    ]
    return "| " + " | ".join(cols) + " |"


def _render_table(headers: list[str], rows: list[str]) -> list[str]:
    if not rows:
        return []
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *rows,
    ]


def _sort_key(result: dict[str, Any]) -> tuple[int, float, str]:
    status = _status(result)
    status_rank = _STATUS_ORDER.index(status) if status in _STATUS_ORDER else 1
    total = _total_time_seconds(result)
    return (status_rank, -(total or 0.0), str(result.get("case_name", "")))


def render_summary(
    *,
    results: list[dict[str, Any]],
    mode: str,
    report_path: Path,
    html_artifact_name: str,
    full_artifact_name: str,
    run_url: str,
    max_rows: int,
) -> str:
    lines: list[str] = [f"## TensorRT-Model-Connect CI Summary ({mode})", ""]

    if run_url:
        lines.append(f"Artifacts: [open this workflow run]({run_url})")
    else:
        lines.append("Artifacts: open this workflow run")
    lines.append(f"HTML report artifact: `{html_artifact_name}`")
    lines.append(f"Full debug artifact: `{full_artifact_name}`")
    if report_path.is_file():
        lines.append("HTML report contents: `e2e_report.html`")
    else:
        lines.append("HTML report contents: not generated for this run")
    lines.append("")

    if not results:
        lines.append("No E2E `result.json` files were found for this run.")
        lines.append("")
        return "\n".join(lines)

    counts: dict[str, int] = {}
    for result in results:
        counts[_status(result)] = counts.get(_status(result), 0) + 1

    lines.append("### E2E Result Counts")
    count_rows = [
        f"| {_md(status)} | {counts[status]} |"
        for status in sorted(counts, key=lambda s: (_STATUS_ORDER.index(s) if s in _STATUS_ORDER else 1, s))
    ]
    lines.extend(_render_table(["Status", "Count"], count_rows))
    lines.append("")

    failures = [
        r for r in results
        if _status(r) not in _PASS_STATUSES and _pytest_status(r) != "XFAIL"
    ]
    if failures:
        lines.append("### Failures")
        rows = [_case_row(r, include_failure=True) for r in sorted(failures, key=_sort_key)[:max_rows]]
        lines.extend(
            _render_table(
                ["Model", "Family", "Task", "Status", "Failure", "Key Metric", "Time"],
                rows,
            )
        )
        if len(failures) > max_rows:
            lines.append(f"\nShowing {max_rows} of {len(failures)} non-passing cases.")
        lines.append("")

    waived = [r for r in results if _pytest_status(r) in {"XFAIL", "XPASS"}]
    if waived:
        lines.append("### Pytest Waive Outcomes")
        rows = [
            _pytest_row(r)
            for r in sorted(
                waived,
                key=lambda item: (
                    _pytest_status(item) != "XPASS",
                    str(item.get("case_name", "")),
                ),
            )
        ]
        lines.extend(_render_table(
            ["Model", "Pytest Status", "Result Status", "Reason"], rows))
        lines.append("")

    timed = [r for r in results if _total_time_seconds(r) is not None]
    if timed:
        lines.append("### Slowest E2E Cases")
        rows = [
            _case_row(r)
            for r in sorted(timed, key=lambda item: _total_time_seconds(item) or 0.0, reverse=True)[:10]
        ]
        lines.extend(_render_table(["Model", "Family", "Task", "Status", "Key Metric", "Time"], rows))
        lines.append("")

    lines.append("### All E2E Model Status")
    rows = [_case_row(r) for r in sorted(results, key=_sort_key)]
    lines.extend(_render_table(["Model", "Family", "Task", "Status", "Key Metric", "Time"], rows))
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a GitHub Actions CI summary.")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--html-artifact-name", required=True)
    parser.add_argument("--full-artifact-name", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--max-rows", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = _load_results(args.artifacts_dir)
    outcomes = _load_pytest_outcomes(_e2e_root_from_artifacts_dir(args.artifacts_dir))
    results = _merge_pytest_outcomes(results, outcomes)
    print(
        render_summary(
            results=results,
            mode=args.mode,
            report_path=args.report_path,
            html_artifact_name=args.html_artifact_name,
            full_artifact_name=args.full_artifact_name,
            run_url=args.run_url,
            max_rows=args.max_rows,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
