# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared public report contract for Accuracy and Performance qualification."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "trtmc.qualification-report/v1"
RGB_RESULTS = ("green", "yellow", "red")
TERMINAL_RESULTS = (*RGB_RESULTS, "white")
CASE_STATES = ("pending", "running", "terminal")
ASSET_DIRECTORY = Path(__file__).with_name("qualification_report_assets")


class QualificationReportError(ValueError):
    """The public qualification report violates its accounting contract."""


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def outcome_accounting(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return mutually exclusive progress and outcome counts for selected rows."""

    progress = {state: 0 for state in CASE_STATES}
    outcomes = {result: 0 for result in TERMINAL_RESULTS}
    for row in results:
        state = str(row.get("state", "terminal"))
        if state not in progress:
            raise QualificationReportError(f"unknown case state: {state}")
        result = row.get("result")
        if state == "terminal":
            if result not in outcomes:
                raise QualificationReportError(
                    f"terminal case must have one result in {TERMINAL_RESULTS}: {result!r}"
                )
            outcomes[str(result)] += 1
        elif result is not None:
            raise QualificationReportError(
                f"{state} case cannot have a traffic-light result: {result!r}"
            )
        progress[state] += 1

    selected = len(results)
    comparable = sum(outcomes[result] for result in RGB_RESULTS)
    if progress["terminal"] != comparable + outcomes["white"]:
        raise QualificationReportError("terminal accounting is not mutually exclusive")
    if selected != sum(progress.values()):
        raise QualificationReportError("selected accounting is not mutually exclusive")
    return {
        "selected": selected,
        "comparable": comparable,
        "operational_coverage_percent": _percentage(comparable, selected),
        "progress": progress,
        "outcomes": outcomes,
        "invariants": {
            "selected": "pending + running + terminal",
            "terminal": "green + yellow + red + white",
            "comparable": "green + yellow + red",
        },
        "definitions": {
            "green": {
                "class": "valid-comparison",
                "label": "Meets the target",
                "denominator": "comparable",
            },
            "yellow": {
                "class": "valid-comparison",
                "label": "Valid comparison in the review band",
                "denominator": "comparable",
            },
            "red": {
                "class": "valid-comparison",
                "label": "Valid comparison that misses the target",
                "denominator": "comparable",
            },
            "white": {
                "class": "coverage-gap",
                "label": "No valid comparison",
                "denominator": "selected",
            },
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_bytes_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _artifact_slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "case"


def _validate_artifact_href(output: Path, href: Any) -> None:
    relative = Path(str(href))
    if relative.is_absolute() or ".." in relative.parts:
        raise QualificationReportError(
            f"report artifact href must be relative to the report root: {href!r}"
        )
    artifact = output / relative
    if not artifact.is_file():
        raise QualificationReportError(f"report artifact does not exist: {href!r}")
    if artifact.is_symlink():
        raise QualificationReportError(
            f"report artifact must be a self-contained regular file: {href!r}"
        )


def _materialize_missing_failure_log(
    output: Path,
    row: dict[str, Any],
) -> None:
    if row.get("state") != "terminal" or row.get("result") != "white":
        return
    debug = row.setdefault("debug", {})
    if not isinstance(debug, dict):
        raise QualificationReportError("case debug evidence must be a JSON object")
    logs = debug.setdefault("logs", [])
    if not isinstance(logs, list):
        raise QualificationReportError("case debug.logs must be a JSON array")
    if logs:
        return
    issue = row.get("issue")
    if not isinstance(issue, Mapping):
        raise QualificationReportError(
            "white case must provide structured issue evidence"
        )
    stage = _artifact_slug(issue.get("stage", "report"))
    path = (
        output
        / "artifacts"
        / _artifact_slug(row.get("id", "case"))
        / "logs"
        / f"{stage}.log"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "case_id": row.get("id"),
        "issue": dict(issue),
        "commands": row.get("commands", {}),
        "reproduce": row.get("reproduce", {}),
    }
    path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logs.append(
        {
            "label": f"{str(issue.get('stage', 'report')).replace('-', ' ').title()} diagnostic",
            "href": path.relative_to(output).as_posix(),
        }
    )


def _validate_public_results(output: Path, results: Sequence[dict[str, Any]]) -> None:
    for row in results:
        if not str(row.get("id", "")).strip():
            raise QualificationReportError(
                "every report result must have a non-empty id"
            )
        _materialize_missing_failure_log(output, row)
        if row.get("result") == "white":
            issue = row.get("issue")
            required = {"priority", "stage", "domain", "code", "message"}
            if not isinstance(issue, Mapping) or not required <= issue.keys():
                raise QualificationReportError(
                    f"white case {row['id']!r} must provide {sorted(required)}"
                )
        debug = row.get("debug", {})
        if not isinstance(debug, Mapping):
            raise QualificationReportError("case debug evidence must be a JSON object")
        for collection in ("logs", "command_artifacts"):
            records = debug.get(collection, [])
            if not isinstance(records, list):
                raise QualificationReportError(
                    f"case debug.{collection} must be a JSON array"
                )
            for record in records:
                if not isinstance(record, Mapping) or not record.get("href"):
                    raise QualificationReportError(
                        f"case debug.{collection} entries must contain href"
                    )
                _validate_artifact_href(output, record["href"])
        sample_differences = row.get("sample_differences")
        if isinstance(sample_differences, Mapping) and sample_differences.get("href"):
            _validate_artifact_href(output, sample_differences["href"])


def _html_shell(title: str) -> str:
    safe_title = (
        title.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<link rel="stylesheet" href="assets/qualification-report.css">
</head><body data-report="report.json">
<main id="report-root"><p class="meta">Loading report.json…</p></main>
<script src="assets/qualification-report.js" defer></script>
</body></html>
"""


def materialize_report(
    output: Path,
    *,
    report_kind: str,
    title: str,
    identity: Mapping[str, Any],
    run: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically publish report.json and a data-only HTML renderer shell."""

    if report_kind not in {"accuracy", "performance"}:
        raise QualificationReportError(f"unknown report kind: {report_kind}")
    public_results = [deepcopy(dict(row)) for row in results]
    _validate_public_results(output, public_results)
    public_run = deepcopy(dict(run))
    environment_path = output / "artifacts" / "run" / "environment.json"
    _write_json_atomic(environment_path, public_run)
    public_run["environment_href"] = environment_path.relative_to(output).as_posix()
    report: dict[str, Any] = {
        "$schema": "assets/qualification-report.schema.json",
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_kind": report_kind,
        "identity": {"title": title, **dict(identity)},
        "run": public_run,
        "accounting": outcome_accounting(public_results),
        "results": public_results,
    }
    if metadata:
        for name, value in metadata.items():
            if name in report:
                raise QualificationReportError(f"metadata cannot replace {name!r}")
            report[name] = value

    json_path = output / "report.json"
    html_path = output / "report.html"
    _write_json_atomic(json_path, report)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in (
        "qualification-report.css",
        "qualification-report.js",
        "qualification-report.schema.json",
    ):
        _write_bytes_if_changed(assets / name, (ASSET_DIRECTORY / name).read_bytes())
    _write_bytes_if_changed(html_path, _html_shell(title).encode("utf-8"))
    return json_path, html_path, report
