#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a self-contained HTML report when isolated model proof cannot run."""

from __future__ import annotations

import argparse
import html
import json
import os
import stat
from pathlib import Path
from typing import Sequence


_DIAGNOSTIC_FILES = (
    "host-error.log",
    "ci-image.log",
    "python-profiles-prepare.log",
    "python-profile-download.log",
    "console.log",
    "projection.stderr.log",
    "projection.json",
    "configure.log",
    "build.log",
)
_MAX_DIAGNOSTIC_CHARS = 16_000
_MAX_DIAGNOSTIC_BYTES = _MAX_DIAGNOSTIC_CHARS * 4


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _diagnostics(root: Path) -> list[tuple[str, str]]:
    excerpts: list[tuple[str, str]] = []
    for filename in _DIAGNOSTIC_FILES:
        path = root / filename
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            continue
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                continue
            offset = max(0, metadata.st_size - _MAX_DIAGNOSTIC_BYTES)
            os.lseek(descriptor, offset, os.SEEK_SET)
            payload = os.read(descriptor, _MAX_DIAGNOSTIC_BYTES)
        except OSError:
            continue
        finally:
            os.close(descriptor)
        text = payload.decode("utf-8", errors="replace")[-_MAX_DIAGNOSTIC_CHARS:]
        if text.strip():
            excerpts.append((filename, text))
    return excerpts


def write_fallback_report(
    artifacts_dir: Path,
    *,
    model: str,
    revision: str,
    suite: str,
    outcome: str,
    phase: str,
    exit_code: int | None,
    details: Sequence[str] = (),
    preserve_rich_report: bool = False,
) -> bool:
    """Write or update a fallback report; return False when a rich report wins."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    status_path = artifacts_dir / "model-proof-status.json"
    report_path = artifacts_dir / "model-proof-report.html"
    status = _load_json(status_path)
    report_kind = str(status.get("report_kind") or "")
    if (
        preserve_rich_report
        and report_path.is_file()
        and report_kind != "workflow_fallback"
    ):
        return False

    status.update({
        "schema_version": 1,
        "report_kind": (
            "workflow_fallback" if phase.startswith("workflow") else "host_fallback"
        ),
        "model": model,
        "source_revision": revision,
        "suite": suite,
        "outcome": outcome,
        "phase": phase,
    })
    if exit_code is not None:
        status["exit_code"] = exit_code
    steps = status.setdefault("steps", {})
    if not isinstance(steps, dict):
        steps = {}
        status["steps"] = steps
    steps["host_setup"] = {
        "status": "running" if outcome == "running" else "failed",
        "evidence": ", ".join(_DIAGNOSTIC_FILES),
    }
    if details:
        status["details"] = list(details)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in (
            ("Model", model),
            ("Pinned revision", revision),
            ("Suite", suite),
            ("Outcome", outcome),
            ("Failure phase", phase),
            ("Exit code", exit_code if exit_code is not None else "pending"),
        )
    )
    detail_html = "".join(f"<li>{html.escape(item)}</li>" for item in details)
    logs = "".join(
        f"<details><summary>{html.escape(name)}</summary>"
        f"<pre>{html.escape(text)}</pre></details>"
        for name, text in _diagnostics(artifacts_dir)
    )
    message = (
        "The isolated model validation is being prepared. This placeholder will "
        "be replaced by the evidence-rich report."
        if outcome == "running"
        else "Validation did not enter or complete the isolated report path. "
        "This fallback report preserves the available setup diagnostics."
    )
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Isolated Model Proof: {html.escape(model)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}
.notice{{padding:1rem;background:#fef2f2;border:1px solid #fca5a5;border-radius:.5rem}}
pre{{white-space:pre-wrap;max-height:420px;overflow:auto;background:#111827;color:#e5e7eb;padding:1rem}}
</style></head><body><h1>Isolated Model Proof: {html.escape(model)}</h1>
<div class="notice"><strong>{html.escape(message)}</strong></div>
<h2>Proof Context</h2><table>{rows}</table>
{f'<h2>Workflow State</h2><ul>{detail_html}</ul>' if detail_html else ''}
<h2>Diagnostics</h2>{logs or '<p>No diagnostic log was produced.</p>'}
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--outcome", choices=("running", "failed"), required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--detail", action="append", default=[])
    parser.add_argument("--preserve-rich-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_fallback_report(
        args.artifacts_dir,
        model=args.model,
        revision=args.revision,
        suite=args.suite,
        outcome=args.outcome,
        phase=args.phase,
        exit_code=args.exit_code,
        details=args.detail,
        preserve_rich_report=args.preserve_rich_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
