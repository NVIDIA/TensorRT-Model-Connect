# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one portable Community GPU report without changing execution verdicts."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import gzip
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile

from tools import e2e_report
from tools.community_gpu_ci import _checked_summary
from tools.community_gpu_lease import validate_report_receipt

MAX_HTML_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNPACKED_BYTES = 40 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 34 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_INDEX_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_ASSET_TOTAL = 24 * 1024 * 1024
MAX_EXECUTION_LOG_BYTES = 1024 * 1024
MAX_CASES = 512
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}")
SHA = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".wav", ".flac", ".mp3", ".ogg", ".mp4", ".webm"}
FILES = MEDIA | {
    ".npy",
    ".npz",
    ".f32",
    ".i32",
    ".u8",
    ".json",
    ".jsonl",
    ".txt",
    ".log",
    ".csv",
    ".cif",
    ".yaml",
    ".a3m",
    ".b2rq",
    ".raw",
    ".ppm",
}
_JS = """function filterCases(){const q=document.getElementById('search').value.toLowerCase();const s=document.getElementById('status').value;let n=0;document.querySelectorAll('.case').forEach(e=>{e.hidden=!(e.dataset.name.includes(q)&&(!s||e.dataset.executionStatus===s));if(!e.hidden)n++;});document.getElementById('visible-count').textContent=n+' cases shown';document.getElementById('no-results').hidden=n!==0;}document.getElementById('search').addEventListener('input',filterCases);document.getElementById('status').addEventListener('change',filterCases);"""


class ReportError(ValueError):
    """Report inputs are invalid or exceed their public evidence bounds."""


def _escape(value: object) -> str:
    # JSON can contain lone surrogates; escape only their displayed spelling.
    text = str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")
    return html.escape(text, quote=True)


def _json(raw: bytes, limit: int = MAX_JSON_BYTES) -> dict:
    if len(raw) > limit:
        raise ReportError("JSON exceeds the report size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReportError("Duplicate report JSON key")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ReportError("Non-finite report JSON number")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReportError("Invalid report JSON") from error
    if not isinstance(value, dict):
        raise ReportError("Report JSON must be an object")
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 24 or nodes > 200000:
            raise ReportError("Report JSON nesting or item limit exceeded")
        if isinstance(item, (list, dict)):
            pending.extend(
                (child, depth + 1) for child in (item.values() if isinstance(item, dict) else item)
            )
        elif isinstance(item, float) and not math.isfinite(item):
            raise ReportError("Non-finite report number")
    return value


def _parts(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if (
        not value
        or len(value) > 1024
        or path.is_absolute()
        or "\\" in value
        or any(ord(c) < 32 for c in value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ReportError("Evidence path must remain within its root")
    return path.parts


def _read(root: Path, relative: str, limit: int, *, tail: bool = False) -> bytes:
    """Open each component without following links, including a concurrently swapped link."""
    parts = _parts(relative)
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in (*root.absolute().parts[1:], *parts[:-1]):
            next_directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (info.st_size > limit and not tail)
            ):
                raise ReportError("Evidence must be a bounded regular file")
            if tail and info.st_size > limit:
                os.lseek(descriptor, info.st_size - limit, os.SEEK_SET)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ReportError("Evidence exceeds its size limit")
            return data
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _metadata(value: dict) -> dict:
    value = _json(json.dumps(value, allow_nan=False).encode(), MAX_INDEX_BYTES)
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(value.get("repository", "")))
        or any(
            type(value.get(key)) is not int or value[key] < 1
            for key in ("run_id", "run_attempt", "pr_number")
        )
        or any(
            not isinstance(value.get(key), str) or not SHA.fullmatch(value[key])
            for key in ("workflow_sha", "source_sha", "head_sha", "base_sha")
        )
        or value.get("lane") not in {"dev", "stable"}
        or any(type(value.get(key)) is not bool for key in ("cleanup_confirmed", "no_allocation"))
        or type(value.get("diagnostic_only", False)) is not bool
        or value.get("failure_class")
        not in {"none", "pr_failure", "infra_failure", "plan_rejected", "cpu_failure"}
        or value.get("gpu_status", "run") not in {"run", "not_started", "not_required"}
        or value.get("cpu_status")
        not in {
            "success",
            "failure",
            "cancelled",
            "timed_out",
            "skipped",
            "in_progress",
            "queued",
            "unknown",
        }
    ):
        raise ReportError("Trusted run metadata is invalid")
    return value


def _archive_member(name: str) -> bool:
    parts = _parts(name)
    if parts in {("summary.json",), ("evidence-index.json",)}:
        return True
    return (
        len(parts) in {4, 5}
        and parts[0] == "evidence"
        and SAFE_NAME.fullmatch(parts[1]) is not None
        and SAFE_NAME.fullmatch(parts[2]) is not None
        and (
            (len(parts) == 4 and parts[3] == "evidence.json")
            or (
                len(parts) == 5
                and parts[3] == "artifacts"
                and PurePosixPath(parts[4]).suffix.lower() in FILES
            )
        )
    )


@contextmanager
def _archive_root(path: Path | None):
    """Unpack only bounded regular evidence members, never tar links or PR HTML."""
    issues = []
    with tempfile.TemporaryDirectory(prefix="trtmc-report-archive-") as temporary:
        root = Path(temporary)
        try:
            if path is None:
                raise ReportError("missing archive")
            packed = _read(path.parent, path.name, MAX_ARCHIVE_BYTES)
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                raw = stream.read(MAX_UNPACKED_BYTES + 1)
            if len(raw) > MAX_UNPACKED_BYTES:
                raise ReportError("Evidence archive exceeds the unpacked limit")
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
                seen = set()
                member_bytes = 0
                for member in archive:
                    if (
                        len(seen) >= 514
                        or member.name in seen
                        or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                        or member.sparse is not None
                    ):
                        raise ReportError("Evidence archive contains invalid members")
                    seen.add(member.name)
                    if not _archive_member(member.name) or not 0 <= member.size <= MAX_JSON_BYTES:
                        raise ReportError("Evidence archive contains an unsupported path or size")
                    member_bytes += member.size
                    if member_bytes > MAX_ARCHIVE_MEMBER_BYTES:
                        raise ReportError("Evidence archive exceeds its total member size limit")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ReportError("Evidence archive member is unavailable")
                    destination = root.joinpath(*_parts(member.name))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source, destination.open("xb") as output:
                        output.write(source.read(MAX_JSON_BYTES + 1))
        except (OSError, ValueError, EOFError, tarfile.TarError):
            issues.append(
                "Detailed evidence archive is missing or invalid; execution outcomes are retained."
            )
            yield None, issues
        else:
            yield root, issues


def _index(root: Path | None, metadata: dict) -> tuple[dict, list[str]]:
    if root is None:
        return {}, ["Per-case evidence was not collected."]
    try:
        value = _json(_read(root, "evidence-index.json", MAX_INDEX_BYTES), MAX_INDEX_BYTES)
        if (
            value.get("schema_version") != 1
            or type(value.get("schema_version")) is not int
            or value.get("source_revision") != metadata["source_sha"]
            or str(value.get("workflow_run_id")) != str(metadata["run_id"])
            or str(value.get("workflow_run_attempt")) != str(metadata["run_attempt"])
            or not isinstance(value.get("cases"), list)
            or len(value["cases"]) > MAX_CASES
        ):
            raise ReportError("Evidence index belongs to a different source or run")
        entries = {}
        for entry in value["cases"]:
            if not isinstance(entry, dict):
                raise ReportError("Invalid evidence index case")
            key = (entry.get("family"), entry.get("case"))
            if (
                any(not isinstance(item, str) or not SAFE_NAME.fullmatch(item) for item in key)
                or key in entries
            ):
                raise ReportError("Duplicate or invalid evidence index identity")
            entries[key] = entry
        return entries, []
    except (OSError, ValueError):
        return {}, [
            "Evidence index is missing, invalid, or bound to another source/run; details were not used."
        ]


def _media_signature(extension: str, data: bytes) -> bool:
    signatures = {
        ".png": b"\x89PNG\r\n\x1a\n",
        ".jpg": b"\xff\xd8\xff",
        ".jpeg": b"\xff\xd8\xff",
        ".flac": b"fLaC",
        ".ogg": b"OggS",
        ".webm": b"\x1aE\xdf\xa3",
    }
    if extension in signatures:
        return data.startswith(signatures[extension])
    if extension in {".wav", ".webp"}:
        return data.startswith(b"RIFF") and data[8:12] == (
            b"WAVE" if extension == ".wav" else b"WEBP"
        )
    if extension == ".gif":
        return data[:6] in {b"GIF87a", b"GIF89a"}
    if extension == ".mp4":
        return data[4:8] == b"ftyp"
    if extension == ".mp3":
        return data.startswith(b"ID3") or (
            len(data) > 1 and data[0] == 255 and data[1] & 224 == 224
        )
    return False


def _copy_assets(
    entry: dict, root: Path, prefix: str, destination: Path, budget: list[int]
) -> tuple[set[str], str, list[str]]:
    descriptors = entry.get("artifacts", [])
    if not isinstance(descriptors, list) or len(descriptors) > 512:
        return set(), "", ["Artifact inventory is invalid."]
    copied, downloads, issues = set(), [], []
    for item in descriptors:
        try:
            if not isinstance(item, dict):
                raise ReportError("invalid asset")
            path = item.get("path", "")
            parts = _parts(path)
            relative = str(PurePosixPath(path).relative_to(prefix))
            if (
                len(parts) != 5
                or parts[3] != "artifacts"
                or relative in copied
                or PurePosixPath(path).suffix.lower() not in FILES
                or type(item.get("size_bytes")) is not int
                or not 0 <= item["size_bytes"] <= min(MAX_ASSET_BYTES, budget[0])
                or not SHA256.fullmatch(str(item.get("sha256", "")))
            ):
                raise ReportError("invalid asset")
            data = _read(root, path, min(MAX_ASSET_BYTES, budget[0]))
            if (
                len(data) != item["size_bytes"]
                or hashlib.sha256(data).hexdigest() != item["sha256"]
            ):
                raise ReportError("asset identity changed")
            extension = PurePosixPath(path).suffix.lower()
            if extension in MEDIA and not _media_signature(extension, data):
                raise ReportError("media type mismatch")
            target = destination.joinpath(*_parts(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            budget[0] -= len(data)
            copied.add(relative)
            # A rich preview may omit duplicate or context-only media. Preserve
            # each verified original even when no preview is selected for it.
            uri = "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")
            downloads.append(
                f'<li><a download="{_escape(parts[-1])}" href="{uri}">{_escape(relative)}</a> ({len(data)} bytes)</li>'
            )
        except (OSError, ValueError, TypeError):
            issues.append(
                "A recorded artifact was unavailable, invalid, or beyond the inline limit."
            )
    download_html = (
        (
            "<details><summary>Download recorded data files</summary><ul>"
            + "".join(downloads)
            + "</ul></details>"
        )
        if downloads
        else ""
    )
    return copied, download_html, issues


def _case_detail(
    root: Path | None,
    entry: dict | None,
    family: str,
    case: str,
    metadata: dict,
    destination: Path,
    budget: list[int],
) -> tuple[dict, str, list[str]]:
    if root is None or entry is None or entry.get("status") not in {"collected", "limited"}:
        return {}, "", ["Detailed evidence was not collected for this case."]
    prefix = f"evidence/{family}/{case}"
    try:
        if entry.get("path") != prefix + "/evidence.json" or not SHA256.fullmatch(
            str(entry.get("sha256", ""))
        ):
            raise ReportError("wrong evidence path")
        raw = _read(root, entry["path"], MAX_JSON_BYTES)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ReportError("evidence digest mismatch")
        data = _json(raw)
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("family") != family
            or data.get("case") != case
            or data.get("source_revision") != metadata["source_sha"]
            or data.get("workflow_run_attempt") != metadata["run_attempt"]
        ):
            raise ReportError("stale case evidence")
        copied, downloads, issues = _copy_assets(entry, root, prefix, destination, budget)
        data["artifacts"] = [
            item
            for item in data.get("artifacts", [])
            if isinstance(item, dict)
            and item.get("path") in copied
            and PurePosixPath(item["path"]).suffix.lower() in MEDIA
        ]
        if entry["status"] == "limited":
            issues.append("Some case evidence exceeded collection limits.")
        return data, downloads, issues
    except (OSError, ValueError, TypeError):
        return (
            {},
            "",
            ["Detailed case evidence is invalid, changed, or bound to another source/attempt."],
        )


def _case_card(
    row: dict,
    case: str,
    status: str,
    data: dict,
    root: Path,
    index: int,
    downloads: str,
    issues: list[str],
) -> str:
    data = dict(data)
    previous = data.get("status")
    if previous is not None and previous != status:
        issues.append(
            "Recorded detail status differs from the coordinator; the coordinator verdict is shown."
        )
    data.update(family=row["family"], case=case, status=status)
    existing = data.get("issues", [])
    data["issues"] = (
        [x for x in existing if isinstance(x, str)] if isinstance(existing, list) else []
    )
    data["issues"].extend(issues)
    if issues:
        data["evidence_status"] = "partial"
    if status != "passed" and row.get("evidence"):
        data["coordinator_failure"] = row["evidence"]
    labels = {"passed": "Passed", "failed": "Failed", "not_run": "Not run", "skipped": "Skipped"}
    assessment = {
        "kind": "passed"
        if status == "passed"
        else "failed"
        if status == "failed"
        else "unverified",
        "label": labels.get(status, "Not run"),
        "summary": f"{row['family']} / {case}: coordinator result {status}; phase {row.get('phase', 'unknown')}; classification {row.get('failure_class') or 'none'}.",
    }
    try:
        body = e2e_report._content(
            data, root, [MAX_ASSET_TOTAL], index, True, assessment=assessment
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError):
        issues.append(
            "Recorded detail fields could not be rendered; coordinator outcome is retained."
        )
        minimal = {
            "family": row["family"],
            "case": case,
            "status": status,
            "evidence_status": "partial",
            "issues": [
                "Recorded detail fields could not be rendered; coordinator outcome is retained."
            ],
        }
        body = e2e_report._content(minimal, root, [0], index, True, assessment=assessment)
    if downloads:
        before, closing, after = body.rpartition("</section>")
        return before + downloads + closing + after
    return body


def _source_identity(metadata: dict) -> str:
    source_url = f"https://github.com/{metadata['repository']}/commit/{metadata['source_sha']}"
    run_url = f"https://github.com/{metadata['repository']}/actions/runs/{metadata['run_id']}"
    return f'<details><summary>Run and source identity</summary><p><a href="{_escape(run_url)}">Run {metadata["run_id"]}, attempt {metadata["run_attempt"]}</a> · {_escape(metadata["lane"])} · PR #{metadata["pr_number"]}</p><p>Tested merge: <a href="{_escape(source_url)}">{metadata["source_sha"]}</a></p><p>PR head: {metadata["head_sha"]}</p><p>Base: {metadata["base_sha"]}</p><p>Trusted CI: {metadata["workflow_sha"]}</p></details>'


def _diagnostic_overview(metadata: dict) -> str:
    released = metadata["cleanup_confirmed"]
    state = (
        "Independent recovery confirmed release"
        if released
        else "Resource recovery remains unconfirmed"
    )
    rows = (
        ("CPU", metadata["cpu_status"]),
        ("Resource owner", metadata.get("owner_result") or "unknown"),
        ("Normal report upload", metadata.get("report_upload_result") or "unknown"),
        ("Independent recovery", metadata.get("recovery_result") or "unknown"),
    )
    table = "".join(
        f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>" for label, value in rows
    )
    no_allocation = (
        "<p>This attempt did not request an allocation.</p>"
        if metadata["no_allocation"] and released
        else ""
    )
    return (
        '<section class="run-overview"><h1>Community GPU CI</h1>'
        "<h2>The GPU report is incomplete</h2>"
        "<p>The normal GPU report could not be verified. No GPU pass is inferred; inspect the run logs.</p>"
        f'<p class="{("passed" if released else "failed")}">{state}</p>'
        + no_allocation
        + "<p>This diagnostic report contains no resource recovery authority. The original operational journals remain necessary for recovery.</p>"
        f'<div class="table-scroll"><table><tbody>{table}</tbody></table></div></section>'
        + _source_identity(metadata)
    )


def _overview(summary: dict | None, metadata: dict, receipt: dict) -> str:
    rows = summary["families"] if summary is not None else []
    gpu_status = metadata.get("gpu_status", "run")
    if metadata["failure_class"] == "plan_rejected":
        headline = "GPU plan rejected before allocation"
    elif gpu_status == "not_required":
        headline = "GPU testing was not required for this plan"
    elif gpu_status == "not_started":
        headline = "GPU testing did not start"
    elif summary is None:
        headline = "GPU execution results are unavailable"
    elif summary["passed"] and summary["complete"]:
        headline = "Selected GPU tests passed"
    else:
        headline = "GPU validation failed or remains incomplete"
    released = metadata["cleanup_confirmed"]
    if receipt["leases"]:
        released = released and all(
            lease.get("cleanup_confirmed") is True
            and lease.get("phase") in {"deleted", "unallocated"}
            for lease in receipt["leases"]
        )
    if metadata["no_allocation"]:
        release = (
            "No allocation in this attempt"
            if released
            else "No allocation in this attempt; earlier allocation recovery remains unconfirmed"
        )
    elif released and all(lease.get("phase") == "unallocated" for lease in receipt["leases"]):
        release = "Resource release confirmed: no instance was allocated"
    else:
        release = "VM release confirmed" if released else "VM release is not confirmed"
    lines = [
        f'<section class="run-overview"><h1>Community GPU CI</h1><h2>{_escape(headline)}</h2>',
        f"<p>CPU: <strong>{_escape(metadata['cpu_status'])}</strong> · Classification: <strong>{_escape(metadata['failure_class'])}</strong></p>",
        f'<p class="{("passed" if released else "failed")}">{_escape(release)}</p>',
        '<p class="note">Release confirmation comes from the resource owner. The final workflow also requires its independent cleanup check.</p>',
        '<p class="note">Execution outcomes come from the coordinator. Recorded assertions and outputs are explanatory evidence, not replacement passing criteria.</p>',
        '<div class="table-scroll"><table class="family-results"><thead><tr><th>Family</th><th>Outcome</th><th>Stage</th><th>Classification</th><th>E2E cases</th><th>Evidence</th></tr></thead><tbody>',
    ]
    if metadata["failure_class"] == "plan_rejected":
        lines.insert(
            2,
            "<p>Split this PR into smaller PRs with at most five directly changed or added families. GPU tests did not run.</p>",
        )
    elif summary is None and gpu_status == "run":
        lines.insert(
            2,
            "<p>Preparation or execution ended without a valid coordinator receipt; inspect the run logs linked below.</p>",
        )
    for row in rows:
        cases = row.get("cases", {})
        counts = ", ".join(
            f"{sum(v == state for v in cases.values())} {state}"
            for state in ("passed", "failed", "not_run", "skipped")
            if state in cases.values()
        )
        values = (
            row["family"],
            row["status"],
            row.get("phase", "unknown"),
            row.get("failure_class") or "none",
            counts or "No runnable cases",
            row.get("evidence", ""),
        )
        lines.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in values) + "</tr>")
    lines.append("</tbody></table></div></section>")
    lines.append(_source_identity(metadata))
    return "".join(lines)


def _execution_log(text: str | None) -> str:
    if text is None:
        return '<details><summary>Execution log</summary><p class="note">Execution log is unavailable in this report; inspect the original Actions run logs.</p></details>'
    raw = text.encode("utf-8", errors="replace")
    truncated = len(raw) > MAX_EXECUTION_LOG_BYTES
    bounded = raw[-MAX_EXECUTION_LOG_BYTES:].decode("utf-8", errors="replace")
    notice = (
        '<p class="note">Log truncated: only the final 1 MiB is included.</p>' if truncated else ""
    )
    return (
        "<details><summary>Execution log</summary>"
        + notice
        + "<pre>"
        + _escape(bounded)
        + "</pre></details>"
    )


def render_report(
    summary: dict | None,
    run_metadata: dict,
    evidence_root: Path | None,
    recovery_receipt: dict | None,
    *,
    issues: list[str] | None = None,
    execution_log: str | None = None,
) -> str:
    """Keep authoritative outcomes and exact recovery authority separate from case details."""
    metadata = _metadata(run_metadata)
    diagnostic = metadata.get("diagnostic_only", False)
    receipt = None
    if diagnostic:
        if recovery_receipt is not None:
            raise ReportError("A diagnostic report cannot contain recovery authority")
    else:
        receipt = _json(json.dumps(recovery_receipt, allow_nan=False).encode(), MAX_INDEX_BYTES)
        validate_report_receipt(
            receipt,
            metadata["repository"],
            metadata["run_id"],
            metadata["run_attempt"],
            metadata["workflow_sha"],
        )
        if not receipt["leases"] and not metadata["no_allocation"]:
            raise ReportError("An empty recovery journal requires trusted no-allocation evidence")
    notices = list(issues or [])
    try:
        checked = (
            _checked_summary(_json(json.dumps(summary, allow_nan=False).encode(), MAX_INDEX_BYTES))
            if summary is not None
            else None
        )
    except (ValueError, TypeError):
        checked = None
    if diagnostic or metadata.get("gpu_status", "run") != "run":
        checked = None
    if checked is None:
        notices.append("No valid coordinator GPU summary is available; no GPU pass is inferred.")
    entries, index_issues = ({}, []) if diagnostic else _index(evidence_root, metadata)
    notices.extend(index_issues)
    receipt_json = (
        json.dumps(receipt, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    overview = (
        _diagnostic_overview(metadata) if diagnostic else _overview(checked, metadata, receipt)
    )
    log_html = _execution_log(execution_log)
    reserved_bytes = len((overview + receipt_json + log_html).encode()) + 4 * 1024 * 1024
    cards, asset_budget, detail_bytes = [], [MAX_ASSET_TOTAL], 0
    with tempfile.TemporaryDirectory(prefix="trtmc-report-render-") as temporary:
        for row in checked["families"] if checked is not None else []:
            for case, status in row.get("cases", {}).items():
                if len(cards) >= MAX_CASES:
                    notices.append(
                        "Case detail count exceeded the report limit; the family summary remains complete."
                    )
                    break
                root = Path(temporary) / str(len(cards))
                root.mkdir()
                data, downloads, detail_issues = _case_detail(
                    evidence_root,
                    entries.get((row["family"], case)),
                    row["family"],
                    case,
                    metadata,
                    root,
                    asset_budget,
                )
                card = _case_card(
                    row, case, status, data, root, len(cards), downloads, detail_issues
                )
                if detail_bytes + len(card.encode()) > MAX_HTML_BYTES - reserved_bytes:
                    card = _case_card(
                        row,
                        case,
                        status,
                        {},
                        root,
                        len(cards),
                        "",
                        ["Detailed evidence exceeded the report size limit."],
                    )
                    detail_issues.append("Detailed evidence exceeded the report size limit.")
                if detail_issues:
                    notices.append(
                        "Some case details or media are missing, rejected, or limited; the case cards explain."
                    )
                cards.append(card)
                detail_bytes += len(card.encode())
    script_hash = base64.b64encode(hashlib.sha256(_JS.encode()).digest()).decode()
    csp = f"default-src 'none'; img-src data:; media-src data:; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"
    notice_html = "".join(f"<li>{_escape(message)}</li>" for message in dict.fromkeys(notices))
    journal = (
        f'<script id="trtmc-community-recovery" type="application/json">{receipt_json}</script>'
        if not diagnostic
        else ""
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="{_escape(csp)}"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Community GPU CI report</title><style>{e2e_report._CSS}</style></head><body>
{overview}<details><summary>Report completeness</summary><ul>{notice_html or "<li>All requested detail inputs were available.</li>"}</ul><p class="note">Missing details never change an execution outcome. This file embeds its available media and data; no adjacent files are required.</p></details>{log_html}
<div class="filters"><input id="search" aria-label="Search family or case" placeholder="Search family or case"><select id="status" aria-label="Filter execution outcome"><option value="">All execution outcomes</option><option value="passed">Passed</option><option value="failed">Failed</option><option value="not_run">Not run</option><option value="skipped">Skipped</option></select></div><p id="visible-count" class="meta">{len(cards)} cases shown</p><p id="no-results" hidden>No matching cases.</p>
{"".join(cards)}{journal}<script>{_JS}</script></body></html>'''
    if len(document.encode()) > MAX_HTML_BYTES:
        raise ReportError("Report exceeds the final HTML size limit")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--evidence-archive", type=Path)
    parser.add_argument("--execution-log", type=Path)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metadata = _json(
            _read(args.run_metadata.parent, args.run_metadata.name, MAX_INDEX_BYTES),
            MAX_INDEX_BYTES,
        )
        receipt = _json(
            _read(args.recovery_receipt.parent, args.recovery_receipt.name, MAX_INDEX_BYTES),
            MAX_INDEX_BYTES,
        )
        try:
            summary = _json(
                _read(args.summary.parent, args.summary.name, MAX_INDEX_BYTES), MAX_INDEX_BYTES
            )
        except (OSError, ValueError):
            summary = None
        execution_log = None
        if args.execution_log is not None:
            try:
                execution_log = _read(
                    args.execution_log.parent,
                    args.execution_log.name,
                    MAX_EXECUTION_LOG_BYTES + 1,
                    tail=True,
                ).decode("utf-8", errors="replace")
            except (OSError, ValueError):
                pass
        with _archive_root(args.evidence_archive) as (root, issues):
            document = render_report(
                summary, metadata, root, receipt, issues=issues, execution_log=execution_log
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".community-report-", dir=args.output.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(document)
            os.replace(temporary, args.output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Community report generation failed: {error}\n")
    print("Wrote the self-contained Community GPU report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
