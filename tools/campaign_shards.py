# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic campaign sharding and receipt-backed report consolidation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import tensorrt_model_connect.utils.fcntl_shim as fcntl
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from tools import qualification_report
from tools.execution_ledger import ExecutionLedger, ExecutionLedgerError


CAMPAIGN_SCHEMA = "trtmc.model-check-sharded-campaign/v1"


class CampaignShardError(ValueError):
    """A shard selection or consolidated campaign is invalid."""


def open_campaign(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create one immutable campaign manifest, or verify the existing one."""

    expected = deepcopy(dict(manifest))
    expected["schema_version"] = CAMPAIGN_SCHEMA
    path = Path(root) / "campaign.json"
    if path.is_file():
        current = _read_object(path, "sharded campaign")
        if current != expected:
            raise CampaignShardError("sharded campaign request does not match the existing run")
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError:
        current = _read_object(path, "sharded campaign")
        if current != expected:
            raise CampaignShardError("sharded campaign request does not match the existing run")
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def load_campaign(root: Path) -> dict[str, Any]:
    campaign = _read_object(Path(root) / "campaign.json", "sharded campaign")
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise CampaignShardError("invalid sharded campaign schema")
    return campaign


@contextmanager
def consolidator_lock(root: Path):
    """Ensure that only one process publishes the global campaign snapshot."""

    path = Path(root) / ".consolidator.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignShardError("another campaign consolidator is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parse_shard(value: str) -> tuple[int, int]:
    """Parse a zero-based INDEX/COUNT shard selector."""

    index_text, separator, count_text = str(value).partition("/")
    try:
        index = int(index_text)
        count = int(count_text)
    except ValueError as error:
        raise CampaignShardError("shard must be a zero-based INDEX/COUNT") from error
    if not separator or count < 1 or index < 0 or index >= count:
        raise CampaignShardError("shard must satisfy 0 <= INDEX < COUNT")
    return index, count


def shard_name(index: int, count: int) -> str:
    if count < 1 or index < 0 or index >= count:
        raise CampaignShardError("shard must satisfy 0 <= INDEX < COUNT")
    width = max(3, len(str(count - 1)))
    return f"{index:0{width}d}-of-{count:0{width}d}"


def assign_cases(
    ordered_case_ids: Sequence[str],
    *,
    index: int,
    count: int,
) -> tuple[str, ...]:
    """Return the stable round-robin subset assigned to one shard."""

    shard_name(index, count)
    normalized = tuple(str(case_id) for case_id in ordered_case_ids)
    if any(not case_id for case_id in normalized):
        raise CampaignShardError("campaign case ids must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise CampaignShardError("campaign case ids must be unique")
    return tuple(
        case_id for position, case_id in enumerate(normalized) if position % count == index
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignShardError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignShardError(f"{label} must contain a JSON object: {path}")
    return value


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise CampaignShardError(f"shard artifact is not a regular file: {source}")
    source_stat = source.stat()
    if destination.is_file():
        destination_stat = destination.stat()
        if (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            return
        if source.suffix == ".log" and source_stat.st_size > destination_stat.st_size:
            with (
                source.open("rb") as source_stream,
                destination.open("ab") as destination_stream,
            ):
                source_stream.seek(destination_stat.st_size)
                shutil.copyfileobj(source_stream, destination_stream)
            os.utime(
                destination,
                ns=(destination_stat.st_atime_ns, source_stat.st_mtime_ns),
            )
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.copy.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _relocate_hrefs(value: Any, *, source: Path, destination: Path, prefix: Path) -> Any:
    if isinstance(value, Mapping):
        relocated: dict[str, Any] = {}
        for name, item in value.items():
            if name == "href" and isinstance(item, str):
                relative = Path(item)
                if relative.is_absolute() or ".." in relative.parts:
                    raise CampaignShardError(f"shard artifact href must be relative: {item!r}")
                target = prefix / relative
                _copy_file(source / relative, destination / target)
                relocated[name] = target.as_posix()
            else:
                relocated[str(name)] = _relocate_hrefs(
                    item,
                    source=source,
                    destination=destination,
                    prefix=prefix,
                )
        return relocated
    if isinstance(value, list):
        return [
            _relocate_hrefs(item, source=source, destination=destination, prefix=prefix)
            for item in value
        ]
    return deepcopy(value)


def _pending_row(case: Mapping[str, Any]) -> dict[str, Any]:
    row = deepcopy(dict(case.get("report", {})))
    row.update(
        {
            "id": str(case["id"]),
            "state": "pending",
            "result": None,
            "progress": {"stage": None, "attempt": 0},
            "precision": {
                "reference": "Not recorded",
                "candidate": "Not recorded",
            },
            "debug": {"logs": [], "command_artifacts": []},
        }
    )
    return row


def merge_receipt_reports(
    output: Path,
    *,
    report_kind: str,
    campaign: Mapping[str, Any],
    expected_cases: Sequence[Mapping[str, Any]],
    shard_outputs: Sequence[tuple[str, Path]],
) -> tuple[Path, Path, dict[str, Any]]:
    """Merge shard-local receipt projections into one ordered public report."""

    if report_kind not in {"accuracy", "performance"}:
        raise CampaignShardError(f"unknown qualification report kind: {report_kind}")
    task_kind = "accuracy" if report_kind == "accuracy" else "performance"
    expected = [deepcopy(dict(case)) for case in expected_cases]
    expected_ids = [str(case.get("id", "")) for case in expected]
    if any(not case_id for case_id in expected_ids) or len(expected_ids) != len(set(expected_ids)):
        raise CampaignShardError("consolidated case inventory is invalid")
    shard_count = campaign.get("shard_count")
    if not isinstance(shard_count, int) or shard_count < 1:
        raise CampaignShardError("consolidated shard count is invalid")
    expected_by_shard = {
        shard_name(index, shard_count): [
            str(case["id"]) for case in expected if case.get("shard") == index
        ]
        for index in range(shard_count)
    }
    if sum(len(case_ids) for case_ids in expected_by_shard.values()) != len(expected):
        raise CampaignShardError("every consolidated case must belong to one shard")

    rows: dict[str, dict[str, Any]] = {str(case["id"]): _pending_row(case) for case in expected}
    shard_runs: list[dict[str, Any]] = []
    receipt_sources: dict[str, str] = {}
    output = Path(output)
    for label, source in shard_outputs:
        source = Path(source)
        try:
            ledger = ExecutionLedger.load(source, task_kind=task_kind)
        except ExecutionLedgerError as error:
            raise CampaignShardError(str(error)) from error
        report = _read_object(source / "report.json", "shard report")
        if report.get("report_kind") != report_kind:
            raise CampaignShardError(f"shard {label} report kind does not match {report_kind}")
        report_rows = report.get("results")
        if not isinstance(report_rows, list):
            raise CampaignShardError(f"shard {label} report results must be an array")
        report_by_id = {
            str(row.get("id", "")): row for row in report_rows if isinstance(row, Mapping)
        }
        ledger_ids = [str(case["id"]) for case in ledger.cases()]
        assigned_ids = tuple(str(case_id) for case_id in expected_by_shard.get(label, ()))
        if tuple(ledger_ids) != assigned_ids:
            raise CampaignShardError(
                f"shard {label} receipt inventory does not match its assignment"
            )
        if set(report_by_id) != set(ledger_ids):
            raise CampaignShardError(f"shard {label} report does not match its receipts")
        ledger_prefix = Path("ledger") / "shards" / label
        ledger_manifest = _read_object(source / "ledger" / "campaign.json", "shard ledger")
        receipt_paths = {
            str(entry["case"]["id"]): str(entry["receipt"])
            for entry in ledger_manifest.get("cases", [])
            if isinstance(entry, Mapping) and isinstance(entry.get("case"), Mapping)
        }
        for ledger_file in sorted((source / "ledger").rglob("*.json")):
            _copy_file(
                ledger_file,
                output / ledger_prefix / ledger_file.relative_to(source / "ledger"),
            )
        for case_id in ledger_ids:
            receipt = ledger.receipt(case_id)
            row = report_by_id[case_id]
            if row.get("state") != receipt.get("state") or row.get("result") != receipt.get(
                "result"
            ):
                raise CampaignShardError(f"shard {label} report is stale for case {case_id!r}")
            rows[case_id] = _relocate_hrefs(
                row,
                source=source,
                destination=output,
                prefix=Path("artifacts") / "shards" / label,
            )
            receipt_sources[case_id] = (ledger_prefix / receipt_paths[case_id]).as_posix()
        shard_run = report.get("run", {})
        shard_run = shard_run if isinstance(shard_run, Mapping) else {}
        shard_runs.append(
            {
                "shard": label,
                **_relocate_hrefs(
                    shard_run,
                    source=source,
                    destination=output,
                    prefix=Path("artifacts") / "shards" / label,
                ),
            }
        )

    public_run = {
        "source_revision": campaign.get("revision"),
        "platform": campaign.get("platform"),
        "hostname": "multiple shards",
        "shards": shard_runs,
    }
    public_identity = {
        "run_id": campaign.get("run_id"),
        "disposition": "completed",
        "source_revision": campaign.get("revision"),
    }
    if any(row["state"] != "terminal" for row in rows.values()):
        public_identity["disposition"] = "running"
    return qualification_report.materialize_report(
        output,
        report_kind=report_kind,
        title=(
            "TRTMC Accuracy & Fidelity Qualification"
            if report_kind == "accuracy"
            else "TRTMC Performance Qualification"
        ),
        identity=public_identity,
        run=public_run,
        results=[rows[case_id] for case_id in expected_ids],
        metadata={
            "campaign": {
                "schema_version": CAMPAIGN_SCHEMA,
                "shard_count": campaign.get("shard_count"),
            },
            "receipt_sources": receipt_sources,
        },
    )
