# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small durable execution ledger shared by qualification runners."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trtmc.execution-ledger/v1"
CASE_STATES = ("pending", "running", "terminal")
TERMINAL_RESULTS = ("green", "yellow", "red", "white")
ATTEMPT_OUTCOMES = ("completed", "failed", "timed_out")
ATTEMPT_STATES = ("running", *ATTEMPT_OUTCOMES, "interrupted")


class ExecutionLedgerError(ValueError):
    """The persisted campaign or a requested state transition is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionLedgerError(f"cannot read execution ledger file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExecutionLedgerError(f"execution ledger file must contain an object: {path}")
    return value


def _case_token(case_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", case_id).strip("-") or "case"
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{digest}"


class ExecutionLedger:
    """Persist one immutable campaign inventory and one receipt per selected case."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root
        self._manifest = deepcopy(dict(manifest))
        entries = self._manifest.get("cases")
        if not isinstance(entries, list):
            raise ExecutionLedgerError("execution ledger case inventory must be an array")
        self._cases: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("case"), Mapping):
                raise ExecutionLedgerError("execution ledger case entry must be an object")
            case_id = str(entry["case"].get("id", "")).strip()
            expected_receipt = f"cases/{_case_token(case_id)}/receipt.json"
            if not case_id or entry.get("receipt") != expected_receipt:
                raise ExecutionLedgerError("execution ledger case entry is invalid")
            if case_id in self._cases:
                raise ExecutionLedgerError("execution ledger case ids must be unique")
            self._cases[case_id] = deepcopy(dict(entry))

    @classmethod
    def load(cls, output: Path, *, task_kind: str | None = None) -> ExecutionLedger:
        root = Path(output) / "ledger"
        manifest = _read_json(root / "campaign.json")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ExecutionLedgerError("invalid execution ledger campaign schema")
        if task_kind is not None and manifest.get("task_kind") != task_kind:
            raise ExecutionLedgerError("execution ledger task does not match the report")
        ledger = cls(root, manifest)
        for case in ledger.cases():
            ledger.receipt(str(case["id"]))
        return ledger

    @classmethod
    def open(
        cls,
        output: Path,
        *,
        campaign_id: str,
        task_kind: str,
        fingerprint: str,
        cases: Sequence[Mapping[str, Any]],
    ) -> ExecutionLedger:
        root = Path(output) / "ledger"
        manifest_path = root / "campaign.json"
        normalized_cases = [deepcopy(dict(case)) for case in cases]
        case_ids = [str(case.get("id", "")).strip() for case in normalized_cases]
        if any(not case_id for case_id in case_ids):
            raise ExecutionLedgerError("every selected case must have a non-empty id")
        if len(set(case_ids)) != len(case_ids):
            raise ExecutionLedgerError("selected case ids must be unique")

        entries = [
            {
                "case": case,
                "receipt": f"cases/{_case_token(case_id)}/receipt.json",
            }
            for case_id, case in zip(case_ids, normalized_cases, strict=True)
        ]
        expected = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": str(campaign_id),
            "task_kind": str(task_kind),
            "fingerprint": str(fingerprint),
            "cases": entries,
        }
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            for field in ("schema_version", "campaign_id", "task_kind", "fingerprint"):
                if manifest.get(field) != expected[field]:
                    raise ExecutionLedgerError(
                        f"execution ledger {field} does not match this campaign"
                    )
            if manifest.get("cases") != entries:
                raise ExecutionLedgerError(
                    "execution ledger case inventory does not match this campaign"
                )
            ledger = cls(root, manifest)
            for case_id in case_ids:
                path = ledger._receipt_path(case_id)
                if not path.exists():
                    raise ExecutionLedgerError(
                        f"execution ledger is missing receipt for case {case_id!r}"
                    )
                ledger._validate_receipt(case_id, _read_json(path))
            return ledger

        manifest = expected
        ledger = cls(root, manifest)
        for case_id in case_ids:
            _write_json_atomic(
                ledger._receipt_path(case_id),
                ledger._pending_receipt(case_id),
            )
        _write_json_atomic(manifest_path, manifest)
        return ledger

    def cases(self) -> list[dict[str, Any]]:
        return [deepcopy(entry["case"]) for entry in self._manifest["cases"]]

    def receipt(self, case_id: str) -> dict[str, Any]:
        path = self._receipt_path(case_id)
        if not path.is_file() or path.is_symlink():
            raise ExecutionLedgerError(f"execution ledger has no regular receipt for {case_id!r}")
        receipt = _read_json(path)
        self._validate_receipt(case_id, receipt)
        return deepcopy(receipt)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"case": case, "receipt": self.receipt(str(case["id"]))}
            for case in self.cases()
        ]

    def begin(
        self,
        case_id: str,
        *,
        stage: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> int:
        receipt = self.receipt(case_id)
        if receipt["state"] == "terminal":
            raise ExecutionLedgerError(f"case {case_id!r} is already terminal")
        if receipt["state"] == "running":
            raise ExecutionLedgerError(f"case {case_id!r} is already running")
        attempt_number = len(receipt["attempts"]) + 1
        receipt["state"] = "running"
        receipt["stage"] = str(stage)
        receipt["active_attempt"] = attempt_number
        receipt["attempts"].append(
            {
                "attempt": attempt_number,
                "state": "running",
                "stage": str(stage),
                "started_at": _now(),
                "evidence": deepcopy(dict(evidence or {})),
            }
        )
        receipt["updated_at"] = _now()
        self._write_receipt(case_id, receipt)
        return attempt_number

    def update_stage(
        self,
        case_id: str,
        stage: str,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        receipt = self.receipt(case_id)
        if receipt["state"] != "running":
            raise ExecutionLedgerError(f"case {case_id!r} is not running")
        receipt["stage"] = str(stage)
        receipt["attempts"][-1]["stage"] = str(stage)
        receipt["attempts"][-1]["evidence"].update(
            deepcopy(dict(evidence or {}))
        )
        receipt["updated_at"] = _now()
        self._write_receipt(case_id, receipt)

    def finish(
        self,
        case_id: str,
        *,
        result: str,
        payload: Mapping[str, Any],
        attempt_outcome: str = "completed",
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if result not in TERMINAL_RESULTS:
            raise ExecutionLedgerError(f"unknown traffic-light result: {result!r}")
        if attempt_outcome not in ATTEMPT_OUTCOMES:
            raise ExecutionLedgerError(f"unknown attempt outcome: {attempt_outcome!r}")
        receipt = self.receipt(case_id)
        if receipt["state"] != "running":
            raise ExecutionLedgerError(f"case {case_id!r} is not running")
        finished_at = _now()
        receipt["state"] = "terminal"
        receipt["stage"] = receipt["attempts"][-1]["stage"]
        receipt["active_attempt"] = None
        receipt["result"] = result
        receipt["payload"] = deepcopy(dict(payload))
        receipt["attempts"][-1]["state"] = attempt_outcome
        receipt["attempts"][-1]["finished_at"] = finished_at
        receipt["attempts"][-1]["evidence"].update(
            deepcopy(dict(evidence or {}))
        )
        receipt["updated_at"] = finished_at
        self._write_receipt(case_id, receipt)

    def retry(
        self,
        case_id: str,
        *,
        attempt_outcome: str = "failed",
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        """Close one failed attempt while leaving its case eligible to run again."""

        if attempt_outcome not in {"failed", "timed_out"}:
            raise ExecutionLedgerError(
                f"retry attempt must be failed or timed_out: {attempt_outcome!r}"
            )
        receipt = self.receipt(case_id)
        if receipt["state"] != "running":
            raise ExecutionLedgerError(f"case {case_id!r} is not running")
        finished_at = _now()
        receipt["attempts"][-1]["state"] = attempt_outcome
        receipt["attempts"][-1]["finished_at"] = finished_at
        receipt["attempts"][-1]["evidence"].update(
            deepcopy(dict(evidence or {}))
        )
        receipt["state"] = "pending"
        receipt["stage"] = None
        receipt["active_attempt"] = None
        receipt["updated_at"] = finished_at
        self._write_receipt(case_id, receipt)

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        for case in self.cases():
            case_id = str(case["id"])
            receipt = self.receipt(case_id)
            if receipt["state"] != "running":
                continue
            finished_at = _now()
            receipt["attempts"][-1]["state"] = "interrupted"
            receipt["attempts"][-1]["finished_at"] = finished_at
            receipt["state"] = "pending"
            receipt["stage"] = None
            receipt["active_attempt"] = None
            receipt["updated_at"] = finished_at
            self._write_receipt(case_id, receipt)
            recovered.append(case_id)
        return recovered

    def reopen_retryable(self) -> list[str]:
        reopened: list[str] = []
        for case in self.cases():
            case_id = str(case["id"])
            receipt = self.receipt(case_id)
            if receipt["state"] != "terminal" or receipt["result"] != "white":
                continue
            evidence = receipt["attempts"][-1]["evidence"]
            if evidence.get("retryable") is not True:
                continue
            receipt["previous_results"].append(
                {
                    "attempt": receipt["attempts"][-1]["attempt"],
                    "result": receipt["result"],
                    "payload": receipt["payload"],
                    "finished_at": receipt["updated_at"],
                }
            )
            receipt["state"] = "pending"
            receipt["stage"] = None
            receipt["result"] = None
            receipt["payload"] = None
            receipt["updated_at"] = _now()
            self._write_receipt(case_id, receipt)
            reopened.append(case_id)
        return reopened

    def _pending_receipt(self, case_id: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self._manifest["campaign_id"],
            "task_kind": self._manifest["task_kind"],
            "case_id": case_id,
            "state": "pending",
            "stage": None,
            "active_attempt": None,
            "attempts": [],
            "previous_results": [],
            "result": None,
            "payload": None,
            "updated_at": _now(),
        }

    def _receipt_path(self, case_id: str) -> Path:
        try:
            relative = self._cases[str(case_id)]["receipt"]
        except KeyError as error:
            raise ExecutionLedgerError(f"unknown case: {case_id!r}") from error
        return self.root / str(relative)

    def _write_receipt(self, case_id: str, receipt: Mapping[str, Any]) -> None:
        self._validate_receipt(case_id, receipt)
        _write_json_atomic(self._receipt_path(case_id), receipt)

    def _validate_receipt(self, case_id: str, receipt: Mapping[str, Any]) -> None:
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise ExecutionLedgerError(f"invalid receipt schema for case {case_id!r}")
        if receipt.get("campaign_id") != self._manifest["campaign_id"]:
            raise ExecutionLedgerError(f"receipt campaign mismatch for case {case_id!r}")
        if receipt.get("task_kind") != self._manifest["task_kind"]:
            raise ExecutionLedgerError(f"receipt task mismatch for case {case_id!r}")
        if receipt.get("case_id") != case_id:
            raise ExecutionLedgerError(f"receipt case id mismatch for case {case_id!r}")
        state = receipt.get("state")
        if state not in CASE_STATES:
            raise ExecutionLedgerError(f"invalid receipt state for case {case_id!r}: {state!r}")
        attempts = receipt.get("attempts")
        if not isinstance(attempts, list):
            raise ExecutionLedgerError(f"receipt attempts must be an array for case {case_id!r}")
        for index, attempt in enumerate(attempts, start=1):
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("attempt") != index
                or attempt.get("state") not in ATTEMPT_STATES
                or not str(attempt.get("stage", "")).strip()
                or not attempt.get("started_at")
                or not isinstance(attempt.get("evidence"), Mapping)
            ):
                raise ExecutionLedgerError(f"invalid attempt history for case {case_id!r}")
            if attempt.get("state") == "running":
                if index != len(attempts) or attempt.get("finished_at") is not None:
                    raise ExecutionLedgerError(f"invalid attempt history for case {case_id!r}")
            elif not attempt.get("finished_at"):
                raise ExecutionLedgerError(f"invalid attempt history for case {case_id!r}")
        previous_results = receipt.get("previous_results")
        if not isinstance(previous_results, list) or any(
            not isinstance(previous, Mapping)
            or previous.get("result") != "white"
            or not isinstance(previous.get("payload"), Mapping)
            for previous in previous_results
        ):
            raise ExecutionLedgerError(f"invalid previous results for case {case_id!r}")
        active_attempt = receipt.get("active_attempt")
        result = receipt.get("result")
        if state == "running":
            if not attempts or attempts[-1].get("state") != "running":
                raise ExecutionLedgerError(f"running case {case_id!r} needs one active attempt")
            if active_attempt != attempts[-1].get("attempt") or result is not None:
                raise ExecutionLedgerError(f"running receipt is inconsistent for case {case_id!r}")
        elif active_attempt is not None:
            raise ExecutionLedgerError(f"non-running case {case_id!r} has an active attempt")
        if state == "pending":
            if receipt.get("stage") is not None:
                raise ExecutionLedgerError(f"pending case {case_id!r} has a stage")
            if attempts and attempts[-1].get("state") not in {
                "interrupted",
                "failed",
                "timed_out",
            }:
                raise ExecutionLedgerError(f"pending case {case_id!r} has no retryable attempt")
        if state in {"running", "terminal"} and receipt.get("stage") != attempts[-1].get(
            "stage"
        ):
            raise ExecutionLedgerError(f"case and attempt stage differ for {case_id!r}")
        if state == "terminal":
            if result not in TERMINAL_RESULTS or not isinstance(receipt.get("payload"), dict):
                raise ExecutionLedgerError(f"terminal receipt is incomplete for case {case_id!r}")
            if not attempts or attempts[-1].get("state") not in ATTEMPT_OUTCOMES:
                raise ExecutionLedgerError(f"terminal case {case_id!r} has no final attempt")
        elif result is not None or receipt.get("payload") is not None:
            raise ExecutionLedgerError(f"non-terminal receipt has a result for case {case_id!r}")
