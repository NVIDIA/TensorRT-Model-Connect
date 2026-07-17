# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, versioned contracts shared by Model Validation tooling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0"
PLAN_KIND = "model_validation_plan"


class Assessment(str, Enum):
    """An independently reported validation lane."""

    TASK = "task"
    FIDELITY = "fidelity"
    PERFORMANCE = "performance"


class AssessmentStatus(str, Enum):
    """Result of one independently evaluated assessment lane."""

    PASSED = "passed"
    FAILED = "failed"
    OBSERVED = "observed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


class CompatibilityMode(str, Enum):
    """How a plan is executed during the incremental migration."""

    LEGACY_TASK_EVAL = "legacy_task_eval"
    NATIVE = "native"


class WorkloadResolution(str, Enum):
    """Whether ordered samples were resolved while compiling the plan."""

    DEFERRED_TO_LEGACY_RUNTIME = "deferred_to_legacy_runtime"
    DEFERRED_TO_NATIVE_PREPARE = "deferred_to_native_prepare"
    RESOLVED = "resolved"


class PlanIntegrityError(ValueError):
    """Raised when a serialized plan does not match its declared digest."""


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(item) for item in value), key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot canonicalize value of type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by all plan digests."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest_value(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-like contract data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class ValidationRequest:
    """User intent before repository configuration is resolved into a plan."""

    suite_id: str
    model_selectors: tuple[str, ...] = ()
    assessments: tuple[Assessment, ...] = ()
    dataset_override: str | None = None
    limit: int | None = None
    seed: int | None = None
    performance_profile_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.suite_id, "suite_id")
        if any(not selector.strip() for selector in self.model_selectors):
            raise ValueError("model_selectors cannot contain empty values")
        if len(set(self.model_selectors)) != len(self.model_selectors):
            raise ValueError("model_selectors cannot contain duplicates")
        if len(set(self.assessments)) != len(self.assessments):
            raise ValueError("assessments cannot contain duplicates")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if Assessment.PERFORMANCE in self.assessments and not self.performance_profile_id:
            raise ValueError("performance_profile_id is required for performance assessment")
        if (
            Assessment.PERFORMANCE not in self.assessments
            and self.performance_profile_id is not None
        ):
            raise ValueError("performance_profile_id is only valid for performance assessment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "model_selectors": list(self.model_selectors),
            "assessments": [assessment.value for assessment in self.assessments],
            "dataset_override": self.dataset_override,
            "limit": self.limit,
            "seed": self.seed,
            "performance_profile_id": self.performance_profile_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationRequest:
        return cls(
            suite_id=str(payload.get("suite_id", "")),
            model_selectors=tuple(str(item) for item in payload.get("model_selectors", [])),
            assessments=tuple(Assessment(str(item)) for item in payload.get("assessments", [])),
            dataset_override=(
                str(payload["dataset_override"])
                if payload.get("dataset_override") is not None
                else None
            ),
            limit=int(payload["limit"]) if payload.get("limit") is not None else None,
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            performance_profile_id=(
                str(payload["performance_profile_id"])
                if payload.get("performance_profile_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SuiteContract:
    """Minimal immutable identity for a resolved suite contract."""

    suite_id: str
    user_contract: str
    dataset_kind: str
    contract_digest: str

    def __post_init__(self) -> None:
        _require_non_empty(self.suite_id, "suite_id")
        _require_non_empty(self.dataset_kind, "dataset_kind")
        _require_sha256(self.contract_digest, "contract_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "suite_id": self.suite_id,
            "user_contract": self.user_contract,
            "dataset_kind": self.dataset_kind,
            "contract_digest": self.contract_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SuiteContract:
        return cls(
            suite_id=str(payload.get("suite_id", "")),
            user_contract=str(payload.get("user_contract", "")),
            dataset_kind=str(payload.get("dataset_kind", "")),
            contract_digest=str(payload.get("contract_digest", "")),
        )


@dataclass(frozen=True)
class WorkloadSpec:
    """Dataset selection known when a validation plan is compiled."""

    dataset_path: str | None
    dataset_revision: str | None
    limit: int | None
    seed: int | None
    ordered_sample_ids: tuple[str, ...]
    resolution: WorkloadResolution

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive when provided")
        if len(set(self.ordered_sample_ids)) != len(self.ordered_sample_ids):
            raise ValueError("ordered_sample_ids cannot contain duplicates")
        if (
            self.resolution is WorkloadResolution.DEFERRED_TO_LEGACY_RUNTIME
            and self.ordered_sample_ids
        ):
            raise ValueError("legacy-deferred workloads cannot declare resolved sample IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_revision": self.dataset_revision,
            "limit": self.limit,
            "seed": self.seed,
            "ordered_sample_ids": list(self.ordered_sample_ids),
            "resolution": self.resolution.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkloadSpec:
        return cls(
            dataset_path=(
                str(payload["dataset_path"]) if payload.get("dataset_path") is not None else None
            ),
            dataset_revision=(
                str(payload["dataset_revision"])
                if payload.get("dataset_revision") is not None
                else None
            ),
            limit=int(payload["limit"]) if payload.get("limit") is not None else None,
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            ordered_sample_ids=tuple(str(item) for item in payload.get("ordered_sample_ids", [])),
            resolution=WorkloadResolution(str(payload.get("resolution", ""))),
        )


@dataclass(frozen=True)
class CasePlan:
    """One model/backend case selected for a validation run."""

    case_id: str
    model_name: str
    hf_id: str
    bundle: str
    runtime_strategy: str
    task_strategy: str
    reference_family: str
    user_contract: str
    manifest: str
    model_contract_digest: str

    def __post_init__(self) -> None:
        _require_non_empty(self.case_id, "case_id")
        _require_non_empty(self.model_name, "model_name")
        _require_sha256(self.model_contract_digest, "model_contract_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "model_name": self.model_name,
            "hf_id": self.hf_id,
            "bundle": self.bundle,
            "runtime_strategy": self.runtime_strategy,
            "task_strategy": self.task_strategy,
            "reference_family": self.reference_family,
            "user_contract": self.user_contract,
            "manifest": self.manifest,
            "model_contract_digest": self.model_contract_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CasePlan:
        return cls(**{field: str(payload.get(field, "")) for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ValidationPlan:
    """Immutable plan whose digest covers every execution-semantic field."""

    schema_version: str
    compatibility_mode: CompatibilityMode
    suite: SuiteContract
    request: ValidationRequest
    workload: WorkloadSpec
    task_adapter_kind: str
    task_adapter_version: str
    cases: tuple[CasePlan, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported validation plan schema {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if self.request.suite_id != self.suite.suite_id:
            raise ValueError("request and suite IDs do not match")
        if self.compatibility_mode is CompatibilityMode.NATIVE:
            _require_non_empty(self.task_adapter_kind, "task_adapter_kind")
            _require_non_empty(self.task_adapter_version, "task_adapter_version")
        elif self.task_adapter_kind or self.task_adapter_version:
            raise ValueError("legacy plans cannot declare a native task adapter")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case IDs must be unique")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": PLAN_KIND,
            "compatibility_mode": self.compatibility_mode.value,
            "suite": self.suite.to_dict(),
            "request": self.request.to_dict(),
            "workload": self.workload.to_dict(),
            "task_adapter": {
                "kind": self.task_adapter_kind,
                "version": self.task_adapter_version,
            },
            "cases": [case.to_dict() for case in self.cases],
        }

    @property
    def plan_digest(self) -> str:
        return digest_value(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._content_dict()
        payload["plan_digest"] = self.plan_digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ValidationPlan:
        if str(payload.get("kind", "")) != PLAN_KIND:
            raise ValueError(f"Expected plan kind {PLAN_KIND!r}")
        raw_cases = payload.get("cases", [])
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("cases must be a sequence")
        raw_adapter = _mapping(payload.get("task_adapter", {}), "task_adapter")
        plan = cls(
            schema_version=str(payload.get("schema_version", "")),
            compatibility_mode=CompatibilityMode(str(payload.get("compatibility_mode", ""))),
            suite=SuiteContract.from_dict(_mapping(payload.get("suite"), "suite")),
            request=ValidationRequest.from_dict(_mapping(payload.get("request"), "request")),
            workload=WorkloadSpec.from_dict(_mapping(payload.get("workload"), "workload")),
            task_adapter_kind=str(raw_adapter.get("kind", "")),
            task_adapter_version=str(raw_adapter.get("version", "")),
            cases=tuple(CasePlan.from_dict(_mapping(case, "case")) for case in raw_cases),
        )
        declared_digest = str(payload.get("plan_digest", ""))
        if declared_digest != plan.plan_digest:
            raise PlanIntegrityError("Validation plan digest does not match its semantic content")
        return plan


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value
