"""Shared domain types for CI mutation testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Finding:
    """A single gate or manifest validation finding."""

    code: str
    passed: bool
    severity: str
    message: str
    model: str = ""
    bucket: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "model": self.model,
            "bucket": self.bucket,
        }


@dataclass(frozen=True)
class MandatoryPolicy:
    """Mandatory model matrix and CI robustness thresholds."""

    required_buckets: tuple[str, ...]
    tier_a_models: tuple[str, ...]
    negative_test_models: dict[str, tuple[str, ...]]
    metamorphic_test_models: dict[str, tuple[str, ...]]
    assertion_strength_min: float
    negative_test_count_min: int
    skip_xfail_delta_max: int
    report_integrity_min: float


@dataclass(frozen=True)
class PolicyBundle:
    """All manifest and gate policies loaded from ``model_connect_ci/manifests``."""

    manifest_dir: Path
    supported_models: dict[str, Any]
    reference_corpus: dict[str, Any]
    mutation_catalog: dict[str, Any]
    tolerance_policy: dict[str, Any]
    mandatory: MandatoryPolicy


@dataclass(frozen=True)
class ModelCase:
    """Normalized supported model entry derived from tests/e2e model manifests."""

    name: str
    hf_id: str
    family: str
    runtime_strategy: str
    task_strategy: str
    reference_family: str
    ci_tier: str
    architecture_bucket: str
    tier: str
    source_path: Path
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_skipped(self) -> bool:
        return bool(self.raw.get("skip") or self.raw.get("skip_reason"))

    @property
    def has_revision_pin(self) -> bool:
        return any(
            self.raw.get(field_name)
            for field_name in (
                "revision",
                "model_revision",
                "hf_revision",
                "tokenizer_revision",
                "processor_revision",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "family": self.family,
            "runtime_strategy": self.runtime_strategy,
            "task_strategy": self.task_strategy,
            "reference_family": self.reference_family,
            "ci_tier": self.ci_tier,
            "architecture_bucket": self.architecture_bucket,
            "tier": self.tier,
            "source_path": str(self.source_path),
            "has_revision_pin": self.has_revision_pin,
            "is_skipped": self.is_skipped,
        }


@dataclass(frozen=True)
class ModelInventory:
    """A normalized view of every supported E2E model manifest."""

    models: tuple[ModelCase, ...]

    def by_tier(self, tier: str) -> tuple[ModelCase, ...]:
        return tuple(model for model in self.models if model.tier == tier)

    def by_bucket(self, bucket: str) -> tuple[ModelCase, ...]:
        return tuple(model for model in self.models if model.architecture_bucket == bucket)

    def without_model(self, name: str) -> ModelInventory:
        return ModelInventory(tuple(model for model in self.models if model.name != name))

    def without_bucket(self, bucket: str) -> ModelInventory:
        return ModelInventory(tuple(model for model in self.models if model.architecture_bucket != bucket))

    @property
    def names(self) -> set[str]:
        return {model.name for model in self.models}

    def validate_mandatory_matrix(self, policy: MandatoryPolicy) -> list[Finding]:
        findings: list[Finding] = []
        names = self.names

        for model_name in policy.tier_a_models:
            present = model_name in names
            findings.append(
                Finding(
                    code="mandatory_model_present" if present else "mandatory_model_missing",
                    passed=present,
                    severity="error",
                    model=model_name,
                    message=(
                        f"Tier A model {model_name} is present"
                        if present
                        else f"Tier A model {model_name} is missing from the supported manifest set"
                    ),
                )
            )

        for bucket in policy.required_buckets:
            present = bool(self.by_bucket(bucket))
            findings.append(
                Finding(
                    code="required_bucket_present" if present else "required_bucket_empty",
                    passed=present,
                    severity="error",
                    bucket=bucket,
                    message=(
                        f"Required architecture bucket {bucket} has supported models"
                        if present
                        else f"Required architecture bucket {bucket} has no supported models"
                    ),
                )
            )

        for bucket, model_names in policy.negative_test_models.items():
            present = any(model_name in names for model_name in model_names)
            findings.append(
                Finding(
                    code="negative_test_present" if present else "negative_test_missing",
                    passed=present,
                    severity="error",
                    bucket=bucket,
                    message=(
                        f"Negative/rejection coverage exists for {bucket}"
                        if present
                        else f"Negative/rejection coverage is missing for {bucket}"
                    ),
                )
            )

        for bucket, model_names in policy.metamorphic_test_models.items():
            present = any(model_name in names for model_name in model_names)
            findings.append(
                Finding(
                    code="metamorphic_test_present" if present else "metamorphic_test_missing",
                    passed=present,
                    severity="error",
                    bucket=bucket,
                    message=(
                        f"Metamorphic coverage exists for {bucket}"
                        if present
                        else f"Metamorphic coverage is missing for {bucket}"
                    ),
                )
            )

        return findings

    def coverage_matrix(self, required_buckets: tuple[str, ...]) -> dict[str, Any]:
        by_bucket: dict[str, list[str]] = {}
        by_tier: dict[str, int] = {}
        for model in self.models:
            by_bucket.setdefault(model.architecture_bucket, []).append(model.name)
            by_tier[model.tier] = by_tier.get(model.tier, 0) + 1

        return {
            "model_count": len(self.models),
            "required_buckets": list(required_buckets),
            "covered_buckets": sorted(by_bucket),
            "models_by_bucket": {key: sorted(value) for key, value in sorted(by_bucket.items())},
            "models_by_tier": dict(sorted(by_tier.items())),
        }
