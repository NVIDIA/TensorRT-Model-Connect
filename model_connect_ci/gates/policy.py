"""Gate result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model_connect_ci.types import Finding


@dataclass(frozen=True)
class GateVerdict:
    """A product or CI robustness gate verdict."""

    name: str
    passed: bool
    findings: tuple[Finding, ...]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "findings": [finding.to_dict() for finding in self.findings],
        }
