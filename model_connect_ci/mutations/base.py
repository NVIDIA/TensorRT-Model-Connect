"""Base mutation catalog types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MutationOperator:
    """Declarative mutation operator loaded from ``mutation_catalog.yaml``."""

    id: str
    taxonomy: str
    layer: str
    family: str
    expected_outcome: str
    critical: bool
    applies_to_buckets: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "taxonomy": self.taxonomy,
            "layer": self.layer,
            "family": self.family,
            "expected_outcome": self.expected_outcome,
            "critical": self.critical,
            "applies_to_buckets": list(self.applies_to_buckets),
            "description": self.description,
        }


@dataclass(frozen=True)
class MutationCatalog:
    """Collection of declarative mutation operators."""

    operators: tuple[MutationOperator, ...]

    def by_taxonomy(self, taxonomy: str) -> tuple[MutationOperator, ...]:
        return tuple(operator for operator in self.operators if operator.taxonomy == taxonomy)

    def by_layer(self, layer: str) -> tuple[MutationOperator, ...]:
        return tuple(operator for operator in self.operators if operator.layer == layer)

    def to_dict(self) -> dict[str, Any]:
        return {"operators": [operator.to_dict() for operator in self.operators]}
