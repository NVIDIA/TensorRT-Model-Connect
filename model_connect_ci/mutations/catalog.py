"""Load the mutation operator catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from model_connect_ci.mutations.base import MutationCatalog, MutationOperator


def _operator_from_dict(raw: dict[str, Any]) -> MutationOperator:
    return MutationOperator(
        id=str(raw["id"]),
        taxonomy=str(raw["taxonomy"]),
        layer=str(raw["layer"]),
        family=str(raw["family"]),
        expected_outcome=str(raw["expected_outcome"]),
        critical=bool(raw.get("critical", False)),
        applies_to_buckets=tuple(str(item) for item in raw.get("applies_to_buckets", [])),
        description=str(raw.get("description", "")),
    )


def load_mutation_catalog(path: Path) -> MutationCatalog:
    """Load ``mutation_catalog.yaml``."""

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_operators = data.get("operators", [])
    if not isinstance(raw_operators, list):
        raise ValueError(f"{path} field 'operators' must be a list")
    return MutationCatalog(tuple(_operator_from_dict(item) for item in raw_operators))
