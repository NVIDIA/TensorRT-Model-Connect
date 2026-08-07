#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load model identifiers without knowing task-specific bindings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class ModelSelectionError(ValueError):
    """A model selection is malformed or cannot be resolved."""


def normalize_models(models: Iterable[str]) -> tuple[str, ...]:
    """Return non-empty model identifiers, de-duplicated in input order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_model in models:
        if not isinstance(raw_model, str):
            raise ModelSelectionError("model names must be strings")
        model = raw_model.strip()
        if not model:
            raise ModelSelectionError("model names must not be empty")
        if model not in seen:
            normalized.append(model)
            seen.add(model)
    return tuple(normalized)


def _models_from_matrix(payload: Mapping[str, Any]) -> tuple[str, ...]:
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping):
        return ()
    include = matrix.get("include")
    if not isinstance(include, list):
        return ()
    models: list[str] = []
    for index, item in enumerate(include):
        if not isinstance(item, Mapping) or not isinstance(item.get("model"), str):
            raise ModelSelectionError(
                f"matrix.include[{index}] must contain a string model"
            )
        models.append(str(item["model"]))
    return normalize_models(models)


def models_from_payload(payload: Any) -> tuple[str, ...]:
    """Read model names from model_ci output or the minimal {models: [...]} form."""

    if not isinstance(payload, Mapping):
        raise ModelSelectionError("model selection must contain a JSON object")

    matrix_models = _models_from_matrix(payload)
    if matrix_models:
        return matrix_models

    for field in ("affected_models", "models"):
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, list):
            raise ModelSelectionError(f"{field} must be a JSON array")
        return normalize_models(value)

    raise ModelSelectionError(
        "model selection must contain matrix.include, affected_models, or models"
    )


def load_model_selection(path: Path) -> tuple[str, ...]:
    """Load model owner/family IDs from a JSON selection file."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelSelectionError(f"cannot read model selection {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ModelSelectionError(f"invalid model selection JSON {path}: {exc}") from exc

    models = models_from_payload(payload)
    if not models:
        raise ModelSelectionError(f"model selection contains no models: {path}")
    return models
