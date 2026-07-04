# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared exact-name selection helpers for E2E model cases."""

from __future__ import annotations

from os import PathLike
from typing import Iterable, Mapping, Protocol, TypeVar


class _NamedCase(Protocol):
    name: str


_CaseT = TypeVar("_CaseT", bound=_NamedCase)


class _FilterableCase(_NamedCase, Protocol):
    family: str
    runtime_strategy: str
    task_strategy: str
    metadata: Mapping[str, object]


def parse_e2e_model_filters(values: Iterable[str]) -> set[str]:
    filters: set[str] = set()
    for raw in values:
        filters.update(item.strip() for item in str(raw).split(",") if item.strip())
    return filters


def case_matches_e2e_model(case: _FilterableCase, filters: set[str]) -> bool:
    """Match exact case names and supported aliases, never HF ID basenames."""
    if not filters:
        return True
    metadata = case.metadata or {}
    fields = {
        case.name,
        case.family,
        case.runtime_strategy,
        case.task_strategy,
        str(metadata.get("family", "")),
        str(metadata.get("runtime_strategy", "")),
    }
    return bool(filters & {field for field in fields if field})


def read_e2e_models_file(path: str | PathLike[str]) -> set[str]:
    """Read exact manifest names from a models file or pytest node-id file."""
    names: set[str] = set()
    with open(path, encoding="utf-8") as models_file:
        for raw in models_file:
            value = raw.split("#", 1)[0].strip()
            if not value:
                continue
            if "[" in value and "]" in value:
                value = value.rsplit("[", 1)[1].split("]", 1)[0]
            names.add(value)
    return names


def select_cases_from_models_file(
    cases: Iterable[_CaseT], path: str | PathLike[str]
) -> list[_CaseT]:
    """Return only cases whose manifest names are explicitly listed."""
    selected_names = read_e2e_models_file(path)
    return [case for case in cases if case.name in selected_names]
