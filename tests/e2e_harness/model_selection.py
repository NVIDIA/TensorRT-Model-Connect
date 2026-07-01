# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared exact-name selection helpers for E2E model cases."""

from __future__ import annotations

from os import PathLike
from typing import Iterable, Protocol, TypeVar


BUNDLE_GROUP_PREFIX = "bundle:"


class _NamedCase(Protocol):
    name: str


_CaseT = TypeVar("_CaseT", bound=_NamedCase)


def case_names_from_param(value: str) -> list[str]:
    if value.startswith(BUNDLE_GROUP_PREFIX):
        payload = value[len(BUNDLE_GROUP_PREFIX):]
        return [name for name in payload.split("+") if name]
    return [value] if value else []


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
            names.update(case_names_from_param(value))
    return names


def select_cases_from_models_file(
    cases: Iterable[_CaseT], path: str | PathLike[str]
) -> list[_CaseT]:
    """Return only cases whose manifest names are explicitly listed."""
    selected_names = read_e2e_models_file(path)
    return [case for case in cases if case.name in selected_names]
