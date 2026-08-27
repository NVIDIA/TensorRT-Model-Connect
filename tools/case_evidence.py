# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Case-level source provenance and per-model qualification identity."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable, Mapping


EXACT_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")


class CaseEvidenceError(ValueError):
    """Case provenance cannot support a trustworthy qualification result."""


def exact_source_revision(value: Any, *, label: str = "source revision") -> str:
    revision = str(value or "").strip().lower()
    if EXACT_SOURCE_REVISION.fullmatch(revision) is None:
        raise CaseEvidenceError(f"{label} must be an exact 40-character Git SHA")
    return revision


def stamp_case(payload: Mapping[str, Any], source_revision: str) -> dict[str, Any]:
    """Return one case payload bound to exactly one observed source revision."""

    revision = exact_source_revision(source_revision)
    stamped = deepcopy(dict(payload))
    embedded = str(stamped.get("source_revision", "") or "").strip().lower()
    if embedded and embedded != revision:
        raise CaseEvidenceError(
            "case source revision conflict: "
            f"embedded={embedded}, execution={revision}"
        )
    stamped["source_revision"] = revision
    return stamped


def summarize_model_revisions(
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize whether every terminal case for each model uses one revision."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        model = str(case.get("model", "") or "").strip()
        if not model:
            raise CaseEvidenceError("qualification case is missing its model identity")
        grouped.setdefault(model, []).append(case)

    models: dict[str, dict[str, Any]] = {}
    campaign_revisions: set[str] = set()
    for model in sorted(grouped):
        model_cases = grouped[model]
        revisions: set[str] = set()
        missing = False
        incomplete = False
        case_ids: list[str] = []
        tasks: set[str] = set()
        for case in model_cases:
            case_ids.append(str(case.get("id", "") or ""))
            task = str(case.get("task", "") or "").strip()
            if task:
                tasks.add(task)
            if case.get("state") != "terminal":
                incomplete = True
                continue
            revision = str(case.get("source_revision", "") or "").strip().lower()
            if EXACT_SOURCE_REVISION.fullmatch(revision) is None:
                missing = True
                continue
            revisions.add(revision)

        campaign_revisions.update(revisions)
        if incomplete:
            status = "incomplete"
        elif missing:
            status = "missing"
        elif len(revisions) != 1:
            status = "mixed"
        else:
            status = "consistent"
        models[model] = {
            "status": status,
            "source_revision": next(iter(revisions)) if status == "consistent" else None,
            "source_revisions": sorted(revisions),
            "case_ids": case_ids,
            "tasks": sorted(tasks),
        }

    return {
        "consistent": all(record["status"] == "consistent" for record in models.values()),
        "source_revisions": sorted(campaign_revisions),
        "models": models,
    }
