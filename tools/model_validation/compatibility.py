# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade between legacy Task Eval and versioned plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import Assessment, ValidationPlan, ValidationRequest
from .planner import compile_legacy_plan, compile_native_plan
from .registry import TaskAdapter


class LegacyTaskEvalFacade:
    """Compile and persist plans without changing legacy runtime behavior."""

    def compile_eval_plan(
        self,
        *,
        suite: Mapping[str, Any],
        models: Sequence[Mapping[str, Any]],
        model_selectors: Sequence[str] = (),
        dataset_override: str | None = None,
        limit: int | None = None,
        seed: int | None = None,
        performance_profile_id: str | None = None,
        native_adapter: TaskAdapter | None = None,
    ) -> ValidationPlan:
        normalized_limit = limit if limit and limit > 0 else None
        assessments = [Assessment.TASK, Assessment.FIDELITY]
        if performance_profile_id:
            assessments.append(Assessment.PERFORMANCE)
        request = ValidationRequest(
            suite_id=str(suite.get("id", "")),
            model_selectors=tuple(model_selectors),
            assessments=tuple(assessments),
            dataset_override=dataset_override or None,
            limit=normalized_limit,
            seed=seed,
            performance_profile_id=performance_profile_id or None,
        )
        if native_adapter is not None:
            return compile_native_plan(
                request,
                suite=suite,
                models=models,
                task_adapter_kind=native_adapter.kind,
                task_adapter_version=native_adapter.version,
            )
        return compile_legacy_plan(request, suite=suite, models=models)

    def write_plan(self, plan: ValidationPlan, artifact_root: Path) -> Path:
        """Atomically write a self-validating plan next to legacy artifacts."""

        artifact_root.mkdir(parents=True, exist_ok=True)
        destination = artifact_root / "validation_plan.json"
        temporary = artifact_root / ".validation_plan.json.tmp"
        try:
            temporary.write_text(
                json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
