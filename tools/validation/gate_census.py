# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic review inventory from resolved validation gates."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tools.validation import catalog as validation_catalog
from tools.validation.gate_policy import (
    describe_sample_acceptance,
    describe_shadow_gate_policy,
)


def _variant(
    *,
    suite: Mapping[str, Any],
    sample_count: int | None,
    models: Sequence[str],
) -> dict[str, Any]:
    gates = suite.get("gates", {})
    gates = gates if isinstance(gates, Mapping) else {}
    acceptance = suite.get("sample_acceptance")
    acceptance = acceptance if isinstance(acceptance, Mapping) else None
    metric_kinds = suite.get("gate_metric_kinds", {})
    metric_kinds = metric_kinds if isinstance(metric_kinds, Mapping) else {}
    sample_count_metrics = suite.get("gate_sample_count_metrics", {})
    sample_count_metrics = (
        sample_count_metrics if isinstance(sample_count_metrics, Mapping) else {}
    )
    policy = describe_shadow_gate_policy(
        configured_gates=gates,
        sample_count=sample_count,
        # An empty aggregate-gate list is valid when sample acceptance is the
        # blocking rule.
        policy_mode=(
            "observation_only"
            if acceptance and not gates
            else str(suite.get("gate_policy", "blocking") or "blocking")
        ),
        metric_kinds={str(name): str(kind) for name, kind in metric_kinds.items()},
        sample_count_metrics={
            str(name): str(metric) for name, metric in sample_count_metrics.items()
        },
    )
    if acceptance:
        policy["policy_mode"] = "blocking"
    variant = {
        "models": list(models),
        "policy": policy,
    }
    if acceptance:
        variant["sample_acceptance"] = describe_sample_acceptance(
            policy=acceptance,
            sample_count=sample_count,
        )
    return variant


def _review(
    *,
    variants: Sequence[Mapping[str, Any]],
    has_models: bool,
    sample_count: int | None,
    selection: str,
) -> list[dict[str, Any]]:
    review: list[dict[str, Any]] = []
    if selection not in {"model_inventory", "explicit_only"}:
        review.append({"code": "unsupported_selection", "value": selection})
    elif selection == "model_inventory" and not has_models:
        review.append({"code": "no_selected_models"})
    elif selection == "explicit_only" and has_models:
        review.append({"code": "explicit_only_has_selected_models"})
    if sample_count is None:
        review.append({"code": "sample_limit_unconfigured"})
    if not any(
        variant.get("policy", {}).get("policy_mode") == "blocking"
        for variant in variants
    ):
        return review
    if any(variant.get("sample_acceptance", {}).get("issues") for variant in variants):
        review.append({"code": "invalid_sample_acceptance"})
    return review


def _sample_count(catalog: Mapping[str, Any], suite_id: str) -> tuple[int | None, str]:
    configured = catalog.get("sample_limits", {}).get(suite_id)
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured, "configured"
    return None, "full_dataset" if configured == -1 else "unconfigured"


def _signature(suite: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "gates": suite.get("gates", {}),
            "gate_policy": suite.get("gate_policy", "blocking"),
            "gate_metric_kinds": suite.get("gate_metric_kinds", {}),
            "gate_sample_count_metrics": suite.get("gate_sample_count_metrics", {}),
            "sample_acceptance": suite.get("sample_acceptance", {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def build_gate_census(
    *,
    catalog: Mapping[str, Any],
    suites: Mapping[str, Mapping[str, Any]],
    task_models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Group model bindings by resolved gate policy and list approval gaps."""

    suite_rows: list[dict[str, Any]] = []
    binding_count = 0
    for suite_id, suite in suites.items():
        sample_count, sample_limit_source = _sample_count(catalog, suite_id)
        selection = str(suite.get("selection", "model_inventory") or "model_inventory")
        grouped: dict[str, dict[str, Any]] = {}
        for model_name, model_spec in catalog.get("models", {}).items():
            if suite_id not in model_spec.get("workloads", []):
                continue
            binding_count += 1
            resolved = validation_catalog.resolve_suite_for_model(
                dict(suite),
                dict(task_models[model_name]),
            )
            signature = _signature(resolved)
            if signature not in grouped:
                grouped[signature] = _variant(
                    suite=resolved,
                    sample_count=sample_count,
                    models=[],
                )
            grouped[signature]["models"].append(str(model_name))
        variants = list(grouped.values()) or [
            _variant(suite=suite, sample_count=sample_count, models=[])
        ]
        review = _review(
            variants=variants,
            has_models=any(variant["models"] for variant in variants),
            sample_count=sample_count,
            selection=selection,
        )
        suite_rows.append(
            {
                "id": str(suite_id),
                "owner": {"kind": "workload", "id": str(suite_id)},
                "rationale": str(suite.get("description", "") or "").strip(),
                "configured_sample_count": sample_count,
                "sample_limit_source": sample_limit_source,
                "selection": selection,
                "variants": variants,
                "review": review,
            }
        )
    variants = [variant for suite in suite_rows for variant in suite["variants"]]
    return {
        "schema_version": "trtmc.validation-gate-census/v1",
        "summary": {
            "suites": len(suite_rows),
            "bindings": binding_count,
            "variants": len(variants),
            "blocking_variants": sum(
                variant["policy"]["policy_mode"] == "blocking"
                for variant in variants
            ),
            "observation_only_variants": sum(
                variant["policy"]["policy_mode"] == "observation_only"
                for variant in variants
            ),
            "invalid_variants": sum(
                bool(variant["policy"]["issues"])
                or bool(variant.get("sample_acceptance", {}).get("issues"))
                for variant in variants
            ),
            "review_required_suites": sum(bool(suite["review"]) for suite in suite_rows),
        },
        "suites": suite_rows,
    }
