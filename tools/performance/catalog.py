# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve performance suite definitions behind one catalog interface."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import yaml

from benchmarks.performance.baselines.timing_contracts import timing_contract
from tensorrt_model_connect.benchmark.catalog import ManifestCatalog
from tools import model_selection


SUITE_SCHEMA = "trtmc.perf-suite/v2"
L0_PROFILE_PATTERN = re.compile(r"(?:^|-)l0(?:-|$)", re.IGNORECASE)
TASK_REFERENCE_ADAPTERS = {
    "hf-diffusers",
    "hf-qwen3-omni",
    "hf-transformers-asr",
    "hf-transformers-embedding",
    "hf-transformers-reranking",
    "hf-transformers-tts",
    "hf-transformers-vision",
    "hf-transformers-vlm",
    "nemo-asr",
    "nemo-tts",
    "pytorch-personaplex",
    "pytorch-timeseries",
    "upstream-elf",
    "upstream-lance",
    "upstream-sana-wm",
}


class PerformanceSuiteError(ValueError):
    """The performance suite definition or selection is invalid."""


@dataclass(frozen=True)
class PerformanceSuite:
    """A validated performance definition with its expanded executable cases."""

    source: Path
    definition: dict[str, Any]
    cases: tuple[dict[str, Any], ...]
    excluded_profiles: dict[str, str]

    def select(
        self,
        *,
        entries: Sequence[str] = (),
        models: Sequence[str] = (),
        families: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Select executable cases using exactly one supported selection mode."""
        return _selected_cases(
            self.cases,
            entries,
            requested_models=models,
            requested_families=families,
            excluded_profiles=self.excluded_profiles,
        )


def load_suite(path: Path) -> PerformanceSuite:
    """Load, expand, and validate one performance qualification suite."""
    source = Path(path).resolve()
    definition = _read_definition(source)
    cases = _cases(definition)
    excluded_profiles = _excluded_profiles(definition)
    _validate_coverage(cases, excluded_profiles)
    return PerformanceSuite(
        source=source,
        definition=definition,
        cases=tuple(cases),
        excluded_profiles=excluded_profiles,
    )


def is_l0_profile(name: str) -> bool:
    """Return whether a canonical model name denotes an L0 duplicate profile."""
    return L0_PROFILE_PATTERN.search(name) is not None


def validate_case(case: Mapping[str, Any]) -> None:
    """Validate one fully resolved executable performance case."""
    _validate_case_shape(case)


def validate_release_coverage(
    cases: Sequence[Mapping[str, Any]],
    excluded_profiles: Iterable[str] = (),
) -> None:
    """Validate cases against all release-ready non-L0 model profiles."""
    _validate_coverage(cases, excluded_profiles)


def _read_definition(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PerformanceSuiteError(f"cannot read performance suite {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PerformanceSuiteError("performance suite must contain a YAML object")
    if value.get("schema_version") != SUITE_SCHEMA:
        raise PerformanceSuiteError(f"suite schema_version must be {SUITE_SCHEMA!r}")
    return value


def _cases(suite: Mapping[str, Any]) -> list[dict[str, Any]]:
    defaults = suite.get("defaults", {})
    configured = suite.get("entries")
    additional_profiles = suite.get("additional_profiles", [])
    if not isinstance(defaults, Mapping):
        raise PerformanceSuiteError("suite defaults must be an object")
    if not isinstance(configured, list) or not configured:
        raise PerformanceSuiteError("suite entries must be a non-empty list")
    if not isinstance(additional_profiles, list):
        raise PerformanceSuiteError("suite additional_profiles must be a list")
    cases: list[dict[str, Any]] = []
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerformanceSuiteError("every suite entry must be an object")
        merged = _merge_case(defaults, raw)
        _validate_case_shape(merged)
        cases.append(merged)
    cases.extend(_additional_profile_cases(cases, additional_profiles))
    _validate_unique_ids(cases)
    return cases


def _excluded_profiles(suite: Mapping[str, Any]) -> dict[str, str]:
    configured = suite.get("excluded_profiles", [])
    if not isinstance(configured, list):
        raise PerformanceSuiteError("suite excluded_profiles must be a list")
    exclusions: dict[str, str] = {}
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerformanceSuiteError("every excluded profile must be an object")
        unsupported = sorted(set(raw) - {"model", "reason"})
        if unsupported:
            raise PerformanceSuiteError(
                "excluded profile has unsupported fields: " + ", ".join(unsupported)
            )
        model = raw.get("model")
        reason = raw.get("reason")
        if not isinstance(model, str) or not model.strip():
            raise PerformanceSuiteError("excluded profile model must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise PerformanceSuiteError(
                f"excluded profile {model} reason must be a non-empty string"
            )
        model = model.strip()
        if model in exclusions:
            raise PerformanceSuiteError(f"duplicate excluded profile: {model}")
        exclusions[model] = reason.strip()
    return exclusions


def _additional_profile_cases(
    base_cases: Sequence[Mapping[str, Any]],
    configured: Sequence[Any],
) -> list[dict[str, Any]]:
    templates = {str(case["id"]): case for case in base_cases}
    cases: list[dict[str, Any]] = []
    allowed = {"id", "model", "inherit", "workload", "measurement", "baseline"}
    for raw in configured:
        if not isinstance(raw, Mapping):
            raise PerformanceSuiteError("every additional profile must be an object")
        unsupported = sorted(set(raw) - allowed)
        if unsupported:
            raise PerformanceSuiteError(
                "additional profile has unsupported fields: " + ", ".join(unsupported)
            )
        model = raw.get("model")
        inherited_id = raw.get("inherit")
        if not isinstance(model, str) or not model.strip():
            raise PerformanceSuiteError("additional profile model must be a non-empty string")
        if not isinstance(inherited_id, str) or inherited_id not in templates:
            raise PerformanceSuiteError(
                f"additional profile {model} inherits unknown entry {inherited_id!r}"
            )
        configured_id = raw.get("id")
        if configured_id is not None and (
            not isinstance(configured_id, str) or not configured_id.strip()
        ):
            raise PerformanceSuiteError(f"additional profile {model} id must be a non-empty string")
        overrides = {key: value for key, value in raw.items() if key in {"measurement", "baseline"}}
        case = _merge_case(templates[inherited_id], overrides)
        case["id"] = configured_id or f"{inherited_id}@{model}"
        case["model"] = model
        workload = deepcopy(dict(case["workload"]))
        workload["testcase"] = model
        configured_workload = raw.get("workload")
        if configured_workload is not None:
            if not isinstance(configured_workload, Mapping):
                raise PerformanceSuiteError(
                    f"additional profile {model} workload must be an object"
                )
            workload.update(deepcopy(dict(configured_workload)))
        case["workload"] = workload
        _validate_case_shape(case)
        cases.append(case)
    return cases


def _merge_case(defaults: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(defaults))
    for key, value in case.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_case_shape(case: Mapping[str, Any]) -> None:
    required = (
        "id",
        "family",
        "operation",
        "model",
        "workload",
        "measurement",
        "baseline",
    )
    missing = [name for name in required if name not in case]
    if missing:
        raise PerformanceSuiteError(f"case is missing {', '.join(missing)}: {case}")
    _validate_workload(case)
    _validate_measurement(case)
    _validate_baseline(case)


def _validate_workload(case: Mapping[str, Any]) -> None:
    workload = case["workload"]
    if not isinstance(workload, Mapping):
        raise PerformanceSuiteError(f"case {case['id']} workload must be an object")
    unsupported = sorted(set(workload) - {"testcase", "request", "runtime"})
    if unsupported:
        raise PerformanceSuiteError(
            f"case {case['id']} workload has unsupported fields: " + ", ".join(unsupported)
        )
    testcase = workload.get("testcase")
    if not isinstance(testcase, str) or not testcase.strip():
        raise PerformanceSuiteError(
            f"case {case['id']} workload.testcase must be explicit; "
            "dataset workloads are not implemented yet"
        )
    request = workload.get("request", {})
    if not isinstance(request, Mapping):
        raise PerformanceSuiteError(f"case {case['id']} workload.request must be an object")
    runtime = workload.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise PerformanceSuiteError(f"case {case['id']} workload.runtime must be an object")


def _validate_measurement(case: Mapping[str, Any]) -> None:
    measurement = case["measurement"]
    if not isinstance(measurement, Mapping):
        raise PerformanceSuiteError(f"case {case['id']} measurement must be an object")
    for name in ("warmup", "iterations"):
        value = measurement.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if name == "warmup" else 1)
        ):
            raise PerformanceSuiteError(f"case {case['id']} measurement.{name} is invalid")


def _validate_baseline(case: Mapping[str, Any]) -> None:
    baseline = case["baseline"]
    if not isinstance(baseline, Mapping) or baseline.get("runner") not in {
        "hf-transformers",
        "task-reference",
    }:
        raise PerformanceSuiteError(f"case {case['id']} has an unsupported baseline runner")
    if baseline.get("runner") == "task-reference":
        adapter = str(baseline.get("adapter", ""))
        if adapter not in TASK_REFERENCE_ADAPTERS:
            raise PerformanceSuiteError(
                f"case {case['id']} has an unsupported task-reference adapter: {adapter}"
            )
        if not baseline.get("reference_backend"):
            raise PerformanceSuiteError(
                f"case {case['id']} task-reference baseline needs a reference_backend"
            )
        if not isinstance(baseline.get("adapter_options", {}), Mapping):
            raise PerformanceSuiteError(
                f"case {case['id']} task-reference adapter_options must be an object"
            )
        expected_mode = (
            "pytorch-eager"
            if adapter
            in {
                "nemo-asr",
                "nemo-tts",
                "pytorch-personaplex",
                "pytorch-timeseries",
                "upstream-elf",
                "upstream-lance",
                "upstream-sana-wm",
            }
            else "hf-eager"
        )
        if baseline.get("mode") != expected_mode:
            raise PerformanceSuiteError(
                f"case {case['id']} adapter {adapter} requires mode {expected_mode}"
            )
    token_policy = baseline.get("output_token_policy", "new-tokens")
    if token_policy not in {"new-tokens", "strip-start", "strip-start-and-eos"}:
        raise PerformanceSuiteError(f"case {case['id']} baseline output token policy is invalid")
    if baseline.get("padding", "longest") not in {"longest", "max-length"}:
        raise PerformanceSuiteError(f"case {case['id']} baseline padding is invalid")
    if baseline.get("precision") not in {None, "fp16", "fp32", "bf16"}:
        raise PerformanceSuiteError(f"case {case['id']} baseline precision is invalid")
    if baseline.get("model_class", "task") not in {"task", "auto"}:
        raise PerformanceSuiteError(f"case {case['id']} baseline model class is invalid")
    if baseline.get("generation_method", "generate") not in {
        "generate",
        "ar-generate",
    }:
        raise PerformanceSuiteError(f"case {case['id']} baseline generation method is invalid")
    if not isinstance(baseline.get("local_files_only", False), bool):
        raise PerformanceSuiteError(f"case {case['id']} baseline local_files_only must be boolean")
    if baseline.get("experts_implementation") not in {
        None,
        "eager",
        "batched_mm",
        "grouped_mm",
    }:
        raise PerformanceSuiteError(f"case {case['id']} baseline experts implementation is invalid")
    output_contract = baseline.get("output_contract", "exact-token-ids")
    if output_contract not in {
        "audio-shape",
        "exact-token-ids",
        "exact-text",
        "generated-token-count",
        "image-features-shape",
        "localization",
        "media-shape",
        "normalized-text",
        "ocr-text",
        "segmentation-shape",
        "token-agreement",
    }:
        raise PerformanceSuiteError(f"case {case['id']} baseline output contract is invalid")
    if output_contract == "token-agreement":
        for name in (
            "min_positional_token_agreement",
            "max_normalized_edit_distance",
        ):
            value = baseline.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise PerformanceSuiteError(
                    f"case {case['id']} token-agreement contract has invalid {name}"
                )
    if output_contract == "localization":
        bounded = {
            "min_localization_box_iou": (0.0, 1.0),
            "max_normalized_edit_distance": (0.0, 1.0),
        }
        for name, (minimum, maximum) in bounded.items():
            value = baseline.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not minimum <= float(value) <= maximum
            ):
                raise PerformanceSuiteError(
                    f"case {case['id']} localization contract has invalid {name}"
                )
        point_limit = baseline.get("max_localization_point_distance")
        if (
            isinstance(point_limit, bool)
            or not isinstance(point_limit, (int, float))
            or not math.isfinite(float(point_limit))
            or float(point_limit) < 0.0
        ):
            raise PerformanceSuiteError(
                f"case {case['id']} localization contract has invalid "
                "max_localization_point_distance"
            )
    if output_contract == "ocr-text":
        required = baseline.get("required_substrings")
        if (
            not isinstance(required, list)
            or not required
            or any(not isinstance(value, str) or not value.strip() for value in required)
        ):
            raise PerformanceSuiteError(
                f"case {case['id']} OCR output contract needs required_substrings"
            )
        limit = baseline.get("max_normalized_edit_distance")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not 0.0 <= float(limit) <= 1.0
        ):
            raise PerformanceSuiteError(
                f"case {case['id']} OCR output contract has an invalid edit-distance limit"
            )
    expected_timing = timing_contract(
        runner=str(baseline["runner"]),
        family=str(case["family"]),
    )
    for name in (
        "timing_scope",
        "input_preparation_included",
        "asset_loading_included",
    ):
        if baseline.get(name) != expected_timing[name]:
            raise PerformanceSuiteError(
                f"case {case['id']} baseline.{name} must be "
                f"{expected_timing[name]!r} for its reference"
            )
    mode = baseline.get("mode", "torch-compile")
    allowed_modes = (
        {"hf-eager", "pytorch-eager"}
        if baseline.get("runner") == "task-reference"
        else {"torch-compile", "hf-eager"}
    )
    if mode not in allowed_modes:
        raise PerformanceSuiteError(f"case {case['id']} baseline mode is invalid: {mode}")


def _validate_unique_ids(cases: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(case["id"]) for case in cases]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise PerformanceSuiteError(f"duplicate entry ids: {', '.join(duplicates)}")


def _validate_coverage(
    cases: Sequence[Mapping[str, Any]],
    excluded_profiles: Iterable[str] = (),
) -> None:
    catalog_profiles = {
        entry.name: (entry.family, entry.operation)
        for entry in ManifestCatalog().entries()
        if entry.status == "ready" and not is_l0_profile(entry.name)
    }
    excluded = set(excluded_profiles)
    invalid_exclusions = sorted(excluded - set(catalog_profiles))
    expected = {
        model: contract for model, contract in catalog_profiles.items() if model not in excluded
    }
    actual_models = [str(case["model"]) for case in cases]
    actual = set(actual_models)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(catalog_profiles))
    excluded_in_cases = sorted(actual & excluded)
    duplicates = sorted(model for model, count in Counter(actual_models).items() if count > 1)
    mismatched = sorted(
        (
            model,
            f"{case['family']}.{case['operation']}",
            f"{catalog_profiles[model][0]}.{catalog_profiles[model][1]}",
        )
        for case in cases
        if (model := str(case["model"])) in catalog_profiles
        and (str(case["family"]), str(case["operation"])) != catalog_profiles[model]
    )
    if missing or extra or duplicates or mismatched or invalid_exclusions or excluded_in_cases:
        details = _coverage_details(
            missing,
            extra,
            duplicates,
            mismatched,
            invalid_exclusions,
            excluded_in_cases,
        )
        raise PerformanceSuiteError(
            "suite profile coverage does not match the release-ready catalog "
            "(configured and L0 exclusions applied): " + "; ".join(details)
        )


def _coverage_details(
    missing: Sequence[str],
    extra: Sequence[str],
    duplicates: Sequence[str],
    mismatched: Sequence[tuple[str, str, str]],
    invalid_exclusions: Sequence[str],
    excluded_in_cases: Sequence[str],
) -> list[str]:
    details = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if extra:
        details.append("extra=" + ",".join(extra))
    if duplicates:
        details.append("duplicate=" + ",".join(duplicates))
    if mismatched:
        details.append(
            "family-operation="
            + ",".join(f"{model}:{actual}!={expected}" for model, actual, expected in mismatched)
        )
    if invalid_exclusions:
        details.append("invalid-exclusion=" + ",".join(invalid_exclusions))
    if excluded_in_cases:
        details.append("excluded-and-configured=" + ",".join(excluded_in_cases))
    return details


def _selected_cases(
    cases: Sequence[dict[str, Any]],
    requested: Sequence[str],
    *,
    requested_models: Sequence[str] = (),
    requested_families: Sequence[str] = (),
    excluded_profiles: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    requested = [value for value in requested if value]
    requested_models = list(model_selection.normalize_models(requested_models))
    requested_families = list(model_selection.normalize_models(requested_families))
    if sum(bool(value) for value in (requested, requested_models, requested_families)) > 1:
        raise PerformanceSuiteError("entry, model, and family selections are mutually exclusive")
    _validate_requested_cases(cases, requested)
    _validate_requested_models(
        cases,
        requested_models,
        excluded_profiles=excluded_profiles or {},
    )
    _validate_requested_families(cases, requested_families)
    if requested:
        selected = [case for case in cases if case["id"] in requested]
        selected.sort(key=lambda case: str(case["id"]))
        return selected
    if requested_models:
        model_order = {model: index for index, model in enumerate(requested_models)}
        selected = [case for case in cases if str(case["model"]) in model_order]
        selected.sort(
            key=lambda case: (
                model_order[str(case["model"])],
                str(case["id"]),
            )
        )
        return selected
    if requested_families:
        family_order = {family: index for index, family in enumerate(requested_families)}
        selected = [case for case in cases if str(case["family"]) in family_order]
        selected.sort(
            key=lambda case: (
                family_order[str(case["family"])],
                str(case["model"]),
                str(case["id"]),
            )
        )
        return selected
    selected = list(cases)
    selected.sort(key=lambda case: str(case["id"]))
    return selected


def _validate_requested_cases(cases: Sequence[Mapping[str, Any]], requested: Sequence[str]) -> None:
    known = {str(case["id"]) for case in cases}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PerformanceSuiteError(f"unknown entry ids: {', '.join(unknown)}")


def _validate_requested_models(
    cases: Sequence[Mapping[str, Any]],
    requested: Sequence[str],
    *,
    excluded_profiles: Mapping[str, str],
) -> None:
    known = {str(case["model"]) for case in cases}
    excluded = sorted(set(requested).intersection(excluded_profiles))
    if excluded:
        details = "; ".join(f"{model}: {excluded_profiles[model]}" for model in excluded)
        raise PerformanceSuiteError(f"excluded performance models: {details}")
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PerformanceSuiteError("models have no performance entries: " + ", ".join(unknown))


def _validate_requested_families(
    cases: Sequence[Mapping[str, Any]], requested: Sequence[str]
) -> None:
    known = {str(case["family"]) for case in cases}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PerformanceSuiteError(
            "model owners have no performance entries: " + ", ".join(unknown)
        )
