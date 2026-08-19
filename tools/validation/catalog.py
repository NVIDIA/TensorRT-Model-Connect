# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve validation workloads and model manifests behind one catalog interface."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
import warnings

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tests.e2e_harness.manifest_loader import iter_manifest_paths, load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITES = REPO_ROOT / "tests" / "validation" / "workloads.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"
SUITES_SCHEMA = "trtmc.validation-suites/v2"
OWNER_VALIDATION_SCHEMA = "trtmc.validation-owner/v1"


def load_structured_file(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML catalog with a mapping at its root."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        raise RuntimeError("PyYAML is required for validation YAML files") from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def _load_owner_validation(
    suites: list[dict[str, Any]],
    models_dir: Path,
    *,
    owners: set[str] | None = None,
    require_all_suites: bool = True,
) -> None:
    suite_by_name = {str(suite["id"]): suite for suite in suites}
    if owners is None:
        manifests = load_manifest_records(models_dir)
    else:
        manifests = [
            manifest
            for owner in sorted(owners)
            for manifest in load_manifest_records(models_dir / owner)
        ]
    manifest_order = [str(manifest["name"]) for manifest in manifests]
    manifests_by_name: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        name = str(manifest["name"])
        if name in manifests_by_name:
            raise ValueError(f"duplicate model manifest name {name!r}")
        manifests_by_name[name] = manifest

    if owners is None:
        fragments = sorted(models_dir.glob("*/validation.yaml"))
    else:
        fragments = sorted(models_dir / owner / "validation.yaml" for owner in owners)
    bound_models: dict[str, set[str]] = {suite_id: set() for suite_id in suite_by_name}
    default_models: dict[str, set[str]] = {suite_id: set() for suite_id in suite_by_name}
    model_profiles: dict[str, dict[str, Any]] = {
        suite_id: {} for suite_id in suite_by_name
    }
    family_profiles: dict[str, dict[str, Any]] = {
        suite_id: {} for suite_id in suite_by_name
    }
    model_overrides: dict[str, dict[str, Any]] = {
        suite_id: {} for suite_id in suite_by_name
    }
    family_overrides: dict[str, dict[str, Any]] = {
        suite_id: {} for suite_id in suite_by_name
    }
    qualification_models: dict[str, dict[str, Any]] = {
        suite_id: {} for suite_id in suite_by_name
    }
    dataset_defaults: dict[str, tuple[str, str]] = {}

    owner_dirs = sorted(
        path
        for path in models_dir.iterdir()
        if path.is_dir() and (path / "MODEL.toml").is_file()
    )
    if owners is not None:
        owner_dirs = [path for path in owner_dirs if path.name in owners]
    missing_fragments = [
        owner / "validation.yaml"
        for owner in owner_dirs
        if not (owner / "validation.yaml").is_file()
    ]
    if missing_fragments:
        missing = ", ".join(str(path) for path in missing_fragments)
        raise ValueError(f"model owners are missing validation bindings: {missing}")

    for fragment in fragments:
        owner = fragment.parent.name
        if not (fragment.parent / "MODEL.toml").is_file():
            raise ValueError(f"{fragment}: validation owner has no MODEL.toml")
        raw = load_structured_file(fragment)
        if raw.get("schema_version") != OWNER_VALIDATION_SCHEMA:
            raise ValueError(
                f"{fragment}: schema_version must be {OWNER_VALIDATION_SCHEMA!r}"
            )
        unsupported = sorted(set(raw) - {"schema_version", "bindings"})
        if unsupported:
            raise ValueError(
                f"{fragment}: unsupported fields: {', '.join(unsupported)}"
            )
        bindings = raw.get("bindings")
        if not isinstance(bindings, dict):
            raise ValueError(f"{fragment}: bindings must be a mapping")

        for suite_id, binding in bindings.items():
            if suite_id not in suite_by_name:
                raise ValueError(f"{fragment}: unknown validation suite {suite_id!r}")
            if not isinstance(binding, dict):
                raise ValueError(f"{fragment}: binding {suite_id!r} must be a mapping")
            unsupported = sorted(
                set(binding)
                - {
                    "models",
                    "default_models",
                    "family_profile",
                    "model_profiles",
                    "family_override",
                    "model_overrides",
                    "qualification",
                    "dataset",
                }
            )
            if unsupported:
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} has unsupported fields: "
                    + ", ".join(unsupported)
                )
            models = binding.get("models")
            if (
                not isinstance(models, list)
                or not models
                or not all(isinstance(model, str) and model for model in models)
            ):
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} models must be a non-empty string list"
                )
            if len(set(models)) != len(models):
                raise ValueError(f"{fragment}: binding {suite_id!r} has duplicate models")
            dataset = binding.get("dataset")
            if dataset is not None:
                if not isinstance(dataset, dict) or set(dataset) != {"default_path"}:
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} dataset must contain only default_path"
                    )
                default_path = dataset["default_path"]
                if not isinstance(default_path, str) or not default_path.strip():
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} dataset default_path must be a non-empty string"
                    )
                previous = dataset_defaults.get(suite_id)
                if previous is not None:
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} duplicates the dataset default owned by "
                        f"{previous[0]!r}"
                    )
                dataset_defaults[suite_id] = (owner, default_path)
            defaults = binding.get("default_models", [])
            if not isinstance(defaults, list) or not all(
                isinstance(model, str) and model for model in defaults
            ):
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} default_models must be a string list"
                )
            if len(set(defaults)) != len(defaults):
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} has duplicate default models"
                )
            if not set(defaults) <= set(models):
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} defaults must also be bound models"
                )
            duplicate_bindings = sorted(set(models) & bound_models[suite_id])
            if duplicate_bindings:
                raise ValueError(
                    f"{fragment}: duplicate bindings for {suite_id!r}: "
                    + ", ".join(duplicate_bindings)
                )
            for model in models:
                manifest = manifests_by_name.get(model)
                if manifest is None:
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} references unknown model {model!r}"
                    )
                if manifest.get("family") != owner:
                    raise ValueError(
                        f"{fragment}: model {model!r} is owned by "
                        f"{manifest.get('family')!r}, not {owner!r}"
                    )

            for field in ("model_profiles", "model_overrides"):
                configured = binding.get(field, {})
                if not isinstance(configured, dict):
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} {field} must be a mapping"
                    )
                unknown = sorted(set(configured) - set(models))
                if unknown:
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} {field} references unbound "
                        f"models: {', '.join(unknown)}"
                    )
                invalid = sorted(
                    model
                    for model, value in configured.items()
                    if not isinstance(value, dict)
                )
                if invalid:
                    raise ValueError(
                        f"{fragment}: binding {suite_id!r} {field} values must be "
                        f"mappings: {', '.join(invalid)}"
                    )
                destination = (
                    model_profiles[suite_id]
                    if field == "model_profiles"
                    else model_overrides[suite_id]
                )
                destination.update(copy.deepcopy(configured))
            qualification = binding.get("qualification", {})
            if not isinstance(qualification, dict):
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} qualification must be a mapping"
                )
            unknown = sorted(set(qualification) - set(models))
            if unknown:
                raise ValueError(
                    f"{fragment}: binding {suite_id!r} qualification references "
                    f"unbound models: {', '.join(unknown)}"
                )
            for model, options in qualification.items():
                if not isinstance(options, dict):
                    raise ValueError(
                        f"{fragment}: qualification options for {model!r} must be a mapping"
                    )
                unsupported_options = sorted(
                    set(options) - {"reference_cache_identity"}
                )
                if unsupported_options:
                    raise ValueError(
                        f"{fragment}: qualification options for {model!r} have "
                        f"unsupported fields: {', '.join(unsupported_options)}"
                    )
                identity = options.get("reference_cache_identity")
                if identity is not None and (
                    not isinstance(identity, str) or not identity.strip()
                ):
                    raise ValueError(
                        f"{fragment}: qualification reference_cache_identity for "
                        f"{model!r} must be a non-empty string"
                    )
                qualification_models[suite_id][model] = copy.deepcopy(options)
            for field, destination in (
                ("family_profile", family_profiles[suite_id]),
                ("family_override", family_overrides[suite_id]),
            ):
                configured = binding.get(field)
                if configured is not None:
                    if not isinstance(configured, dict):
                        raise ValueError(
                            f"{fragment}: binding {suite_id!r} {field} must be a mapping"
                        )
                    if owner in destination:
                        raise ValueError(
                            f"{fragment}: duplicate {field} for owner {owner!r}"
                        )
                    destination[owner] = copy.deepcopy(configured)
            bound_models[suite_id].update(models)
            default_models[suite_id].update(defaults)

    for suite_id, suite in suite_by_name.items():
        if not bound_models[suite_id]:
            if require_all_suites:
                raise ValueError(f"validation suite {suite_id!r} has no model-owned bindings")
            continue
        selectors = suite.setdefault("selectors", {})
        if not isinstance(selectors, dict):
            raise ValueError(f"validation suite {suite_id!r} selectors must be a mapping")
        selectors["model_names"] = [
            model for model in manifest_order if model in bound_models[suite_id]
        ]
        suite["default_model_names"] = [
            model for model in manifest_order if model in default_models[suite_id]
        ]
        if model_profiles[suite_id]:
            suite["model_profiles"] = model_profiles[suite_id]
        if family_profiles[suite_id]:
            suite["family_profiles"] = family_profiles[suite_id]
        if model_overrides[suite_id] or family_overrides[suite_id]:
            suite["model_overrides"] = {
                "by_family": family_overrides[suite_id],
                "by_model": model_overrides[suite_id],
            }
        if qualification_models[suite_id]:
            suite["qualification_models"] = qualification_models[suite_id]
        dataset_default = dataset_defaults.get(suite_id)
        if dataset_default is not None:
            dataset = suite.get("dataset")
            if not isinstance(dataset, dict):
                raise ValueError(
                    f"validation suite {suite_id!r} has an owner-local dataset path but no dataset mapping"
                )
            if "default_path" in dataset:
                raise ValueError(
                    f"validation suite {suite_id!r} defines default_path both centrally and in owner "
                    f"{dataset_default[0]!r}"
                )
            dataset["default_path"] = dataset_default[1]


def qualification_models_for_owner(
    suite_id: str,
    owner: str,
    *,
    path: Path = DEFAULT_SUITES,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> tuple[str, ...]:
    """Return owner-local qualification models for one suite, failing closed."""
    models_dir = Path(models_dir)
    owner_dir = models_dir / owner
    if not owner or not (owner_dir / "MODEL.toml").is_file():
        raise ValueError(f"unknown model owner {owner!r}")

    suites = load_suites(
        path,
        models_dir,
        _owners={owner},
        _require_all_suites=False,
    )
    suite = suite_by_id(suites, suite_id)
    manifests = load_manifest_records(models_dir)
    owner_manifests = [model for model in manifests if model.get("family") == owner]
    qualified = suite.get("qualification_models", {})
    if not isinstance(qualified, dict):
        raise ValueError(
            f"validation suite {suite_id!r} qualification_models must be a mapping"
        )
    selected = tuple(
        str(model["name"])
        for model in owner_manifests
        if str(model["name"]) in qualified
    )

    selectors = copy.deepcopy(suite.get("selectors", {}))
    selectors.pop("model_names", None)
    shared_suite = {**suite, "selectors": selectors}
    applicable = [
        str(model["name"])
        for model in owner_manifests
        if suite_match_reason(shared_suite, model)[0]
    ]
    if applicable and not selected:
        raise ValueError(
            f"model owner {owner!r} matches validation suite {suite_id!r} but has no "
            "owner-local qualification"
        )
    return selected


def qualification_catalog(suites: list[dict[str, Any]]) -> dict[str, Any]:
    """Project model-owned qualification bindings into the model-first CLI view."""
    sample_limits: dict[str, int] = {}
    models: dict[str, dict[str, Any]] = {}
    for suite in suites:
        suite_id = str(suite["id"])
        qualified = suite.get("qualification_models", {})
        if not isinstance(qualified, dict):
            raise ValueError(
                f"validation suite {suite_id!r} qualification_models must be a mapping"
            )
        limit = suite.get("sample_limit")
        if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or (limit != -1 and limit <= 0)
        ):
            raise ValueError(
                f"validation suite {suite_id!r} sample_limit must be -1 or a positive integer"
            )
        if qualified and limit is None:
            raise ValueError(
                f"validation suite {suite_id!r} with qualification bindings must "
                "define sample_limit"
            )
        if limit is not None:
            sample_limits[suite_id] = limit
        if not qualified:
            continue
        for model, options in qualified.items():
            if not isinstance(options, dict):
                raise ValueError(
                    f"validation suite {suite_id!r} options for {model!r} must be a mapping"
                )
            spec = models.setdefault(str(model), {"workloads": []})
            spec["workloads"].append(suite_id)
            identity = options.get("reference_cache_identity")
            if identity is not None:
                previous = spec.get("reference_cache_identity")
                if previous is not None and previous != identity:
                    raise ValueError(
                        f"model {model!r} has conflicting reference_cache_identity values"
                    )
                spec["reference_cache_identity"] = identity
    return {
        "sample_limits": sample_limits,
        "models": {model: models[model] for model in sorted(models)},
    }


def load_suites(
    path: Path = DEFAULT_SUITES,
    models_dir: Path = DEFAULT_MODELS_DIR,
    *,
    _owners: set[str] | None = None,
    _require_all_suites: bool = True,
) -> list[dict[str, Any]]:
    """Load shared suite definitions plus strict model-owned bindings."""
    path = Path(path)
    data = load_structured_file(path)
    if data.get("schema_version") != SUITES_SCHEMA:
        raise ValueError(f"{path}: schema_version must be {SUITES_SCHEMA!r}")
    unsupported_root = sorted(set(data) - {"schema_version", "suites"})
    if unsupported_root:
        raise ValueError(f"{path}: unsupported fields: {', '.join(unsupported_root)}")
    suites = data.get("suites", [])
    if not isinstance(suites, list):
        raise ValueError(f"{path}: 'suites' must be a list")
    suites = copy.deepcopy(suites)
    seen: set[str] = set()
    for suite in suites:
        suite_id = suite.get("id")
        if not suite_id or not isinstance(suite_id, str):
            raise ValueError(f"{path}: every suite needs a string id")
        if suite_id in seen:
            raise ValueError(f"{path}: duplicate suite id {suite_id!r}")
        seen.add(suite_id)
        forbidden = sorted(
            set(suite)
            & {
                "default_model_names",
                "family_profiles",
                "model_profiles",
                "model_overrides",
            }
        )
        if forbidden:
            raise ValueError(
                f"{path}: suite {suite_id!r} contains model-owned fields: "
                + ", ".join(forbidden)
            )
        selectors = suite.get("selectors", {})
        if not isinstance(selectors, dict):
            raise ValueError(f"{path}: suite {suite_id!r} selectors must be a mapping")
        forbidden_selectors = sorted(
            set(selectors)
            & {
                "model_names",
                "exclude_model_names",
                "runtime_strategies",
                "families",
                "exclude_families",
            }
        )
        if forbidden_selectors:
            raise ValueError(
                f"{path}: suite {suite_id!r} contains model-owned selectors: "
                + ", ".join(forbidden_selectors)
            )
    _load_owner_validation(
        suites,
        Path(models_dir),
        owners=_owners,
        require_all_suites=_require_all_suites,
    )
    return suites


def suite_by_id(suites: list[dict[str, Any]], suite_id: str) -> dict[str, Any]:
    """Resolve one workload suite or report all known IDs."""
    for suite in suites:
        if suite["id"] == suite_id:
            return suite
    known = ", ".join(sorted(suite["id"] for suite in suites))
    raise ValueError(f"Unknown suite {suite_id!r}. Known suites: {known}")


def infer_reference_family(raw: dict[str, Any]) -> str:
    return str(raw.get("reference_family", "") or "")


def infer_user_contract(raw: dict[str, Any], reference_family: str) -> str:
    del reference_family
    return str(raw.get("user_contract", "") or "")


def _model_reference_cache(path: Path) -> dict[str, Any]:
    owner_path = (
        path.parents[2] / "MODEL.toml"
        if path.parent.name == "manifests"
        else path.parent / "MODEL.toml"
    )
    if not owner_path.is_file():
        return {}
    owner = tomllib.loads(owner_path.read_text(encoding="utf-8"))
    contract = owner.get("model_reference_cache", {})
    return copy.deepcopy(contract) if isinstance(contract, dict) else {}


def manifest_record(path: Path) -> dict[str, Any]:
    """Project one E2E manifest onto the fields needed by validation."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    model_name = str(raw.get("name", path.stem))
    testcases = raw.pop("testcases", [])
    if isinstance(testcases, list) and testcases:
        canonical = next(
            (
                testcase
                for testcase in testcases
                if isinstance(testcase, dict) and testcase.get("name") == model_name
            ),
            testcases[0],
        )
        if isinstance(canonical, dict):
            raw = {**raw, **canonical, "name": model_name}
    build_args = raw.get("build_args", {})
    validation_config = raw.get("task_eval", {})
    if not isinstance(validation_config, dict):
        validation_config = {}
    runtime_strategy = str(raw.get("runtime_strategy") or "")
    task_strategy = str(raw.get("task_strategy") or runtime_strategy)
    reference_family = str(validation_config.get("reference_family") or infer_reference_family(raw))
    reference_backend = str(
        validation_config.get("reference_backend") or raw.get("reference_backend", "") or ""
    )
    if not reference_backend:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                reference_backend = load_manifest(path).reference_backend
        except Exception:
            reference_backend = "hf_transformers"
    user_contract = str(
        validation_config.get("user_contract") or infer_user_contract(raw, reference_family)
    )
    distributed = raw.get("distributed_runtime", {})
    requires_multi_device = bool(distributed.get("enabled")) or (
        str(raw.get("ci_tier", "")) == "multi_device"
    )
    return {
        "name": model_name,
        "manifest": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "hf_id": raw.get("hf_id") or raw.get("model_id", ""),
        "hf_revision": str(raw.get("hf_revision", "") or ""),
        "bundle": raw.get("bundle", f"{path.stem}.bundle"),
        "family": raw.get("family", ""),
        "runtime_strategy": runtime_strategy,
        "task_strategy": task_strategy,
        "reference_family": reference_family,
        "reference_backend": reference_backend,
        "execution_profiles": (
            raw.get("execution_profiles", {})
            if isinstance(raw.get("execution_profiles", {}), dict)
            else {}
        ),
        "user_contract": user_contract,
        "test_category": raw.get("test_category", "e2e"),
        "ci_tier": raw.get("ci_tier", "default"),
        "core": bool(raw.get("core", False)),
        "skip": raw.get("skip", ""),
        "requires_multi_device": requires_multi_device,
        "l0_replacement": raw.get("l0_replacement", ""),
        "gated": bool(raw.get("gated", False)),
        "trust_remote_code": bool(raw.get("trust_remote_code", False)),
        "max_cache_length": raw.get(
            "max_cache_length",
            (build_args.get("max_cache_length", 256) if isinstance(build_args, dict) else 256),
        ),
        "precision": raw.get("precision", "fp32"),
        "fp32_layers": raw.get("fp32_layers", []),
        "quantization": raw.get("quantization", {}),
        "build_args": build_args if isinstance(build_args, dict) else {},
        "fp8_scales": raw.get("fp8_scales", ""),
        "image_height": raw.get("image_height"),
        "image_width": raw.get("image_width"),
        "video_num_frames": raw.get("video_num_frames"),
        "video_height": raw.get("video_height"),
        "video_width": raw.get("video_width"),
        "num_inference_steps": raw.get("num_inference_steps"),
        "task_eval": validation_config,
        "runtime_config": (
            raw.get("runtime_config", {}) if isinstance(raw.get("runtime_config", {}), dict) else {}
        ),
        "max_new_tokens": raw.get("max_new_tokens"),
        "model_reference_cache": _model_reference_cache(path),
    }


def load_manifest_records(
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> list[dict[str, Any]]:
    """Load the validation projection for every discovered model manifest."""
    return [manifest_record(path) for path in iter_manifest_paths(models_dir)]


def _selector_values(selectors: dict[str, Any], key: str) -> set[str]:
    values = selectors.get(key, [])
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def suite_match_reason(suite: dict[str, Any], model: dict[str, Any]) -> tuple[bool, str]:
    """Explain whether a model satisfies a workload suite's selectors."""
    selectors = suite.get("selectors", {})
    model_names = _selector_values(selectors, "model_names")
    exclude_model_names = _selector_values(selectors, "exclude_model_names")
    task_strategies = _selector_values(selectors, "task_strategies")
    runtime_strategies = _selector_values(selectors, "runtime_strategies")
    user_contracts = _selector_values(selectors, "user_contracts")
    exclude_user_contracts = _selector_values(selectors, "exclude_user_contracts")
    families = _selector_values(selectors, "families")
    exclude_families = _selector_values(selectors, "exclude_families")

    if model_names and model["name"] not in model_names:
        return False, f"model={model['name']} not selected"
    if exclude_model_names and model["name"] in exclude_model_names:
        return False, f"model={model['name']} excluded"
    if task_strategies and model["task_strategy"] not in task_strategies:
        return False, f"task_strategy={model['task_strategy']} not selected"
    if runtime_strategies and model["runtime_strategy"] not in runtime_strategies:
        return False, (f"runtime_strategy={model['runtime_strategy']} not selected")
    if user_contracts and model["user_contract"] not in user_contracts:
        return False, (f"user_contract={model['user_contract'] or '<empty>'} not selected")
    if exclude_user_contracts and model["user_contract"] in exclude_user_contracts:
        return False, f"user_contract={model['user_contract']} excluded"
    if families and model["family"] not in families:
        return False, f"family={model['family']} not selected"
    if exclude_families and model["family"] in exclude_families:
        return False, f"family={model['family']} excluded"
    if model.get("skip"):
        return False, f"manifest skip: {model['skip']}"
    return True, "selected"


def resolve_suite_for_model(suite: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Resolve manifest dimensions and family/model profiles for one run."""
    resolved = copy.deepcopy(suite)
    generation = dict(resolved.get("generation", {}))
    for key in (
        "image_height",
        "image_width",
        "video_num_frames",
        "video_height",
        "video_width",
        "num_inference_steps",
    ):
        value = model.get(key)
        if value is not None:
            generation[key] = value

    family_profiles = resolved.get("family_profiles", {})
    if not isinstance(family_profiles, dict):
        raise ValueError(f"Suite {suite['id']} family_profiles must be a mapping")
    family_profile = family_profiles.get(str(model.get("family", "")), {})
    if not isinstance(family_profile, dict):
        raise ValueError(
            f"Suite {suite['id']} family profile for {model.get('family')} must be a mapping"
        )

    profiles = resolved.get("model_profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Suite {suite['id']} model_profiles must be a mapping")
    profile = profiles.get(str(model.get("name", "")), {})
    if not isinstance(profile, dict):
        raise ValueError(f"Suite {suite['id']} profile for {model.get('name')} must be a mapping")
    for owner, source in (
        (model.get("family"), family_profile),
        (model.get("name"), profile),
    ):
        profile_generation = source.get("generation", {})
        if not isinstance(profile_generation, dict):
            raise ValueError(
                f"Suite {suite['id']} generation profile for {owner} must be a mapping"
            )
        generation.update(profile_generation)
    resolved["generation"] = generation

    gates = dict(resolved.get("gates", {}))
    for owner, source in (
        (model.get("family"), family_profile),
        (model.get("name"), profile),
    ):
        profile_gates = source.get("gates", {})
        if not isinstance(profile_gates, dict):
            raise ValueError(f"Suite {suite['id']} gate profile for {owner} must be a mapping")
        gates.update(profile_gates)
    resolved["gates"] = gates
    sample_acceptance = dict(resolved.get("sample_acceptance", {}))
    for owner, source in (
        (model.get("family"), family_profile),
        (model.get("name"), profile),
    ):
        profile_acceptance = source.get("sample_acceptance", {})
        if not isinstance(profile_acceptance, dict):
            raise ValueError(
                f"Suite {suite['id']} sample_acceptance profile for {owner} "
                "must be a mapping"
            )
        sample_acceptance.update(profile_acceptance)
    if sample_acceptance:
        resolved["sample_acceptance"] = sample_acceptance
    else:
        resolved.pop("sample_acceptance", None)
    if gates or sample_acceptance:
        resolved["gate_policy"] = "blocking"
    return resolved
