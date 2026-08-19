# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve model-owned family modules from their local metadata."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None

_METADATA_CACHE: list["_FamilyMetadata"] | None = None
_METADATA_INDEX_CACHE: "_FamilyMetadataIndex | None" = None


@dataclass(frozen=True)
class _FamilyMetadata:
    id: str
    import_module: str
    aliases: frozenset[str]
    compact_aliases: frozenset[str]
    prefixes: frozenset[str]
    compact_prefixes: frozenset[str]
    capabilities: frozenset[str]
    architecture_patterns: frozenset[str]
    diffusion_pipeline_classes: frozenset[str]
    nemo_target_patterns: frozenset[str]
    nemo_model_type: str
    nemo_archive_adapter: str = ""
    hf_allow_patterns: tuple[str, ...] = ()
    hf_required_files: tuple[str, ...] = ()
    hf_warm_dependencies: tuple[str, ...] = ()
    hf_warm_files: tuple[str, ...] = ()
    config_adapter: str = ""
    model_dir_adapter: str = ""
    debug_runner: str = ""
    debug_runtime_strategies: frozenset[str] = frozenset()
    python_profile_specs: tuple[str, ...] = ()
    default_execution_profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ModuleCandidate:
    id: str
    import_module: str


@dataclass(frozen=True)
class _FamilyMetadataIndex:
    aliases: dict[str, tuple[_ModuleCandidate, ...]]
    compact_aliases: dict[str, tuple[_ModuleCandidate, ...]]
    prefixes: dict[str, tuple[_ModuleCandidate, ...]]
    compact_prefixes: dict[str, tuple[_ModuleCandidate, ...]]


def _normalize_key(value: str) -> str:
    return (value or "").lower().replace("-", "_").replace(".", "_")


def _compact_key(value: str) -> str:
    return _normalize_key(value).replace("_", "")


def _read_model_toml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)

    # Python 3.10 fallback for this repo's flat string/list metadata shape.
    data: dict[str, str | list[str]] = {}
    current_key: str | None = None
    current_values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if current_key is not None:
            if "]" in line:
                before_close = line.split("]", 1)[0]
                current_values.extend(_parse_toml_string_values(before_close))
                data[current_key] = current_values
                current_key = None
                current_values = []
            else:
                current_values.extend(_parse_toml_string_values(line))
            continue
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if value.startswith("["):
            value = value[1:]
            if "]" in value:
                data[key] = _parse_toml_string_values(value.split("]", 1)[0])
            else:
                current_key = key
                current_values = _parse_toml_string_values(value)
        elif value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1]
    return data


def _parse_toml_string_values(value: str) -> list[str]:
    values: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item.startswith('"') and item.endswith('"'):
            values.append(item[1:-1])
    return values


def _metadata_strings(raw: object) -> frozenset[str]:
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(value for value in raw if isinstance(value, str) and value)


def _load_family_metadata() -> list[_FamilyMetadata]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE

    metadata: list[_FamilyMetadata] = []
    pkg_dir = Path(__file__).parent
    for index_path in sorted(pkg_dir.glob("*/MODEL.toml")):
        raw = _read_model_toml(index_path)
        family_id = raw.get("id")
        if not isinstance(family_id, str) or not family_id:
            raise ValueError(f"Missing model id in {index_path}")
        if family_id != index_path.parent.name:
            raise ValueError(
                f"Model id {family_id!r} must match owner folder "
                f"{index_path.parent.name!r}"
            )

        aliases = set(_metadata_strings(raw.get("aliases")))
        aliases.add(family_id)
        aliases.add(index_path.parent.name)
        normalized_aliases = frozenset(_normalize_key(value) for value in aliases)
        compact_aliases = frozenset(_compact_key(value) for value in aliases)

        prefixes = set(_metadata_strings(raw.get("prefixes")))
        normalized_prefixes = frozenset(_normalize_key(value) for value in prefixes)
        compact_prefixes = frozenset(_compact_key(value) for value in prefixes)

        metadata.append(_FamilyMetadata(
            id=family_id,
            import_module=index_path.parent.name,
            aliases=normalized_aliases,
            compact_aliases=compact_aliases,
            prefixes=normalized_prefixes,
            compact_prefixes=compact_prefixes,
            capabilities=_metadata_strings(raw.get("capabilities")),
            architecture_patterns=_metadata_strings(raw.get("architecture_patterns")),
            diffusion_pipeline_classes=_metadata_strings(
                raw.get("diffusion_pipeline_classes")
            ),
            nemo_target_patterns=_metadata_strings(
                raw.get("nemo_target_patterns")
            ),
            nemo_model_type=raw.get("nemo_model_type", "")
            if isinstance(raw.get("nemo_model_type"), str) else "",
            nemo_archive_adapter=raw.get("nemo_archive_adapter", "")
            if isinstance(raw.get("nemo_archive_adapter"), str) else "",
            hf_allow_patterns=tuple(_metadata_strings(raw.get("hf_allow_patterns"))),
            hf_required_files=tuple(
                _metadata_strings(raw.get("hf_required_files"))
            ),
            hf_warm_dependencies=tuple(
                _metadata_strings(raw.get("hf_warm_dependencies"))
            ),
            hf_warm_files=tuple(
                _metadata_strings(raw.get("hf_warm_files"))
            ),
            config_adapter=raw.get("config_adapter", "")
            if isinstance(raw.get("config_adapter"), str) else "",
            model_dir_adapter=raw.get("model_dir_adapter", "")
            if isinstance(raw.get("model_dir_adapter"), str) else "",
            debug_runner=raw.get("debug_runner", "")
            if isinstance(raw.get("debug_runner"), str) else "",
            debug_runtime_strategies=_metadata_strings(
                raw.get("debug_runtime_strategies")
            ),
            python_profile_specs=tuple(
                _metadata_strings(raw.get("python_profile_specs"))
            ),
            default_execution_profiles=tuple(
                _metadata_strings(raw.get("default_execution_profiles"))
            ),
        ))

    _METADATA_CACHE = metadata
    return metadata


def _add_index_value(
    index: dict[str, list[_ModuleCandidate]],
    key: str,
    candidate: _ModuleCandidate,
) -> None:
    if not key:
        return
    bucket = index.setdefault(key, [])
    if candidate not in bucket:
        bucket.append(candidate)


def _freeze_index(
    index: dict[str, list[_ModuleCandidate]],
) -> dict[str, tuple[_ModuleCandidate, ...]]:
    return {
        key: tuple(sorted(value, key=lambda candidate: candidate.id))
        for key, value in index.items()
    }


def _load_metadata_index() -> _FamilyMetadataIndex:
    global _METADATA_INDEX_CACHE
    if _METADATA_INDEX_CACHE is not None:
        return _METADATA_INDEX_CACHE

    aliases: dict[str, list[_ModuleCandidate]] = {}
    compact_aliases: dict[str, list[_ModuleCandidate]] = {}
    prefixes: dict[str, list[_ModuleCandidate]] = {}
    compact_prefixes: dict[str, list[_ModuleCandidate]] = {}

    for meta in _load_family_metadata():
        candidate = _ModuleCandidate(meta.id, meta.import_module)
        for alias in meta.aliases:
            _add_index_value(aliases, alias, candidate)
        for alias in meta.compact_aliases:
            _add_index_value(compact_aliases, alias, candidate)
        for prefix in meta.prefixes:
            _add_index_value(prefixes, prefix, candidate)
        for prefix in meta.compact_prefixes:
            _add_index_value(compact_prefixes, prefix, candidate)

    _METADATA_INDEX_CACHE = _FamilyMetadataIndex(
        aliases=_freeze_index(aliases),
        compact_aliases=_freeze_index(compact_aliases),
        prefixes=_freeze_index(prefixes),
        compact_prefixes=_freeze_index(compact_prefixes),
    )
    return _METADATA_INDEX_CACHE


def _candidate_prefix_keys(value: str) -> list[str]:
    return [value[:length] for length in range(len(value), 0, -1)]


def _append_candidate_modules(
    modules: list[str],
    seen: set[str],
    candidates: tuple[_ModuleCandidate, ...],
) -> None:
    for candidate in candidates:
        if candidate.import_module in seen:
            continue
        modules.append(candidate.import_module)
        seen.add(candidate.import_module)


def _candidate_module_names(model_type: str) -> list[str]:
    """Return likely family modules from indexed per-family MODEL.toml metadata."""
    normalized = _normalize_key(model_type)
    compact = _compact_key(model_type)
    index = _load_metadata_index()
    modules: list[str] = []
    seen: set[str] = set()

    _append_candidate_modules(modules, seen, index.aliases.get(normalized, ()))
    _append_candidate_modules(modules, seen, index.compact_aliases.get(compact, ()))

    prefix_scored: list[tuple[int, str, _ModuleCandidate]] = []
    for key in _candidate_prefix_keys(normalized):
        prefix_scored.extend(
            (len(key), candidate.id, candidate)
            for candidate in index.prefixes.get(key, ())
        )
    for key in _candidate_prefix_keys(compact):
        prefix_scored.extend(
            (len(key), candidate.id, candidate)
            for candidate in index.compact_prefixes.get(key, ())
        )
    for _, _, candidate in sorted(prefix_scored, key=lambda item: (-item[0], item[1])):
        _append_candidate_modules(modules, seen, (candidate,))
    return modules


def load_model_by_id(family_id: str) -> ModuleType | None:
    """Load one required ``models.<id>.model`` module."""
    module_name = _normalize_key(family_id)
    if not module_name:
        return None

    index_path = Path(__file__).parent / module_name / "MODEL.toml"
    if not index_path.is_file():
        return None

    raw = _read_model_toml(index_path)
    declared_id = raw.get("id")
    if not isinstance(declared_id, str) or not declared_id:
        raise ValueError(f"Missing model id in {index_path}")
    if declared_id != index_path.parent.name:
        raise ValueError(
            f"Model id {declared_id!r} must match owner folder "
            f"{index_path.parent.name!r}"
        )
    model = importlib.import_module(f"{__name__}.{index_path.parent.name}.model")
    for required in ("matches", "build"):
        if not callable(getattr(model, required, None)):
            raise TypeError(
                f"Family {declared_id!r} model.py must define {required}()"
            )
    return model


def _architecture_values(config: object) -> list[str]:
    values = getattr(config, "architectures", None)
    if values is None:
        raw = getattr(config, "raw", None)
        if isinstance(raw, dict):
            values = raw.get("architectures")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value]


def _candidate_module_names_from_config(config: object) -> list[str]:
    architectures = [value.lower() for value in _architecture_values(config)]
    if not architectures:
        return []

    modules: list[str] = []
    seen: set[str] = set()
    for meta in _load_family_metadata():
        for pattern in meta.architecture_patterns:
            pattern_key = pattern.lower()
            if not pattern_key:
                continue
            if any(pattern_key in architecture for architecture in architectures):
                if meta.import_module not in seen:
                    modules.append(meta.import_module)
                    seen.add(meta.import_module)
                break
    return modules


def available_family_ids() -> list[str]:
    """Return declared family ids without importing model modules."""
    return sorted(meta.id for meta in _load_family_metadata())


def family_probe_model_types() -> list[str]:
    """Return metadata-declared model_type seeds for discovery tools."""
    values: list[str] = []
    seen: set[str] = set()
    for meta in _load_family_metadata():
        for value in sorted(
            meta.aliases | meta.compact_aliases | meta.prefixes | meta.compact_prefixes
        ):
            if value not in seen:
                values.append(value)
                seen.add(value)
    return values


def family_hf_allow_patterns() -> list[str]:
    """Return extra HF snapshot patterns declared by family manifests."""
    patterns: list[str] = []
    for meta in _load_family_metadata():
        for pattern in meta.hf_allow_patterns:
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


def _metadata_pair(spec: str, field_name: str) -> tuple[str, str]:
    key, sep, value = spec.partition("|")
    if not sep or not key or not value:
        raise ValueError(
            f"Invalid {field_name} entry {spec!r}; expected 'key|value'"
        )
    return key, value


def _metadata_triple(spec: str, field_name: str) -> tuple[str, str, str]:
    first, sep, rest = spec.partition("|")
    second, sep2, third = rest.partition("|")
    if not sep or not sep2 or not first or not second or not third:
        raise ValueError(
            f"Invalid {field_name} entry {spec!r}; expected 'name|hf_id|filename'"
        )
    return first, second, third


def _metadata_warm_file_spec(
    spec: str,
) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in spec.split("|")]
    if len(parts) not in {3, 4} or any(not part for part in parts[:3]):
        raise ValueError(
            f"Invalid hf_warm_files entry {spec!r}; expected "
            "'name|hf_id|filename[|revision]'"
        )
    if len(parts) == 3:
        parts.append("")
    elif not parts[3]:
        raise ValueError(
            f"Invalid hf_warm_files entry {spec!r}; revision must be non-empty"
        )
    return parts[0], parts[1], parts[2], parts[3]


def _metadata_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {field_name} bool {value!r}; expected true or false"
    )


def family_default_execution_profiles(family: object) -> dict[str, str]:
    """Return phase -> Python profile defaults from family-owned metadata."""
    metadata = _matching_family_metadata(family)
    if not metadata:
        return {}

    defaults: dict[str, str] = {}
    for spec in metadata[0].default_execution_profiles:
        phase, profile = _metadata_pair(spec, "default_execution_profiles")
        defaults[phase] = profile
    return defaults


def family_python_profile_specs() -> dict[str, dict[str, object]]:
    """Return Python profile specs declared by family-owned MODEL.toml files.

    Entries use
    ``name|requirements|verification_script|system_site_packages|prebuild``.
    Paths are package-relative so profile assets stay under the owning family.
    """
    profiles: dict[str, dict[str, object]] = {}
    for meta in _load_family_metadata():
        for spec in meta.python_profile_specs:
            parts = [part.strip() for part in spec.split("|")]
            if len(parts) not in {3, 4, 5} or any(
                not part for part in parts[:3]
            ):
                raise ValueError(
                    f"Invalid python_profile_specs entry {spec!r} for family "
                    f"{meta.id}; expected "
                    "'name|requirements|verification_script|"
                    "system_site_packages|prebuild'"
                )
            name, requirements, verification_script_file = parts[:3]
            system_site_packages = (
                _metadata_bool(parts[3], "python_profile_specs")
                if len(parts) >= 4 else True
            )
            prebuild = (
                _metadata_bool(parts[4], "python_profile_specs")
                if len(parts) == 5 else True
            )
            if name in profiles:
                raise ValueError(
                    f"Python profile {name!r} is declared by multiple families"
                )
            profiles[name] = {
                "kind": "venv",
                "requirements": requirements,
                "verification_script_file": verification_script_file,
                "system_site_packages": system_site_packages,
                "prebuild": prebuild,
            }
    return profiles


def family_hf_required_files_by_id() -> dict[str, list[str]]:
    """Return required HF files declared by family manifests, grouped by HF id."""
    required: dict[str, list[str]] = {}
    for meta in _load_family_metadata():
        for spec in meta.hf_required_files:
            hf_id, filename = _metadata_pair(spec, "hf_required_files")
            required.setdefault(hf_id, []).append(filename)
    return required


def family_hf_warm_dependencies(family: object) -> list[tuple[str, str]]:
    """Return extra HF dependencies for a matched family as ``(name, hf_id)``."""
    metadata = _matching_family_metadata(family)
    if not metadata:
        return []
    return [
        _metadata_pair(spec, "hf_warm_dependencies")
        for spec in metadata[0].hf_warm_dependencies
    ]


def family_hf_warm_files(family: object) -> list[tuple[str, str, str]]:
    """Return extra HF files for a matched family as ``(name, hf_id, filename)``."""
    return [spec[:3] for spec in family_hf_warm_file_specs(family)]


def family_hf_warm_file_specs(
    family: object,
) -> list[tuple[str, str, str, str]]:
    """Return warm files as ``(name, hf_id, filename, revision)``.

    Three-field metadata remains valid and produces an empty revision so
    existing ``family_hf_warm_files`` consumers retain their original shape.
    """
    metadata = _matching_family_metadata(family)
    if not metadata:
        return []
    return [
        _metadata_warm_file_spec(spec)
        for spec in metadata[0].hf_warm_files
    ]


def _load_metadata_callable(meta: _FamilyMetadata, spec: str):
    module_spec, sep, attr_name = spec.partition("|")
    if not sep or not module_spec or not attr_name:
        raise ValueError(
            f"Invalid metadata callable {spec!r} for family {meta.id}; "
            "expected module.py|function"
        )
    module_name = module_spec[:-3] if module_spec.endswith(".py") else module_spec
    module_name = module_name.replace("/", ".")
    mod = importlib.import_module(f"{__name__}.{meta.import_module}.{module_name}")
    func = getattr(mod, attr_name)
    if not callable(func):
        raise TypeError(f"{spec!r} for family {meta.id} is not callable")
    return func


def _load_metadata_callable_from_file(meta: _FamilyMetadata, spec: str):
    module_spec, sep, attr_name = spec.partition("|")
    if not sep or not module_spec or not attr_name:
        raise ValueError(
            f"Invalid metadata callable {spec!r} for family {meta.id}; "
            "expected module.py|function"
        )
    module_name = module_spec[:-3] if module_spec.endswith(".py") else module_spec
    module_path = Path(__file__).parent / meta.import_module / Path(
        *module_name.split(".")
    ).with_suffix(".py")
    if not module_path.is_file():
        raise FileNotFoundError(
            f"Metadata callable module for family {meta.id} does not exist: "
            f"{module_path}"
        )
    spec_obj = importlib.util.spec_from_file_location(
        f"_trtmc_family_{meta.import_module}_{module_name.replace('.', '_')}",
        module_path,
    )
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"Cannot load metadata callable module {module_path}")
    mod = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)
    func = getattr(mod, attr_name)
    if not callable(func):
        raise TypeError(f"{spec!r} for family {meta.id} is not callable")
    return func


def resolve_config_from_model_dir(model_dir: str | Path) -> dict[str, Any] | None:
    """Ask family-owned config adapters to parse a non-HF model directory."""
    path = Path(model_dir)
    for meta in _load_family_metadata():
        if not meta.config_adapter:
            continue
        adapter = _load_metadata_callable(meta, meta.config_adapter)
        result = adapter(path)
        if result is None:
            continue
        if not isinstance(result, dict):
            raise TypeError(
                f"Config adapter {meta.config_adapter!r} for family {meta.id} "
                "must return a dict or None"
            )
        return result
    return None


def resolve_family_model_dir(model_dir: str | Path) -> str | None:
    """Ask family adapters to stage a non-flat model repository."""
    path = Path(model_dir)
    for meta in _load_family_metadata():
        if not meta.model_dir_adapter:
            continue
        adapter = _load_metadata_callable_from_file(
            meta, meta.model_dir_adapter)
        result = adapter(path)
        if result is None:
            continue
        if not isinstance(result, (str, Path)):
            raise TypeError(
                f"Model directory adapter {meta.model_dir_adapter!r} for "
                f"family {meta.id} must return a path or None"
            )
        return str(result)
    return None


def resolve_nemo_archive_model_dir(nemo_path: str | Path) -> str | None:
    """Ask family-owned NeMo archive adapters to synthesize a model dir."""
    path = Path(nemo_path)
    for meta in _load_family_metadata():
        if not meta.nemo_archive_adapter:
            continue
        adapter = _load_metadata_callable_from_file(meta, meta.nemo_archive_adapter)
        result = adapter(path)
        if result is None:
            continue
        if not isinstance(result, (str, Path)):
            raise TypeError(
                f"NeMo archive adapter {meta.nemo_archive_adapter!r} for "
                f"family {meta.id} must return a path or None"
            )
        return str(result)
    return None


def resolve_debug_runner(
    runtime_strategy: str,
    *,
    config: dict[str, Any],
    header: dict[str, Any],
    engine_plan: bytes,
    bundle_path: str | Path,
    distributed_communicator: object | None = None,
) -> object | None:
    """Create a family-owned debug runner for a runtime strategy if declared."""
    strategy = str(runtime_strategy or "")
    if not strategy:
        return None
    for meta in _load_family_metadata():
        if strategy not in meta.debug_runtime_strategies:
            continue
        if not meta.debug_runner:
            raise RuntimeError(
                f"Family {meta.id} declares debug strategy {strategy!r} "
                "without a debug_runner adapter"
            )
        factory = _load_metadata_callable(meta, meta.debug_runner)
        return factory(
            runtime_strategy=strategy,
            config=config,
            header=header,
            engine_plan=engine_plan,
            bundle_path=str(bundle_path),
            distributed_communicator=distributed_communicator,
        )
    return None


def _matching_family_metadata(model_type: object) -> list[_FamilyMetadata]:
    """Return metadata records whose owned aliases/prefixes match model_type."""
    model_type_str = str(getattr(model_type, "model_type", model_type))
    modules = _candidate_module_names(model_type_str)
    if not modules:
        return []
    by_module = {
        meta.import_module: meta
        for meta in _load_family_metadata()
    }
    return [
        by_module[module]
        for module in modules
        if module in by_module
    ]


def family_has_capability(model_type: object, capability: str) -> bool:
    """Return True when the matched family metadata opts into a capability."""
    metadata = _matching_family_metadata(model_type)
    if not metadata:
        return False
    return capability in metadata[0].capabilities


def resolve_family_id(model_type: object) -> str | None:
    """Return the matched family id from model-owned metadata, if any."""
    metadata = _matching_family_metadata(model_type)
    if not metadata:
        return None
    return metadata[0].id


def resolve_family_id_from_config(config: object) -> str | None:
    """Resolve one family from config metadata without importing model code."""
    model_type = str(getattr(config, "model_type", config))
    modules: list[str] = []
    for module in (
        *_candidate_module_names_from_config(config),
        *_candidate_module_names(model_type),
    ):
        if module not in modules:
            modules.append(module)
    by_module = {
        metadata.import_module: metadata
        for metadata in _load_family_metadata()
    }
    return next(
        (
            by_module[module].id
            for module in modules
            if module in by_module
        ),
        None,
    )


def resolve_diffusion_family_id(pipeline_class: str) -> str | None:
    """Return the family declaring one diffusion pipeline class."""

    for metadata in _load_family_metadata():
        if pipeline_class in metadata.diffusion_pipeline_classes:
            return metadata.id
    return None


def resolve_nemo_model_type(config: dict) -> str:
    """Resolve a NeMo config to a family-owned model_type.

    The extraction flow is generic; individual families own any target-string
    patterns through their MODEL.toml metadata.
    """
    target = str(config.get("target", "") or config.get("_target_", ""))
    target_key = target.lower()
    for meta in _load_family_metadata():
        if not meta.nemo_model_type or not meta.nemo_target_patterns:
            continue
        for pattern in meta.nemo_target_patterns:
            if pattern.lower() in target_key:
                return meta.nemo_model_type

    model_type = config.get("model_type", "")
    if isinstance(model_type, str) and model_type:
        return model_type
    return "unknown"


def find_model(config: object) -> ModuleType | None:
    """Return the metadata-bounded family module that claims ``config``."""
    if isinstance(config, str):
        family_id = resolve_family_id(config)
        return load_model_by_id(family_id) if family_id else None

    model_type = str(getattr(config, "model_type", config))
    candidates: list[str] = []
    for family_id in (
        *_candidate_module_names_from_config(config),
        *_candidate_module_names(model_type),
    ):
        if family_id not in candidates:
            candidates.append(family_id)
    for family_id in candidates:
        model = load_model_by_id(family_id)
        if model is not None and model.matches(config):
            return model
    return None
