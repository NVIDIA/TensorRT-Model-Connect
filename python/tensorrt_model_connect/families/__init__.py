"""Auto-discover family plugins from this package.

Any .py file or package in this directory (excluding _-prefixed and base.py)
that exposes a module-level ``plugin`` attribute is automatically registered.
Adding a new family = drop a .py file or package, zero edits to shared files.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None

if TYPE_CHECKING:
    from .base import FamilyPlugin


_PLUGINS_DISCOVERED = False
_PLUGIN_CACHE: dict[str, "FamilyPlugin | None"] = {}
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
    diffusion_pipeline_classes: frozenset[str]


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


class _LazyPluginList(list["FamilyPlugin"]):
    def _materialize(self) -> None:
        _ensure_discovered()

    def __iter__(self):
        self._materialize()
        return super().__iter__()

    def __len__(self):
        self._materialize()
        return super().__len__()

    def __getitem__(self, index):
        self._materialize()
        return super().__getitem__(index)

    def __bool__(self):
        self._materialize()
        return super().__len__() != 0

    def __eq__(self, other):
        self._materialize()
        return super().__eq__(other)

    def __repr__(self):
        self._materialize()
        return super().__repr__()


_ALL_PLUGINS: list["FamilyPlugin"] = _LazyPluginList()


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
        plugin_id = raw.get("id") or raw.get("plugin") or index_path.parent.name
        if not isinstance(plugin_id, str) or not plugin_id:
            continue

        aliases = set(_metadata_strings(raw.get("aliases")))
        aliases.add(plugin_id)
        aliases.add(index_path.parent.name)
        normalized_aliases = frozenset(_normalize_key(value) for value in aliases)
        compact_aliases = frozenset(_compact_key(value) for value in aliases)

        prefixes = set(_metadata_strings(raw.get("prefixes")))
        normalized_prefixes = frozenset(_normalize_key(value) for value in prefixes)
        compact_prefixes = frozenset(_compact_key(value) for value in prefixes)

        metadata.append(_FamilyMetadata(
            id=plugin_id,
            import_module=index_path.parent.name,
            aliases=normalized_aliases,
            compact_aliases=compact_aliases,
            prefixes=normalized_prefixes,
            compact_prefixes=compact_prefixes,
            diffusion_pipeline_classes=_metadata_strings(
                raw.get("diffusion_pipeline_classes")
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


def _load_plugin_from_module(module_name: str) -> "FamilyPlugin | None":
    if module_name in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[module_name]
    try:
        mod = importlib.import_module(f"{__name__}.{module_name}")
    except ImportError:
        _PLUGIN_CACHE[module_name] = None
        return None
    plugin = getattr(mod, "plugin", None)
    _PLUGIN_CACHE[module_name] = plugin
    return plugin


def load_plugin_by_id(plugin_id: str) -> "FamilyPlugin | None":
    """Load one family plugin by model-owned id without scanning all metadata."""
    module_name = _normalize_key(plugin_id)
    if not module_name:
        return None

    index_path = Path(__file__).parent / module_name / "MODEL.toml"
    if not index_path.is_file():
        return None

    raw = _read_model_toml(index_path)
    declared_id = raw.get("id") or raw.get("plugin") or index_path.parent.name
    if not isinstance(declared_id, str) or _normalize_key(declared_id) != module_name:
        return None
    return _load_plugin_from_module(index_path.parent.name)


def available_plugin_ids() -> list[str]:
    """Return declared family ids without importing family plugin modules."""
    return sorted(meta.id for meta in _load_family_metadata())


def _discover_plugins() -> None:
    # Scan every .py module or package in this directory.
    _pkg_dir = str(Path(__file__).parent)
    for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
        # Skip private modules and the base protocol definition.
        if _name.startswith("_") or _name == "base":
            continue
        try:
            _mod = importlib.import_module(f"{__name__}.{_name}")
        except ImportError:
            # Skip plugins whose dependencies (e.g. tensorrt) are not installed.
            continue
        _plugin = getattr(_mod, "plugin", None)
        if _plugin is not None:
            list.append(_ALL_PLUGINS, _plugin)


def _ensure_discovered() -> None:
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED or not isinstance(_ALL_PLUGINS, _LazyPluginList):
        return
    _PLUGINS_DISCOVERED = True
    _discover_plugins()


def find_plugin(model_type: object) -> "FamilyPlugin | None":
    """Find the first plugin that matches a model type or config object."""
    model_type_str = str(getattr(model_type, "model_type", model_type))
    if not isinstance(_ALL_PLUGINS, _LazyPluginList):
        for p in _ALL_PLUGINS:
            matches_config = getattr(p, "matches_config", None)
            if callable(matches_config) and matches_config(model_type):
                return p
            if p.matches(model_type_str):
                return p
        return None

    plugin = load_plugin_by_id(model_type_str)
    if plugin is not None and plugin.matches(model_type_str):
        return plugin

    for module_name in _candidate_module_names(model_type_str):
        plugin = _load_plugin_from_module(module_name)
        if plugin is not None and plugin.matches(model_type_str):
            return plugin
    return None


def find_diffusion_plugin(pipeline_class: str) -> "FamilyPlugin | None":
    """Find the first plugin that handles the given diffusers pipeline class.

    Plugins declare supported pipeline classes via a ``pipeline_classes``
    attribute (list of class name strings). This enables auto-discovery
    without a hardcoded mapping dict.
    """
    if not isinstance(_ALL_PLUGINS, _LazyPluginList):
        for p in _ALL_PLUGINS:
            classes = getattr(p, 'pipeline_classes', None)
            if classes and pipeline_class in classes:
                return p
        return None

    for meta in _load_family_metadata():
        if pipeline_class not in meta.diffusion_pipeline_classes:
            continue
        plugin = _load_plugin_from_module(meta.import_module)
        classes = getattr(plugin, 'pipeline_classes', None) if plugin is not None else None
        if classes and pipeline_class in classes:
            return plugin
    return None
