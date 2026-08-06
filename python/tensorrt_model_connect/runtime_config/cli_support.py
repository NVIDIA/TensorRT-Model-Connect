# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI support helpers for the two-flag config surface (``--config``, ``--set``).

This module lets the various entry points (``trtmc build``, the benchmark
scripts, tests) accept the fixed two-flag surface without growing per-knob
flags. All callers end up producing a :class:`LayerContribution` that feeds
:meth:`ConfigBundle.build`.

Intended shape:

- ``--config <file>`` → JSON (or YAML, via PyYAML if installed) loaded as
  a nested ``{namespace: {field: value}}`` dict. Values are already typed.
- ``--set <ns.field=value>`` → one token per invocation; repeatable.
  Values come in as raw strings and get coerced by the schema's declared
  ``type_tag``.

Both sources feed the SESSION_REQUEST layer by default; ``--set`` wins on
collision within that layer (last-write wins). A future ``--platform
<file>`` can reuse :func:`load_layered_file` to target PLATFORM_PROFILE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tensorrt_model_connect.runtime_config.schema_registry import (
    ConfigField,
    Layer,
    Schema,
    SchemaRegistry,
    _singleton_for_testing,
)
from tensorrt_model_connect.runtime_config.config_bundle import (
    ConfigBundle,
    LayerContribution,
    write_effective_config,
)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_layered_file(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a config profile from JSON or YAML.

    Returns the nested ``{namespace: {field: value}}`` dict verbatim. Type
    validation happens during :func:`build_cli_contribution` when the
    schema registry is consulted.

    Accepts:
      - ``.json`` — always supported (stdlib).
      - ``.yaml`` / ``.yml`` — requires PyYAML. Raises ``RuntimeError`` with
        a clear message if PyYAML is absent.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"--config file not found: {resolved}")
    raw = resolved.read_text(encoding="utf-8")

    suffix = resolved.suffix.lower()
    if suffix == ".json":
        data = json.loads(raw)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                f"--config {resolved}: YAML parsing requires PyYAML; "
                f"install it or convert the file to JSON"
            ) from exc
        data = yaml.safe_load(raw)
    else:
        raise ValueError(
            f"--config {resolved}: unsupported extension '{suffix}' "
            f"(expected .json, .yaml, or .yml)"
        )

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"--config {resolved}: top-level must be a mapping "
            f"(got {type(data).__name__})"
        )
    # Shallow validate: each namespace maps to a dict of fields.
    for ns, body in data.items():
        if not isinstance(ns, str):
            raise ValueError(
                f"--config {resolved}: namespace keys must be strings "
                f"(got {type(ns).__name__})"
            )
        if not isinstance(body, dict):
            raise ValueError(
                f"--config {resolved}: namespace '{ns}' must map to a dict "
                f"of fields (got {type(body).__name__})"
            )
    return data


# ---------------------------------------------------------------------------
# --set parsing
# ---------------------------------------------------------------------------


def parse_set_token(token: str) -> Tuple[str, str, str]:
    """Parse ``ns.field=value`` into ``(namespace, field, raw_value_str)``.

    The value is kept as a string — coercion happens in
    :func:`build_cli_contribution` once the schema type is known.
    """
    if "=" not in token:
        raise ValueError(
            f"--set expects 'ns.field=value' (got {token!r}; missing '=')"
        )
    key, _, value = token.partition("=")
    key = key.strip()
    if "." not in key:
        raise ValueError(
            f"--set expects 'ns.field=value' (got {token!r}; missing '.')"
        )
    namespace, _, field_name = key.partition(".")
    namespace = namespace.strip()
    field_name = field_name.strip()
    if not namespace or not field_name:
        raise ValueError(
            f"--set expects 'ns.field=value' (got {token!r}; empty namespace or field)"
        )
    return namespace, field_name, value


def parse_set_tokens(
    tokens: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    """Turn ``--set`` token list into nested raw-string dict.

    Later tokens within the same invocation win on collision — last-write
    semantics for the session layer.
    """
    out: Dict[str, Dict[str, str]] = {}
    for token in tokens:
        ns, field_name, raw_value = parse_set_token(token)
        out.setdefault(ns, {})[field_name] = raw_value
    return out


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"true", "1", "yes", "on"})
_BOOL_FALSE = frozenset({"false", "0", "no", "off"})


def coerce_scalar(raw: str, type_tag: str, where: str) -> Any:
    """Coerce a raw string value to the schema's declared scalar type.

    ``where`` appears in error messages for diagnostics (e.g.
    ``"triattention.kv_budget"``).
    """
    tag = type_tag.lower()
    if tag in ("int", "int32", "int64"):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{where}: expected integer, got {raw!r}"
            ) from exc
    if tag in ("float", "double"):
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{where}: expected float, got {raw!r}"
            ) from exc
    if tag == "bool":
        low = raw.strip().lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        raise ValueError(
            f"{where}: expected bool (true/false), got {raw!r}"
        )
    if tag in ("string", "str", "path"):
        return raw
    raise ValueError(
        f"{where}: schema declares unsupported type_tag {type_tag!r} "
        f"for --set coercion"
    )


# ---------------------------------------------------------------------------
# Merge -> LayerContribution
# ---------------------------------------------------------------------------


def _lookup_field(registry: SchemaRegistry, ns: str, field_name: str,
                  origin: str) -> ConfigField:
    schema: Optional[Schema] = registry.lookup(ns)
    if schema is None:
        raise ValueError(
            f"{origin}: unknown namespace '{ns}'. "
            f"Known: {', '.join(registry.registered_namespaces()) or '<none registered>'}"
        )
    for cfg_field in schema.fields:
        if cfg_field.name == field_name:
            return cfg_field
    known_fields = ", ".join(f.name for f in schema.fields)
    raise ValueError(
        f"{origin}: unknown field '{field_name}' in namespace '{ns}'. "
        f"Known: {known_fields}"
    )


def build_cli_contribution(
    *,
    config_file_values: Optional[Mapping[str, Mapping[str, Any]]] = None,
    set_tokens: Optional[Iterable[str]] = None,
    layer: Layer = Layer.SESSION_REQUEST,
    registry: Optional[SchemaRegistry] = None,
) -> LayerContribution:
    """Merge ``--config`` file values and ``--set`` tokens into one contribution.

    Within ``layer``, ``--set`` wins on collision — the file supplies
    defaults, the command line overrides for the session. The merged
    result is returned as a single :class:`LayerContribution` so the
    downstream :meth:`ConfigBundle.build` sees no same-layer collisions
    (which would otherwise be an error).

    Each namespace and field is validated against the registered schema
    to fail fast on typos. ``--set`` values are coerced from strings
    according to the schema's declared ``type_tag``.
    """
    reg = registry if registry is not None else _singleton_for_testing()
    merged: Dict[str, Dict[str, Any]] = {}

    if config_file_values:
        for ns, body in config_file_values.items():
            for field_name, value in body.items():
                _lookup_field(reg, ns, field_name, origin="--config")
                merged.setdefault(ns, {})[field_name] = value

    if set_tokens:
        raw_sets = parse_set_tokens(set_tokens)
        for ns, body in raw_sets.items():
            for field_name, raw_value in body.items():
                cfg_field = _lookup_field(reg, ns, field_name, origin="--set")
                coerced = coerce_scalar(
                    raw_value, cfg_field.type_tag,
                    where=f"--set {ns}.{field_name}",
                )
                merged.setdefault(ns, {})[field_name] = coerced

    return LayerContribution(layer=layer, values=merged)


# ---------------------------------------------------------------------------
# One-call convenience for CLI entry points
# ---------------------------------------------------------------------------


def resolve_cli_config(
    *,
    config_path: Optional[str | Path] = None,
    set_tokens: Optional[Iterable[str]] = None,
    extra_contributions: Optional[Iterable[LayerContribution]] = None,
    registry: Optional[SchemaRegistry] = None,
) -> ConfigBundle:
    """End-to-end helper: parse flags → merge → build a ConfigBundle.

    ``extra_contributions`` lets callers inject contributions from other
    layers (e.g. a platform profile loaded from a site-specific location,
    or a bundle's ``defaults:`` block read by the runtime).
    """
    config_values: Dict[str, Dict[str, Any]] = {}
    if config_path is not None:
        config_values = load_layered_file(config_path)

    session_contrib = build_cli_contribution(
        config_file_values=config_values,
        set_tokens=set_tokens,
        layer=Layer.SESSION_REQUEST,
        registry=registry,
    )
    contributions: List[LayerContribution] = []
    if extra_contributions:
        contributions.extend(extra_contributions)
    contributions.append(session_contrib)
    return ConfigBundle.build(contributions, registry=registry)


def write_effective_config_next_to(
    bundle: ConfigBundle, artifact_path: str | Path,
    suffix: str = ".effective_config.json",
) -> Path:
    """Write ``effective_config.json`` alongside an output artifact.

    For an output at ``foo/bar.bundle`` and default suffix, the artifact
    is ``foo/bar.effective_config.json``. Returns the written path.
    """
    resolved = Path(artifact_path)
    target = resolved.with_suffix(suffix)
    return write_effective_config(bundle, target)
