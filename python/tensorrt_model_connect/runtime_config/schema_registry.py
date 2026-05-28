"""Python mirror of ``trtmc::config::SchemaRegistry``.

Features declare a namespaced schema (field metadata + defaults) and the
registry stores it. Value resolution happens elsewhere (see
``config_bundle.py``) — this module is strictly a metadata catalog.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


class Layer(enum.IntEnum):
    """Which config layer contributed or may contribute a value.

    Integer values must match the C++ ``trtmc::config::Layer`` enum. The
    priority order is implied by the integer value: higher = higher
    priority during merge.
    """

    SCHEMA_DEFAULT = 0
    BUILD_TIME = 1
    BUNDLE_DEFAULT = 2
    PLATFORM_PROFILE = 3
    SESSION_REQUEST = 4


# Stable layer names used in error messages, provenance dumps, and the
# effective_config.json file. Must match ``layer_name()`` in config_bundle.cpp.
_LAYER_NAMES: Dict[Layer, str] = {
    Layer.SCHEMA_DEFAULT:   "schema_default",
    Layer.BUILD_TIME:       "build_time",
    Layer.BUNDLE_DEFAULT:   "bundle_default",
    Layer.PLATFORM_PROFILE: "platform_profile",
    Layer.SESSION_REQUEST:  "session_request",
}


def layer_name(layer: Layer) -> str:
    return _LAYER_NAMES[layer]


@dataclass(frozen=True)
class ConfigField:
    """Metadata for one field inside a namespaced schema.

    ``type_tag`` is a human-readable string ("int32", "bool", "string",
    "list<string>") used in diagnostics and cross-language schema
    comparison. Runtime type checking inspects the default value and
    any contribution directly; the tag does not drive conversion.

    ``validator`` is optional; when present it must return ``True`` for
    valid values. A ``False`` return raises at merge time with
    ``namespace.field`` in the message.
    """

    name: str
    type_tag: str
    default: Any
    allowed_layers: frozenset  # frozenset[Layer]
    validator: Optional[Callable[[Any], bool]] = None


@dataclass(frozen=True)
class Schema:
    """A namespaced bundle of fields."""

    namespace: str
    fields: tuple  # tuple[ConfigField, ...]


class SchemaRegistry:
    """Process-wide singleton. Fail-fast on authoring mistakes."""

    def __init__(self) -> None:
        self._schemas: Dict[str, Schema] = {}

    def register_schema(self, schema: Schema) -> None:
        if not schema.namespace:
            raise ValueError("Cannot register schema with empty namespace")
        if not schema.fields:
            raise ValueError(
                f"Cannot register schema with no fields for namespace: {schema.namespace}"
            )
        seen: set = set()
        for cfg_field in schema.fields:
            if not cfg_field.name:
                raise ValueError(
                    f"Field with empty name in namespace: {schema.namespace}"
                )
            if cfg_field.name in seen:
                raise ValueError(
                    f"Duplicate field '{cfg_field.name}' in namespace: {schema.namespace}"
                )
            seen.add(cfg_field.name)
            if Layer.SCHEMA_DEFAULT in cfg_field.allowed_layers:
                raise ValueError(
                    f"Field {schema.namespace}.{cfg_field.name} declares "
                    f"SchemaDefault in allowed_layers; that layer is reserved "
                    f"for the baked-in default and cannot be contributed."
                )
            if not cfg_field.allowed_layers:
                raise ValueError(
                    f"Field {schema.namespace}.{cfg_field.name} has empty "
                    f"allowed_layers; at least one layer must be permitted."
                )
        if schema.namespace in self._schemas:
            raise ValueError(
                f"Duplicate config schema for namespace: {schema.namespace}"
            )
        self._schemas[schema.namespace] = schema

    def lookup(self, namespace: str) -> Optional[Schema]:
        return self._schemas.get(namespace)

    def registered_namespaces(self) -> List[str]:
        return sorted(self._schemas.keys())

    def clear_for_testing(self) -> None:
        self._schemas.clear()


# Process-wide singleton. Mirrors the static-local in C++.
_REGISTRY = SchemaRegistry()


def register_schema(schema: Schema) -> None:
    """Register ``schema`` with the process-wide singleton."""
    _REGISTRY.register_schema(schema)


def lookup(namespace: str) -> Optional[Schema]:
    return _REGISTRY.lookup(namespace)


def registered_namespaces() -> List[str]:
    return _REGISTRY.registered_namespaces()


def clear_for_testing() -> None:
    _REGISTRY.clear_for_testing()


def _singleton_for_testing() -> SchemaRegistry:
    """Test-only hatch: hand back the underlying registry object."""
    return _REGISTRY
