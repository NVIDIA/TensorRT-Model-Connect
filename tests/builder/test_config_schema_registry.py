# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# ISO 26262 Traceability
# =============================================================================
# Trace ID:       UT-CFG-PY-01
# Architecture:   ARCH-CFG-001
# Unit Design:    UD-CFG-REG-01
# Intent:         Python mirror of SchemaRegistry + ConfigBundle — registration
#                 rules, layer priority merge, allowlist enforcement, provenance,
#                 and effective_config.json round-trip.
# Preconditions:  None (no GPU, no TRT, no filesystem beyond tmp_path).
# Postconditions: Python registry enforces the same rules as the C++ side;
#                 bundle merge priority is identical; effective-config dump
#                 round-trips through JSON.
# =============================================================================

"""Unit tests for the Python config registry.

These tests mirror ``tests/cpp/test_config_schema_registry.cpp`` case-for-case
so the two implementations stay in semantic sync. No TRT, no GPU; safe under
``pytest tests/builder/`` in any environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from tensorrt_model_connect.runtime_config import (
        ConfigBundle,
        ConfigField,
        Layer,
        LayerContribution,
        Schema,
        clear_for_testing,
        lookup,
        register_schema,
        registered_namespaces,
        write_effective_config,
    )
    from tensorrt_model_connect.runtime_config.schema_registry import layer_name
except ImportError:  # pragma: no cover - environment gate
    pytest.skip("tensorrt_model_connect.runtime_config not importable", allow_module_level=True)


# ---- helpers ---------------------------------------------------------------


def int_field(name: str, default: int, layers):
    return ConfigField(
        name=name,
        type_tag="int32",
        default=default,
        allowed_layers=frozenset(layers),
        validator=None,
    )


def bool_field(name: str, default: bool, layers):
    return ConfigField(
        name=name,
        type_tag="bool",
        default=default,
        allowed_layers=frozenset(layers),
        validator=None,
    )


def contrib(layer: Layer, ns: str, field_name: str, value):
    return LayerContribution(layer=layer, values={ns: {field_name: value}})


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts with an empty registry and cleans up afterward."""
    clear_for_testing()
    yield
    clear_for_testing()


# ---- registration rules ----------------------------------------------------


def test_register_and_lookup():
    register_schema(Schema(
        namespace="ns_a",
        fields=(int_field("budget", 6144, [Layer.SESSION_REQUEST, Layer.BUNDLE_DEFAULT]),),
    ))
    schema = lookup("ns_a")
    assert schema is not None
    assert len(schema.fields) == 1
    assert lookup("ns_b") is None


def test_duplicate_namespace_throws():
    register_schema(Schema(
        namespace="dup",
        fields=(int_field("f", 0, [Layer.SESSION_REQUEST]),),
    ))
    with pytest.raises(ValueError, match="Duplicate"):
        register_schema(Schema(
            namespace="dup",
            fields=(int_field("f", 0, [Layer.SESSION_REQUEST]),),
        ))


def test_empty_namespace_throws():
    with pytest.raises(ValueError, match="empty namespace"):
        register_schema(Schema(
            namespace="",
            fields=(int_field("f", 0, [Layer.SESSION_REQUEST]),),
        ))


def test_empty_fields_throws():
    with pytest.raises(ValueError, match="no fields"):
        register_schema(Schema(namespace="ns", fields=()))


def test_schema_default_in_allowlist_throws():
    with pytest.raises(ValueError, match="SchemaDefault"):
        register_schema(Schema(
            namespace="ns",
            fields=(int_field("f", 0, [Layer.SCHEMA_DEFAULT, Layer.SESSION_REQUEST]),),
        ))


def test_empty_allowlist_throws():
    with pytest.raises(ValueError, match="empty allowed_layers"):
        register_schema(Schema(
            namespace="ns",
            fields=(int_field("f", 0, []),),
        ))


def test_duplicate_field_in_schema_throws():
    with pytest.raises(ValueError, match="Duplicate field"):
        register_schema(Schema(
            namespace="ns",
            fields=(
                int_field("f", 0, [Layer.SESSION_REQUEST]),
                int_field("f", 1, [Layer.SESSION_REQUEST]),
            ),
        ))


def test_registered_namespaces_sorted():
    for ns in ["zeta", "alpha", "mike"]:
        register_schema(Schema(
            namespace=ns,
            fields=(int_field("f", 0, [Layer.SESSION_REQUEST]),),
        ))
    assert registered_namespaces() == ["alpha", "mike", "zeta"]


# ---- bundle merge priority -------------------------------------------------


def test_merge_session_beats_platform():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE]),),
    ))
    bundle = ConfigBundle.build([
        contrib(Layer.PLATFORM_PROFILE, "ns", "k", 200),
        contrib(Layer.SESSION_REQUEST,  "ns", "k", 300),
    ])
    assert bundle.get("ns", "k") == 300
    assert bundle.source_of("ns", "k") == Layer.SESSION_REQUEST


def test_merge_platform_beats_bundle():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.BUNDLE_DEFAULT, Layer.PLATFORM_PROFILE]),),
    ))
    bundle = ConfigBundle.build([
        contrib(Layer.BUNDLE_DEFAULT,   "ns", "k", 200),
        contrib(Layer.PLATFORM_PROFILE, "ns", "k", 250),
    ])
    assert bundle.get("ns", "k") == 250
    assert bundle.source_of("ns", "k") == Layer.PLATFORM_PROFILE


def test_merge_bundle_beats_build():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT]),),
    ))
    bundle = ConfigBundle.build([
        contrib(Layer.BUILD_TIME,     "ns", "k", 200),
        contrib(Layer.BUNDLE_DEFAULT, "ns", "k", 250),
    ])
    assert bundle.get("ns", "k") == 250
    assert bundle.source_of("ns", "k") == Layer.BUNDLE_DEFAULT


def test_merge_fallback_to_schema_default():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.SESSION_REQUEST]),),
    ))
    bundle = ConfigBundle.build([])
    assert bundle.get("ns", "k") == 100
    assert bundle.source_of("ns", "k") == Layer.SCHEMA_DEFAULT


def test_merge_allowlist_violation_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.BUNDLE_DEFAULT]),),  # session NOT allowed
    ))
    with pytest.raises(ValueError, match="not permitted"):
        ConfigBundle.build([
            contrib(Layer.SESSION_REQUEST, "ns", "k", 300),
        ])


def test_merge_unknown_namespace_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 0, [Layer.SESSION_REQUEST]),),
    ))
    with pytest.raises(ValueError, match="unregistered namespace"):
        ConfigBundle.build([
            contrib(Layer.SESSION_REQUEST, "other_ns", "k", 1),
        ])


def test_merge_unknown_field_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 0, [Layer.SESSION_REQUEST]),),
    ))
    with pytest.raises(ValueError, match="unknown field"):
        ConfigBundle.build([
            contrib(Layer.SESSION_REQUEST, "ns", "other_field", 1),
        ])


def test_merge_validator_rejection_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(ConfigField(
            name="k",
            type_tag="int32",
            default=0,
            allowed_layers=frozenset({Layer.SESSION_REQUEST}),
            validator=lambda v: isinstance(v, int) and v > 0,
        ),),
    ))
    with pytest.raises(ValueError, match="Validator rejected"):
        ConfigBundle.build([
            contrib(Layer.SESSION_REQUEST, "ns", "k", -1),
        ])


# ---- typed access + provenance ---------------------------------------------


def test_bundle_get_multiple_kinds():
    register_schema(Schema(
        namespace="ns",
        fields=(
            int_field("budget", 6144, [Layer.SESSION_REQUEST]),
            bool_field("protect", True, [Layer.SESSION_REQUEST]),
        ),
    ))
    bundle = ConfigBundle.build([])
    assert bundle.get("ns", "budget") == 6144
    assert bundle.get("ns", "protect") is True


def test_bundle_get_unknown_namespace_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.SESSION_REQUEST]),),
    ))
    bundle = ConfigBundle.build([])
    with pytest.raises(KeyError, match="unknown namespace"):
        bundle.get("missing", "k")


def test_bundle_get_unknown_field_throws():
    register_schema(Schema(
        namespace="ns",
        fields=(int_field("k", 100, [Layer.SESSION_REQUEST]),),
    ))
    bundle = ConfigBundle.build([])
    with pytest.raises(KeyError, match="unknown field"):
        bundle.get("ns", "missing")


def test_bundle_all_includes_every_field():
    register_schema(Schema(
        namespace="ns_a",
        fields=(
            int_field("f1", 1, [Layer.SESSION_REQUEST]),
            int_field("f2", 2, [Layer.SESSION_REQUEST]),
        ),
    ))
    register_schema(Schema(
        namespace="ns_b",
        fields=(int_field("f3", 3, [Layer.SESSION_REQUEST]),),
    ))
    bundle = ConfigBundle.build([
        contrib(Layer.SESSION_REQUEST, "ns_a", "f2", 20),
    ])
    all_resolved = bundle.all()
    assert set(all_resolved.keys()) == {"ns_a", "ns_b"}
    assert set(all_resolved["ns_a"].keys()) == {"f1", "f2"}
    assert set(all_resolved["ns_b"].keys()) == {"f3"}
    assert all_resolved["ns_a"]["f1"].source == Layer.SCHEMA_DEFAULT
    assert all_resolved["ns_a"]["f2"].source == Layer.SESSION_REQUEST
    assert all_resolved["ns_b"]["f3"].source == Layer.SCHEMA_DEFAULT


def test_layer_name_stable_strings():
    assert layer_name(Layer.SCHEMA_DEFAULT)   == "schema_default"
    assert layer_name(Layer.BUILD_TIME)       == "build_time"
    assert layer_name(Layer.BUNDLE_DEFAULT)   == "bundle_default"
    assert layer_name(Layer.PLATFORM_PROFILE) == "platform_profile"
    assert layer_name(Layer.SESSION_REQUEST)  == "session_request"


# ---- effective_config.json round-trip --------------------------------------


def test_effective_config_json_round_trip(tmp_path: Path):
    register_schema(Schema(
        namespace="ns_a",
        fields=(
            int_field("budget", 100, [Layer.SESSION_REQUEST, Layer.BUNDLE_DEFAULT]),
            bool_field("protect", True, [Layer.SESSION_REQUEST]),
        ),
    ))
    register_schema(Schema(
        namespace="ns_b",
        fields=(int_field("width", 50, [Layer.PLATFORM_PROFILE, Layer.SESSION_REQUEST]),),
    ))
    bundle = ConfigBundle.build([
        contrib(Layer.BUNDLE_DEFAULT,   "ns_a", "budget", 200),
        contrib(Layer.SESSION_REQUEST,  "ns_a", "budget", 300),
        contrib(Layer.PLATFORM_PROFILE, "ns_b", "width", 75),
    ])

    out_path = tmp_path / "effective_config.json"
    returned = write_effective_config(bundle, out_path)
    assert returned == out_path
    assert out_path.exists()

    loaded = json.loads(out_path.read_text())
    # Namespaces are alphabetized for stable output.
    assert list(loaded.keys()) == ["ns_a", "ns_b"]
    # Fields within a namespace are alphabetized too.
    assert list(loaded["ns_a"].keys()) == ["budget", "protect"]
    assert loaded["ns_a"]["budget"] == {"value": 300, "source": "session_request"}
    assert loaded["ns_a"]["protect"] == {"value": True, "source": "schema_default"}
    assert loaded["ns_b"]["width"] == {"value": 75, "source": "platform_profile"}


def test_layer_int_values_match_cpp():
    """Guard that the Python Layer enum values stay in sync with C++.

    The C++ enum uses explicit ``std::uint8_t`` values 0..4. If those
    drift, layered merge comparisons (which depend on integer priority)
    diverge silently. This test pins the contract.
    """
    assert int(Layer.SCHEMA_DEFAULT)   == 0
    assert int(Layer.BUILD_TIME)       == 1
    assert int(Layer.BUNDLE_DEFAULT)   == 2
    assert int(Layer.PLATFORM_PROFILE) == 3
    assert int(Layer.SESSION_REQUEST)  == 4
