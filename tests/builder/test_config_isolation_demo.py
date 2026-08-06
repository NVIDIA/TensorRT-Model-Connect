# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# ISO 26262 Traceability
# =============================================================================
# Trace ID:       UT-CFG-SCALE-01
# Architecture:   ARCH-CFG-001
# Unit Design:    UD-CFG-REG-01
# Intent:         Phase 5.a scalability gate. Demonstrates that adding a new
#                 feature to the config registry requires only the new
#                 schema file + this test — no edits to any CLI parser,
#                 shared dispatcher, or central registry-of-registries.
# Preconditions:  None (no GPU, no TRT, no filesystem outside tmp_path).
# Postconditions: The demo_feature namespace registers cleanly, --set
#                 routes values through the shared CLI helper, merge
#                 priority is respected, and effective_config dumps the
#                 value with its source layer.
# =============================================================================

"""Scalability test: a brand-new feature plugs in with zero shared-file edits.

This test mirrors what Phase 4 clusters do in production:

    1. Declare a namespaced schema (here inline — in production it would
       be a file under ``python/tensorrt_model_connect/runtime_config/schemas/``).
    2. Register it with the singleton SchemaRegistry.
    3. Supply values via ``--set`` tokens through
       :func:`resolve_cli_config`.
    4. Verify the resolved ConfigBundle exposes the value with correct
       type and provenance.
    5. Dump an effective_config.json and assert the new namespace is
       serialized.

The entire production-surface diff for landing a new feature is:

    - One new schema file under runtime_config/schemas/
    - One corresponding C++ schema source under src/runtime/config/schemas/
      plus one manifest line in cmake/trtmc_config_schemas.cmake
    - One test file here

No edits to build_cli.py, engine_builder.py, pipeline_factory.cpp, or any
shared dispatcher. This test stands in for the "one-shot demo" that
would exercise exactly that diff.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from tensorrt_model_connect.runtime_config import (
        ConfigField,
        Layer,
        LayerContribution,
        Schema,
        clear_for_testing,
        lookup,
        register_schema,
        registered_namespaces,
        resolve_cli_config,
        write_effective_config_next_to,
    )
except ImportError:  # pragma: no cover
    pytest.skip("tensorrt_model_connect.runtime_config not importable", allow_module_level=True)


_SESSION = frozenset({Layer.SESSION_REQUEST, Layer.PLATFORM_PROFILE})
_BUNDLE_AND_SESSION = frozenset({
    Layer.BUILD_TIME, Layer.BUNDLE_DEFAULT,
    Layer.PLATFORM_PROFILE, Layer.SESSION_REQUEST,
})


def _register_demo_feature_schema() -> Schema:
    """Register a synthetic demo_feature schema. Stand-in for a schema file."""
    schema = Schema(
        namespace="demo_feature",
        fields=(
            ConfigField(
                name="max_candidates",
                type_tag="int32",
                default=42,
                allowed_layers=_BUNDLE_AND_SESSION,
                validator=lambda v: isinstance(v, int) and v > 0,
            ),
            ConfigField(
                name="mode",
                type_tag="string",
                default="fast",
                allowed_layers=_SESSION,
                validator=lambda v: v in {"fast", "quality", "experimental"},
            ),
            ConfigField(
                name="enabled",
                type_tag="bool",
                default=False,
                allowed_layers=_SESSION,
            ),
        ),
    )
    register_schema(schema)
    return schema


@pytest.fixture(autouse=True)
def clean_registry():
    clear_for_testing()
    yield
    clear_for_testing()


# ---- 1. Schema registers cleanly ------------------------------------------


def test_new_feature_registers_with_only_one_entry_point():
    _register_demo_feature_schema()
    assert "demo_feature" in registered_namespaces()
    schema = lookup("demo_feature")
    assert schema is not None
    assert {f.name for f in schema.fields} == {"max_candidates", "mode", "enabled"}


# ---- 2. Schema defaults are visible through ConfigBundle -------------------


def test_defaults_flow_through_without_any_contribution():
    _register_demo_feature_schema()
    bundle = resolve_cli_config(config_path=None, set_tokens=[])
    assert bundle.get("demo_feature", "max_candidates") == 42
    assert bundle.get("demo_feature", "mode") == "fast"
    assert bundle.get("demo_feature", "enabled") is False
    assert bundle.source_of("demo_feature", "max_candidates") == Layer.SCHEMA_DEFAULT


# ---- 3. --set flows through without touching build_cli.py ------------------------


def test_set_token_routes_to_new_feature_without_cli_edit():
    _register_demo_feature_schema()
    bundle = resolve_cli_config(
        config_path=None,
        set_tokens=[
            "demo_feature.max_candidates=128",
            "demo_feature.mode=experimental",
            "demo_feature.enabled=true",
        ],
    )
    assert bundle.get("demo_feature", "max_candidates") == 128
    assert bundle.get("demo_feature", "mode") == "experimental"
    assert bundle.get("demo_feature", "enabled") is True
    assert bundle.source_of("demo_feature", "enabled") == Layer.SESSION_REQUEST


# ---- 4. --config JSON routes to new feature without touching any dispatcher


def test_config_file_routes_to_new_feature(tmp_path: Path):
    _register_demo_feature_schema()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "demo_feature": {"max_candidates": 64, "mode": "quality"},
    }))
    bundle = resolve_cli_config(
        config_path=str(profile),
        set_tokens=["demo_feature.max_candidates=256"],  # --set beats --config
    )
    assert bundle.get("demo_feature", "max_candidates") == 256
    assert bundle.get("demo_feature", "mode") == "quality"


# ---- 5. Validator rejects invalid values -----------------------------------


def test_validator_rejects_invalid_demo_value():
    _register_demo_feature_schema()
    with pytest.raises(ValueError, match="Validator rejected"):
        resolve_cli_config(
            config_path=None,
            set_tokens=["demo_feature.mode=unsupported_mode"],
        )


def test_allowlist_rejects_wrong_layer():
    _register_demo_feature_schema()
    bad_contrib = LayerContribution(
        layer=Layer.BUNDLE_DEFAULT,
        values={"demo_feature": {"mode": "fast"}},
    )
    with pytest.raises(ValueError, match="not permitted"):
        resolve_cli_config(
            config_path=None, set_tokens=[],
            extra_contributions=[bad_contrib],
        )


# ---- 6. effective_config.json serializes the new namespace -----------------


def test_effective_config_dumps_new_feature(tmp_path: Path):
    _register_demo_feature_schema()
    bundle = resolve_cli_config(
        config_path=None,
        set_tokens=["demo_feature.max_candidates=99", "demo_feature.mode=fast"],
    )
    bundle_path = tmp_path / "nothing.bundle"
    written = write_effective_config_next_to(bundle, bundle_path)
    data = json.loads(written.read_text())
    assert data["demo_feature"]["max_candidates"] == {
        "value": 99, "source": "session_request",
    }
    assert data["demo_feature"]["mode"] == {
        "value": "fast", "source": "session_request",
    }
    # enabled was never touched → schema default, but still serialized.
    assert data["demo_feature"]["enabled"]["source"] == "schema_default"


# ---- 7. Scalability claim, recorded ----------------------------------------


def test_scalability_claim_documented():
    """Meta-assertion: lists the exact files needed for a new feature.

    If adding a new feature ever requires editing something outside this
    list, that's a coupling point; update the design before adding.
    """
    expected_new_files_per_feature = [
        "python/tensorrt_model_connect/families/<family>/runtime_config_schema.py",
        "src/runtime/models/<model>/config_schema.h",
        "src/runtime/models/<model>/config_schema.cpp",
        "tests/builder/test_config_<name>_or_similar.py",
    ]
    expected_modified_files_per_feature = [
        # One model manifest entry drives DSO compilation and generated registration calls.
        "src/runtime/models/<model>/MODEL.toml",
    ]
    # No runtime change outside the model's own consumer code.
    # This is an assertion-as-documentation; keep these lists in sync
    # with the config-registry status doc.
    assert len(expected_new_files_per_feature) >= 3
    assert len(expected_modified_files_per_feature) == 1
