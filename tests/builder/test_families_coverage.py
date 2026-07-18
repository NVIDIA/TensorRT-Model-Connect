# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-focused tests for families discovery helpers and protocol defaults.

Trace: ARCH-FAM-001, UD-FAM-DISCOVERY
Intent: Validate family plugin discovery helpers, diffusion lookup, and protocol default behavior
Preconditions: Family registry is populated with discoverable plugins
Postconditions: Lookup helpers return correct plugins and protocol defaults are correctly applied
"""

from __future__ import annotations

import importlib
import pkgutil
import types

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
import tensorrt_model_connect.families as families
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.base import FamilyPlugin


class _DummyFamilyPlugin:
    """Minimal plugin object used to drive deterministic registry behavior."""

    def __init__(
        self,
        *,
        name: str,
        matched_types: tuple[str, ...] = (),
        pipeline_classes: list[str] | None = None,
    ):
        self.name = name
        self._matched_types = set(matched_types)
        self.pipeline_classes = pipeline_classes

    def matches(self, model_type: str) -> bool:
        return model_type in self._matched_types


def test_find_diffusion_plugin_returns_first_pipeline_class_match(monkeypatch):
    """Intent: cover positive lookup path in find_diffusion_plugin.
    Preconditions: registry contains multiple plugins that declare pipeline_classes.
    Postconditions: the first plugin listing the requested class is returned.
    """
    first = _DummyFamilyPlugin(name="first", pipeline_classes=["SyntheticPipeline"])
    second = _DummyFamilyPlugin(name="second", pipeline_classes=["SyntheticPipeline"])
    monkeypatch.setattr(families, "_ALL_PLUGINS", [first, second])

    resolved = families.find_diffusion_plugin("SyntheticPipeline")

    assert resolved is first


def test_find_diffusion_plugin_returns_none_when_no_plugin_declares_class(monkeypatch):
    """Intent: cover negative lookup path in find_diffusion_plugin.
    Preconditions: plugins either omit pipeline_classes or provide non-matching classes.
    Postconditions: helper returns None for an unknown pipeline class.
    """
    plugins = [
        _DummyFamilyPlugin(name="no_attr", pipeline_classes=None),
        _DummyFamilyPlugin(name="empty", pipeline_classes=[]),
        _DummyFamilyPlugin(name="other", pipeline_classes=["OtherSyntheticPipeline"]),
    ]
    monkeypatch.setattr(families, "_ALL_PLUGINS", plugins)

    assert families.find_diffusion_plugin("SyntheticPipeline") is None


def test_private_diffusion_family_resolution_uses_metadata_without_importing_plugin(monkeypatch):
    metadata = types.SimpleNamespace(
        id="synthetic_family",
        diffusion_pipeline_classes=frozenset({"SyntheticPipeline"}),
    )
    monkeypatch.setattr(families, "_load_family_metadata", lambda: [metadata])
    monkeypatch.setattr(
        families,
        "_load_plugin_from_module",
        lambda _module: (_ for _ in ()).throw(
            AssertionError("metadata-only resolution imported a native plugin")
        ),
    )

    assert families._resolve_diffusion_family_id("SyntheticPipeline") == "synthetic_family"
    assert families._resolve_diffusion_family_id("UnknownPipeline") is None


def test_family_module_discovery_skips_private_base_import_errors_and_missing_plugin_attr(monkeypatch):
    """Intent: exercise module auto-discovery branches in families.__init__.
    Preconditions: iter_modules yields private/base/error/no-plugin/valid entries under monkeypatched import hooks.
    Postconditions: discovery appends only the valid module's plugin and ignores all skipped/error cases.
    """
    good_plugin = _DummyFamilyPlugin(name="good")
    good_mod = types.ModuleType("tensorrt_model_connect.families.good")
    good_mod.plugin = good_plugin
    no_plugin_mod = types.ModuleType("tensorrt_model_connect.families.no_plugin")

    iter_rows = [
        (None, "_private", False),
        (None, "base", False),
        (None, "broken_dep", False),
        (None, "no_plugin", False),
        (None, "good", False),
    ]

    def fake_iter_modules(_paths):
        return iter(iter_rows)

    def fake_import_module(name: str):
        if name.endswith(".broken_dep"):
            raise ImportError("synthetic missing dependency")
        if name.endswith(".no_plugin"):
            return no_plugin_mod
        if name.endswith(".good"):
            return good_mod
        raise AssertionError(f"Unexpected import: {name}")

    with monkeypatch.context() as m:
        m.setattr(pkgutil, "iter_modules", fake_iter_modules)
        m.setattr(importlib, "import_module", fake_import_module)
        reloaded = importlib.reload(families)
        assert reloaded._ALL_PLUGINS == [good_plugin]

    # Restore canonical discovery state for any subsequent tests.
    importlib.reload(families)


def test_find_plugin_returns_none_when_registry_has_no_match(monkeypatch):
    """Intent: explicitly hit find_plugin no-match branch with synthetic plugins.
    Preconditions: registry contains plugins whose matches() all return False for the query.
    Postconditions: find_plugin returns None.
    """
    plugins = [
        _DummyFamilyPlugin(name="a", matched_types=("foo",)),
        _DummyFamilyPlugin(name="b", matched_types=("bar",)),
    ]
    monkeypatch.setattr(families, "_ALL_PLUGINS", plugins)

    assert families.find_plugin("baz") is None


def test_find_plugin_imports_only_candidate_family(monkeypatch):
    """Intent: normal lookup imports the target family, not every family package."""
    reloaded = importlib.reload(families)
    plugin = _DummyFamilyPlugin(
        name="example_family",
        matched_types=("example_model",),
    )
    loaded_families = []

    def fake_load_plugin_from_module(module_name: str):
        loaded_families.append(module_name)
        return plugin if module_name == "example_family" else None

    monkeypatch.setattr(reloaded, "load_plugin_by_id", lambda _model_type: None)
    monkeypatch.setattr(
        reloaded,
        "_candidate_module_names",
        lambda _model_type: ["example_family"],
    )
    monkeypatch.setattr(
        reloaded,
        "_load_plugin_from_module",
        fake_load_plugin_from_module,
    )
    monkeypatch.setattr(reloaded, "_ensure_discovered", lambda: None)
    reloaded._PLUGIN_CACHE.clear()

    resolved = reloaded.find_plugin("example_model")

    assert resolved is not None
    assert resolved.name == "example_family"
    assert loaded_families == ["example_family"]


def test_base_protocol_required_method_bodies_are_executable_defaults():
    """Intent: cover default bodies of required protocol methods.
    Preconditions: protocol methods are called unbound with placeholder args.
    Postconditions: each method executes and returns None (ellipsis expression body).
    """
    cfg = ModelConfig()
    assert FamilyPlugin.matches(object(), "model_type") is None
    assert FamilyPlugin.load_weights(object(), "/tmp/model", cfg) is None
    assert FamilyPlugin.build_engine(object(), cfg, {}, 256, verbose=False) is None


def test_base_protocol_optional_methods_return_none_by_default():
    """Intent: cover optional protocol default-return methods.
    Preconditions: optional methods are invoked unbound with placeholder args.
    Postconditions: each optional method returns None.
    """
    cfg = ModelConfig()
    assert FamilyPlugin.build_vision_engine(
        object(), "/tmp/model", cfg, {}, verbose=False
    ) is None
    assert FamilyPlugin.get_vl_config(object(), cfg) is None
    assert FamilyPlugin.build_components(
        object(), "/tmp/model", cfg, {}, verbose=False
    ) is None
    assert FamilyPlugin.get_diffusion_config(object(), cfg) is None
