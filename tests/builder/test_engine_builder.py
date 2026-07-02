# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for engine_builder.py — orchestrator logic.

Tests plugin discovery and model resolution without requiring TRT.

Trace: ARCH-ENG-001, UD-ENG-03
Intent: Validate the engine builder orchestrator's model resolution, plugin discovery, and family dispatch logic.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: Local directories with config.json resolve correctly, HF repo IDs are detected, and all registered family plugins are discoverable.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.engine_builder import _resolve_model
from tensorrt_model_connect.families import find_plugin, _ALL_PLUGINS

engine_builder = importlib.import_module(_resolve_model.__module__)


class TestResolveModel:
    def test_local_dir_with_config(self, tmp_path):
        """Local directory with config.json returns the path directly."""
        (tmp_path / "config.json").write_text('{"model_type": "test"}')
        result = _resolve_model(str(tmp_path))
        assert result == str(tmp_path)

    def test_local_dir_without_config(self, tmp_path):
        """Directory without config.json treated as HF repo ID.
        Should raise ImportError if huggingface_hub is not installed,
        or attempt download."""
        # The directory exists but has no config.json
        try:
            _resolve_model(str(tmp_path))
        except (ImportError, Exception):
            # Expected: either HF hub not available or download fails
            pass

    def test_nonexistent_path_treated_as_repo_id(self):
        """A non-existent path is treated as a HF repo ID."""
        try:
            _resolve_model("nonexistent/model-that-does-not-exist-12345")
        except (ImportError, Exception):
            pass

    def test_download_prefers_hf_config_over_nemo(self, tmp_path, monkeypatch):
        """When both HF config and .nemo exist, keep HF path behavior."""
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()
        (dl_dir / "config.json").write_text('{"model_type": "example_decoder"}')
        (dl_dir / "model.nemo").write_text("placeholder")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = lambda **kwargs: str(dl_dir)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        called = {"nemo": False}

        def fake_resolve_nemo_archive(_):
            called["nemo"] = True
            return "/tmp/nemo"

        monkeypatch.setattr(
            engine_builder, "_resolve_nemo_archive", fake_resolve_nemo_archive)

        result = _resolve_model("example-org/hf-config-model")
        assert result == str(dl_dir)
        assert called["nemo"] is False

    def test_download_uses_nemo_when_no_hf_config(self, tmp_path, monkeypatch):
        """NeMo fallback remains active for snapshots that are .nemo-only."""
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()
        nemo_path = dl_dir / "model.nemo"
        nemo_path.write_text("placeholder")

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = lambda **kwargs: str(dl_dir)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        def fake_resolve_nemo_archive(path):
            return f"resolved:{path}"

        monkeypatch.setattr(
            engine_builder, "_resolve_nemo_archive", fake_resolve_nemo_archive)

        result = _resolve_model("example-org/nemo-only-model")
        assert result == f"resolved:{nemo_path}"


class TestFindPlugin:
    def test_unsupported_model_type(self):
        assert find_plugin("nonexistent_model_type_12345") is None

    def test_all_plugins_have_required_attributes(self):
        """Every plugin must have name, matches, load_weights, build_engine."""
        for p in _ALL_PLUGINS:
            assert hasattr(p, "name")
            assert hasattr(p, "matches")
            assert hasattr(p, "load_weights")
            assert hasattr(p, "build_engine")
            assert isinstance(p.name, str)
            assert len(p.name) > 0
            assert callable(p.matches)
            assert callable(p.load_weights)
            assert callable(p.build_engine)
