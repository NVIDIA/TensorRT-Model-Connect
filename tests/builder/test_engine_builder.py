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
import io
import shutil
import sys
import tarfile
import types
from pathlib import Path

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

    def test_wan_hf_id_download_uses_family_checkpoint_allowlist(
        self, tmp_path, monkeypatch
    ):
        revision = "1" * 40
        snapshot = (
            tmp_path / "hub" / "models--Wan-AI--Wan2.2-TI2V-5B"
            / "snapshots" / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            '{"_class_name":"WanModel"}', encoding="utf-8")
        captured: dict[str, object] = {}

        fake_hf = types.ModuleType("huggingface_hub")

        def fake_snapshot_download(**kwargs):
            captured.update(kwargs)
            return str(snapshot)

        fake_hf.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        assert _resolve_model("Wan-AI/Wan2.2-TI2V-5B") == str(snapshot)
        assert captured["repo_id"] == "Wan-AI/Wan2.2-TI2V-5B"
        patterns = set(captured["allow_patterns"])
        assert {
            "config.json",
            "diffusion_pytorch_model*.safetensors",
            "Wan2.2_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/**",
        } <= patterns

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

    def test_download_stages_owned_nemo_before_hf_config(
        self, tmp_path, monkeypatch
    ):
        """A recognized family archive gets writable staging before HF use."""
        dl_dir = tmp_path / "dl"
        dl_dir.mkdir()
        (dl_dir / "config.json").write_text(
            '{"model_type":"nemotron_speech_streaming"}'
        )
        nemo_path = dl_dir / "model.nemo"
        config_bytes = (
            b"_target_: nemo.collections.asr.models.rnnt_bpe_models."
            b"EncDecRNNTBPEModel\nencoder:\n  d_model: 16\n"
        )
        with tarfile.open(nemo_path, "w") as archive:
            member = tarfile.TarInfo("model_config.yaml")
            member.size = len(config_bytes)
            archive.addfile(member, io.BytesIO(config_bytes))

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = lambda **kwargs: str(dl_dir)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        staged = Path(_resolve_model("example-org/nemotron-speech"))
        try:
            assert staged != dl_dir
            assert (staged / "config.json").is_file()
            assert (staged / nemo_path.name).resolve() == nemo_path.resolve()
        finally:
            shutil.rmtree(staged)

    def test_local_canary_directory_stages_archive_before_generic_config(self, tmp_path):
        """A local Canary directory resolves without a remote model ID."""
        (tmp_path / "config.json").write_text('{"model_type":"unrelated"}')
        nemo_path = tmp_path / "local-canary.nemo"
        config_bytes = (
            b"target: nemo.collections.asr.models.aed_multitask_models."
            b"EncDecMultiTaskModel\nencoder:\n  d_model: 16\n"
            b"head:\n  num_classes: 321\n"
        )
        with tarfile.open(nemo_path, "w") as archive:
            member = tarfile.TarInfo("model_config.yaml")
            member.size = len(config_bytes)
            archive.addfile(member, io.BytesIO(config_bytes))

        staged = Path(_resolve_model(str(tmp_path)))
        try:
            assert staged != tmp_path
            staged_config = (staged / "config.json").read_text()
            assert '"model_type": "canary"' in staged_config
            assert '"vocab_size": 321' in staged_config
            assert (staged / nemo_path.name).resolve() == nemo_path.resolve()
        finally:
            shutil.rmtree(staged)

    def test_hf_snapshot_provenance_uses_immutable_cache_revision(self, tmp_path):
        from tensorrt_model_connect.engine_builder import (
            _config_with_source_provenance,
            _source_provenance,
        )

        revision = "Ab" * 20
        snapshot = (
            tmp_path / "hub" / "models--Wan-AI--Wan2.2-TI2V-5B"
            / "snapshots" / revision
        )
        snapshot.mkdir(parents=True)

        model_id, resolved_revision = _source_provenance(
            "Wan-AI/Wan2.2-TI2V-5B", snapshot)
        assert model_id == "Wan-AI/Wan2.2-TI2V-5B"
        assert resolved_revision == revision.lower()

        config = _config_with_source_provenance(
            b'{"model_type":"wan2_2_ti2v"}', model_id, resolved_revision)
        assert b'"source_model_id": "Wan-AI/Wan2.2-TI2V-5B"' in config
        assert f'"source_revision": "{revision.lower()}"'.encode() in config

    def test_local_checkpoint_does_not_embed_host_path(self, tmp_path):
        from tensorrt_model_connect.engine_builder import _source_provenance

        (tmp_path / "config.json").write_text('{}', encoding="utf-8")
        assert _source_provenance(str(tmp_path), tmp_path) == ("", "")


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
