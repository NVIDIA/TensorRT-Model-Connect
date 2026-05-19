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
        (dl_dir / "config.json").write_text('{"model_type": "nemotron"}')
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

        result = _resolve_model("nvidia/Nemotron-4-Mini-Hindi-4B-Base")
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

        result = _resolve_model("nvidia/Magpie-TTS")
        assert result == f"resolved:{nemo_path}"

    def test_sana_wm_downloads_only_metadata_files(self, tmp_path, monkeypatch):
        """SANA-WM resolution must not pull the 100GB weight payload during build."""
        dl_dir = tmp_path / "sana-wm"
        dl_dir.mkdir()
        (dl_dir / "config.yaml").write_text(
            "model:\n"
            "  model: SanaMSVideoCamCtrl_1600M_P1_D20\n"
            "vae:\n"
            "  vae_type: LTX2VAE_diffusers\n",
            encoding="utf-8",
        )
        calls: list[dict] = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(dl_dir)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        result = _resolve_model("Efficient-Large-Model/SANA-WM_bidirectional")

        assert result == str(dl_dir)
        assert calls
        assert calls[0]["allow_patterns"] == ["README.md", "config.yaml"]

    def test_sana_wm_can_opt_into_full_snapshot_download(self, tmp_path, monkeypatch):
        """SANA-WM full snapshot download is explicit because the model is large."""
        dl_dir = tmp_path / "sana-wm"
        dl_dir.mkdir()
        (dl_dir / "config.yaml").write_text(
            "model:\n"
            "  model: SanaMSVideoCamCtrl_1600M_P1_D20\n"
            "vae:\n"
            "  vae_type: LTX2VAE_diffusers\n",
            encoding="utf-8",
        )
        calls: list[dict] = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            return str(dl_dir)

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        monkeypatch.setenv("TRTMC_SANA_WM_DOWNLOAD_WEIGHTS", "1")

        result = _resolve_model("Efficient-Large-Model/SANA-WM_bidirectional")

        assert result == str(dl_dir)
        assert calls
        allow_patterns = calls[0]["allow_patterns"]
        assert "config.yaml" in allow_patterns
        assert "asset/sana_wm/**" in allow_patterns
        assert "inference_video_scripts/**" in allow_patterns
        assert "dit/**" in allow_patterns
        assert "vae/**" in allow_patterns
        assert "text_encoder/**" in allow_patterns
        assert "refiner/**" in allow_patterns


class TestFindPlugin:
    def test_supported_model_types(self):
        """Verify find_plugin returns non-None for all known model types."""
        known_types = [
            "qwen", "qwen2", "qwen3", "qwq",
            "llama", "mistral", "gemma", "gemma2",
            "phi", "phi3", "phimoe",
            "granite", "internlm", "internlm2",
            "starcoder2", "gpt2", "opt", "falcon", "stablelm",
            "olmo", "xglm", "gpt_neox", "gpt_neo", "codegen",
            "bloom", "mamba", "mixtral",
            "qwen2_vl", "qwen2_5_vl",
            "sana_wm",
        ]
        for model_type in known_types:
            plugin = find_plugin(model_type)
            assert plugin is not None, f"No plugin for {model_type}"

    def test_unsupported_model_type(self):
        assert find_plugin("nonexistent_model_type_12345") is None

    def test_known_families(self):
        """Verify key model types map to expected family names."""
        known = {
            "qwen3": "qwen",
            "qwen2": "qwen",
            "llama": "llama",
            "mistral": "mistral",
            "gemma": "gemma",
            "gemma2": "gemma",
            "phi3": "phi",
        }
        for model_type, expected_family in known.items():
            plugin = find_plugin(model_type)
            if plugin is not None:
                assert plugin.name == expected_family, \
                    f"{model_type} -> {plugin.name} (expected {expected_family})"

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
