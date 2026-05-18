"""Extended tests for engine_builder.py — helper functions and orchestration.

Uses mocks for TRT, GPU, and filesystem access. No GPU or TRT needed.

Trace: ARCH-ENG-001, UD-ENG-04
Intent: Validate engine builder helper functions including TRT version detection, GPU name retrieval, tokenizer JSON provisioning, and build_bundle orchestration.
Preconditions: tensorrt_model_connect is importable; uses mocks for TRT and GPU introspection.
Postconditions: TRT version is correctly retrieved or falls back to 'unknown', GPU name is detected, tokenizer JSON is ensured, and build_bundle dispatches correctly.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    from tensorrt_model_connect.engine_builder import (
        _get_trt_version,
        _get_gpu_name,
        _ensure_tokenizer_json,
        build_bundle,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


# ---------------------------------------------------------------------------
# _get_trt_version
# ---------------------------------------------------------------------------


class TestGetTrtVersion:
    def test_returns_version_string(self):
        """When tensorrt is importable, returns trt.__version__."""
        mock_trt = MagicMock()
        mock_trt.__version__ = "10.3.0"
        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            assert _get_trt_version() == "10.3.0"

    def test_missing_trt_returns_unknown(self):
        """When tensorrt import fails, returns 'unknown'."""
        with patch.dict("sys.modules", {"tensorrt": None}):
            assert _get_trt_version() == "unknown"

    def test_version_attribute_error_returns_unknown(self):
        """When tensorrt has no __version__, returns 'unknown'."""
        mock_trt = MagicMock(spec=[])  # empty spec — no __version__
        with patch.dict("sys.modules", {"tensorrt": mock_trt}):
            assert _get_trt_version() == "unknown"


# ---------------------------------------------------------------------------
# _get_gpu_name
# ---------------------------------------------------------------------------


class TestGetGpuName:
    def test_parses_nvidia_smi_output(self):
        """Parses GPU name from nvidia-smi CSV output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090\n"

        with patch("subprocess.run",
                    return_value=mock_result) as mock_run:
            name = _get_gpu_name()
            assert name == "NVIDIA GeForce RTX 4090"
            mock_run.assert_called_once_with(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_multi_gpu_returns_first(self):
        """When multiple GPUs are present, returns the first one."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA H100\nNVIDIA H100\nNVIDIA H100\n"

        with patch("subprocess.run",
                    return_value=mock_result):
            assert _get_gpu_name() == "NVIDIA H100"

    def test_nvidia_smi_fails_returns_empty(self):
        """When nvidia-smi returns non-zero, returns empty string."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run",
                    return_value=mock_result):
            assert _get_gpu_name() == ""

    def test_nvidia_smi_not_found_returns_empty(self):
        """When nvidia-smi is not found, returns empty string."""
        with patch("subprocess.run",
                    side_effect=FileNotFoundError("nvidia-smi not found")):
            assert _get_gpu_name() == ""

    def test_nvidia_smi_timeout_returns_empty(self):
        """When nvidia-smi times out, returns empty string."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ):
            assert _get_gpu_name() == ""


# ---------------------------------------------------------------------------
# _ensure_tokenizer_json
# ---------------------------------------------------------------------------


class TestEnsureTokenizerJson:
    def test_tokenizer_json_already_exists(self, tmp_path):
        """When tokenizer.json exists, function is a no-op (early return)."""
        (tmp_path / "tokenizer.json").write_text('{"version": "1.0"}')
        content_before = (tmp_path / "tokenizer.json").read_text()

        # Function should return immediately without trying to import transformers.
        # If it did try to import, it would hit our poisoned module and raise.
        with patch.dict("sys.modules", {"transformers": None}):
            _ensure_tokenizer_json(tmp_path)

        # File unchanged
        assert (tmp_path / "tokenizer.json").read_text() == content_before

    def test_generates_from_slow_tokenizer(self, tmp_path):
        """When tokenizer.json is missing, generates it via AutoTokenizer."""
        (tmp_path / "tokenizer_config.json").write_text('{"model_type": "test"}')
        assert not (tmp_path / "tokenizer.json").exists()

        mock_tok = MagicMock()

        # Simulate save_pretrained creating tokenizer.json
        def _save_pretrained(path):
            (Path(path) / "tokenizer.json").write_text('{"generated": true}')

        mock_tok.save_pretrained.side_effect = _save_pretrained

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            _ensure_tokenizer_json(tmp_path)

        assert (tmp_path / "tokenizer.json").exists()
        assert json.loads((tmp_path / "tokenizer.json").read_text()) == {"generated": True}

    def test_transformers_import_fails_gracefully(self, tmp_path):
        """When transformers is not installed, no error is raised."""
        assert not (tmp_path / "tokenizer.json").exists()

        # Setting the module to None in sys.modules causes `import transformers`
        # to raise ImportError, which _ensure_tokenizer_json catches.
        with patch.dict("sys.modules", {"transformers": None}):
            _ensure_tokenizer_json(tmp_path)

        # No tokenizer.json created, but no exception raised
        assert not (tmp_path / "tokenizer.json").exists()

    def test_save_pretrained_doesnt_create_file(self, tmp_path):
        """When save_pretrained runs but doesn't create tokenizer.json, no error."""
        assert not (tmp_path / "tokenizer.json").exists()

        mock_tok = MagicMock()
        mock_tok.save_pretrained.return_value = None  # does nothing

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            _ensure_tokenizer_json(tmp_path)

        # No tokenizer.json created, but no exception raised
        assert not (tmp_path / "tokenizer.json").exists()

    def test_auto_tokenizer_from_pretrained_raises(self, tmp_path):
        """When AutoTokenizer.from_pretrained raises, error is caught."""
        assert not (tmp_path / "tokenizer.json").exists()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = \
            OSError("Model not found")

        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            # Should not raise
            _ensure_tokenizer_json(tmp_path)

        assert not (tmp_path / "tokenizer.json").exists()


# ---------------------------------------------------------------------------
# build_bundle orchestration
# ---------------------------------------------------------------------------


class TestBuildBundleOrchestration:
    def _make_model_dir(self, tmp_path, model_type="qwen3"):
        """Create a minimal model directory with config.json."""
        config = {
            "model_type": model_type,
            "architectures": [f"{model_type.capitalize()}ForCausalLM"],
            "vocab_size": 100,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        return tmp_path

    def test_unknown_model_type_raises(self, tmp_path):
        """Unknown model_type raises ValueError with helpful message."""
        model_dir = self._make_model_dir(
            tmp_path, model_type="nonexistent_model_xyz")

        with pytest.raises(ValueError, match="No family plugin"):
            build_bundle(str(model_dir), str(tmp_path / "out.trtfb"))

    def test_unknown_model_type_lists_supported(self, tmp_path):
        """Error message for unknown model_type lists supported families."""
        model_dir = self._make_model_dir(
            tmp_path, model_type="nonexistent_model_xyz")

        with pytest.raises(ValueError, match="Supported:"):
            build_bundle(str(model_dir), str(tmp_path / "out.trtfb"))

    def test_missing_config_json_raises(self, tmp_path):
        """Missing config.json raises FileNotFoundError."""
        # Empty directory — no config.json
        with pytest.raises(FileNotFoundError):
            build_bundle(str(tmp_path), str(tmp_path / "out.trtfb"))

    def test_orchestration_flow(self, tmp_path):
        """Verify the correct flow: config -> plugin -> weights -> engine -> bundle."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        # Create a mock plugin
        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = ""
        mock_plugin.load_weights.return_value = {
            "embedding": b"fake",
            "_attention_size": 64,
        }
        mock_plugin.build_engine.return_value = b"FAKE_ENGINE_PLAN"

        # Remove optional attributes so getattr() returns defaults
        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.3.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value="NVIDIA H100"):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(str(model_dir), output_path)

                            # Verify plugin was called with correct arguments
                            mock_plugin.load_weights.assert_called_once()
                            mock_plugin.build_engine.assert_called_once()

                            # Verify write_bundle was called
                            mock_write.assert_called_once()
                            call_args = mock_write.call_args
                            assert call_args[0][0] == output_path

                            # Verify BundleInfo fields
                            info = call_args[0][1]
                            assert info.model_type == "qwen3"
                            assert info.family == "qwen"
                            assert info.trt_version == "10.3.0"
                            assert info.trt_abi == "10.3"
                            assert info.gpu_name == "NVIDIA H100"
                            assert info.vocab_size == 100
                            assert info.hidden_size == 64
                            assert info.num_layers == 2

    def test_engine_plan_in_sections(self, tmp_path):
        """Verify engine_plan is the first section."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = ""
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"FAKE_ENGINE_PLAN_DATA"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(str(model_dir), output_path)

                            sections = mock_write.call_args[0][2]
                            engine_section = sections[0]
                            assert engine_section.name == "engine_plan"
                            assert engine_section.data == b"FAKE_ENGINE_PLAN_DATA"

    def test_max_cache_length_forwarded(self, tmp_path):
        """max_cache_length is forwarded to plugin.build_engine."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = ""
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle"):
                            build_bundle(
                                str(model_dir), output_path,
                                max_cache_length=512)

                            call_args = mock_plugin.build_engine.call_args
                            assert call_args[0][2] == 512  # max_cache_length positional

    def test_config_json_embedded_in_sections(self, tmp_path):
        """config.json from model dir is embedded in bundle sections."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = ""
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(str(model_dir), output_path)

                            sections = mock_write.call_args[0][2]
                            section_names = [s.name for s in sections]
                            assert "config.json" in section_names

    def test_yaml_only_elf_synthesizes_config_json_section(self, tmp_path):
        """GitHub ELF YAML-only directories still get runtime config.json."""
        (tmp_path / "train_owt_ELF-B.yml").write_text(
            "\n".join([
                "model: ELF-B",
                "max_length: 1024",
                "encoder_model_name: t5-small",
                "num_time_tokens: 4",
                "num_self_cond_cfg_tokens: 4",
                "num_model_mode_tokens: 4",
                "denoiser_p_mean: -1.5",
                "denoiser_p_std: 0.8",
                "denoiser_noise_scale: 2.0",
                "self_cond_prob: 0.5",
            ]),
            encoding="utf-8",
        )
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "elf"
        mock_plugin.runtime_strategy = "elf_flow"
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"
        mock_plugin.get_bundle_config_overrides.return_value = {
            "runtime_strategy": "elf_flow",
            "model_type": "elf",
            "elf_max_length": 1024,
            "elf_max_input_length": 0,
            "elf_text_encoder_dim": 512,
            "elf_input_dim": 1024,
            "elf_denoiser_noise_scale": 2.0,
        }

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                   return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                       return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                           return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(str(tmp_path), output_path)

        sections = mock_write.call_args[0][2]
        section_map = {section.name: section.data for section in sections}
        cfg = json.loads(section_map["config.json"].decode("utf-8"))
        assert cfg["runtime_strategy"] == "elf_flow"
        assert cfg["model_type"] == "elf"
        assert cfg["model"] == "ELF-B"
        assert cfg["elf_max_length"] == 1024
        assert cfg["elf_max_input_length"] == 0
        assert cfg["elf_text_encoder_dim"] == 512
        assert cfg["elf_input_dim"] == 1024
        assert cfg["elf_denoiser_noise_scale"] == 2.0

    def test_runtime_strategy_injected(self, tmp_path):
        """runtime_strategy from plugin is injected into config.json section."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = "decoder_moe"
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(str(model_dir), output_path)

                            sections = mock_write.call_args[0][2]
                            config_section = [
                                s for s in sections if s.name == "config.json"
                            ][0]
                            cfg = json.loads(config_section.data.decode("utf-8"))
                            assert cfg["runtime_strategy"] == "decoder_moe"

                            # Also verify BundleInfo.runtime_strategy
                            info = mock_write.call_args[0][1]
                            assert info.runtime_strategy == "decoder_moe"

    def test_bundle_info_max_cache_length(self, tmp_path):
        """BundleInfo records the max_cache_length."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = ""
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                            build_bundle(
                                str(model_dir), output_path,
                                max_cache_length=1024)

                            info = mock_write.call_args[0][1]
                            assert info.max_cache_length == 1024

    def test_triattention_embeds_stats_and_config(self, tmp_path):
        """TriAttention build options add config and stats sections."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = "decoder_kv_cache"
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        tri_stats = b'{"version": 1, "sampled_heads": [[0, 0]], "stats": {}}'

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch(
                            "tensorrt_model_connect.engine_builder.export_triattention_stats_section",
                            return_value=tri_stats,
                        ) as mock_export:
                            with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                                build_bundle(
                                    str(model_dir),
                                    output_path,
                                    max_cache_length=256,
                                    triattention_stats_path="triattention.pt",
                                    triattention_kv_budget=96,
                                    triattention_recent_window=24,
                                    triattention_score_aggregation="max",
                                    triattention_count_prompt_tokens=False,
                                    triattention_protect_prefill=True,
                                    triattention_disable_mlr=True,
                                )

        mock_export.assert_called_once()
        sections = mock_write.call_args[0][2]
        section_map = {section.name: section.data for section in sections}
        assert section_map["triattention_stats.json"] == tri_stats

        cfg = json.loads(section_map["config.json"].decode("utf-8"))
        tri_cfg = cfg["triattention"]
        assert tri_cfg["enabled"] is True
        assert tri_cfg["kv_budget"] == 96
        assert tri_cfg["divide_length"] == 128
        assert tri_cfg["recent_window"] == 24
        assert tri_cfg["score_aggregation"] == "max"
        assert tri_cfg["count_prompt_tokens"] is False
        assert tri_cfg["protect_prefill"] is True
        assert tri_cfg["disable_mlr"] is True
        assert tri_cfg["disable_trig"] is False
        assert cfg["dynamic_kv_cache"] is True
        assert cfg["dynamic_kv_profile_rows"] == [96, 192, 256]

    def test_large_triattention_budget_adds_lower_warmup_profile(self, tmp_path):
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        mock_plugin = MagicMock()
        mock_plugin.name = "qwen"
        mock_plugin.runtime_strategy = "decoder_kv_cache"
        mock_plugin.load_weights.return_value = {}
        mock_plugin.build_engine.return_value = b"PLAN"

        del mock_plugin.build_vision_engine
        del mock_plugin.build_extra_engines
        del mock_plugin.embed_input
        del mock_plugin.get_vl_config
        del mock_plugin.get_segmentation_config
        del mock_plugin.get_audio_config
        del mock_plugin.get_bundle_config_overrides

        tri_stats = b'{"version": 1, "sampled_heads": [[0, 0]], "stats": {}}'

        with patch("tensorrt_model_connect.engine_builder.find_plugin",
                    return_value=mock_plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version",
                        return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name",
                            return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch(
                            "tensorrt_model_connect.engine_builder.export_triattention_stats_section",
                            return_value=tri_stats,
                        ):
                            with patch("tensorrt_model_connect.engine_builder.write_bundle") as mock_write:
                                build_bundle(
                                    str(model_dir),
                                    output_path,
                                    max_cache_length=12288,
                                    triattention_stats_path="triattention.pt",
                                    triattention_kv_budget=6144,
                                    triattention_divide_length=1024,
                                    triattention_recent_window=128,
                                )

        sections = mock_write.call_args[0][2]
        section_map = {section.name: section.data for section in sections}
        cfg = json.loads(section_map["config.json"].decode("utf-8"))
        assert cfg["dynamic_kv_profile_rows"] == [3072, 6144, 12288]

    def test_load_weights_precision_forwarded_when_supported(self, tmp_path):
        """build_bundle forwards precision to load_weights when supported."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")
        seen = {}

        class _Plugin:
            name = "qwen"
            runtime_strategy = ""

            def load_weights(self, model_dir, config, *, precision="fp32"):
                seen["precision"] = precision
                return {}

            def build_engine(self, config, weights, max_cache_length, *, precision="fp32", verbose=False):
                return b"PLAN"

        plugin = _Plugin()

        with patch("tensorrt_model_connect.engine_builder.find_plugin", return_value=plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version", return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name", return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle"):
                            build_bundle(
                                str(model_dir),
                                output_path,
                                precision="fp16",
                            )

        assert seen["precision"] == "fp16"

    def test_load_weights_precision_not_forwarded_when_unsupported(self, tmp_path):
        """build_bundle remains compatible with plugins that do not accept precision."""
        model_dir = self._make_model_dir(tmp_path, model_type="qwen3")
        output_path = str(tmp_path / "output.trtfb")

        class _Plugin:
            name = "qwen"
            runtime_strategy = ""

            def load_weights(self, model_dir, config):
                return {}

            def build_engine(self, config, weights, max_cache_length, *, precision="fp32", verbose=False):
                return b"PLAN"

        plugin = _Plugin()

        with patch("tensorrt_model_connect.engine_builder.find_plugin", return_value=plugin):
            with patch("tensorrt_model_connect.engine_builder._get_trt_version", return_value="10.0"):
                with patch("tensorrt_model_connect.engine_builder._get_gpu_name", return_value=""):
                    with patch("tensorrt_model_connect.engine_builder._ensure_tokenizer_json"):
                        with patch("tensorrt_model_connect.engine_builder.write_bundle"):
                            build_bundle(
                                str(model_dir),
                                output_path,
                                precision="fp16",
                            )
