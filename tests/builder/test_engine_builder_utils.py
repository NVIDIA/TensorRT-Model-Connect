# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for engine_builder.py utility functions — pure Python, no TRT needed.

Tests _is_hf_model_dir, _detect_tokenizer_add_special_tokens, _resolve_model
(local path), _ensure_tokenizer_json (skips), and build() public API error paths.

Trace: ARCH-ENG-001, UD-ENG-05
Intent: Validate engine builder utility functions including HF model directory detection, tokenizer special-token detection, model resolution, and public API error paths.
Preconditions: tensorrt_model_connect is importable; no TRT or GPU required.
Postconditions: HF directories are correctly identified, tokenizer special-token flags match HF config, local paths resolve as expected, and invalid inputs raise appropriate errors.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tensorrt_model_connect.engine_builder as engine_builder

try:
    from tensorrt_model_connect.engine_builder import (
        _is_hf_model_dir,
        _detect_tokenizer_add_special_tokens,
        _detect_tokenizer_special_frame,
        _resolve_model,
        _get_trt_version,
        _trt_abi_from_version,
        _get_gpu_name,
        _HF_ALLOW_PATTERNS,
        _diffusion_tokenizer_add_special_tokens_from_plugin,
        _ensure_tokenizer_json,
        _tokenizer_json_bundle_override_from_plugin,
        build_bundle,
    )
    from tensorrt_model_connect.families import family_hf_allow_patterns
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


class TestIsHfModelDir:
    """Test _is_hf_model_dir detection."""

    def test_with_config_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _is_hf_model_dir(tmp_path)

    def test_with_model_index_json(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        assert _is_hf_model_dir(tmp_path)

    def test_empty_dir(self, tmp_path):
        assert not _is_hf_model_dir(tmp_path)

    def test_with_other_files(self, tmp_path):
        (tmp_path / "random.txt").write_text("hello")
        assert not _is_hf_model_dir(tmp_path)


class TestDetectTokenizerAddSpecialTokens:
    """Test _detect_tokenizer_add_special_tokens from tokenizer_config.json."""

    def test_explicit_add_bos_token_true(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": True}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is True

    def test_explicit_add_bos_token_false(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": False}))
        assert _detect_tokenizer_add_special_tokens(tmp_path) is False

    def test_default_encode_differs_despite_false_config(self, tmp_path, monkeypatch):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"add_bos_token": False, "add_eos_token": False}))

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                ids = [101]
                return ids + [102] if add_special_tokens else ids

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=True):
                assert Path(path) == tmp_path
                assert trust_remote_code is True
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        assert _detect_tokenizer_add_special_tokens(tmp_path) is True

    def test_detects_exact_prefix_suffix_frame(self, tmp_path, monkeypatch):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                ids = [11, 12]
                return [1] + ids + [2] if add_special_tokens else ids

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=True):
                assert Path(path) == tmp_path
                assert trust_remote_code is True
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        assert _detect_tokenizer_special_frame(tmp_path) == ([1], [2])

    def test_detects_prefix_only_frame(self, tmp_path, monkeypatch):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                ids = [11, 12]
                return [1] + ids if add_special_tokens else ids

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=True):
                assert Path(path) == tmp_path
                assert trust_remote_code is True
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        assert _detect_tokenizer_special_frame(tmp_path) == ([1], [])

    def test_prepare_preserves_pre_conversion_special_frame(
        self,
        tmp_path,
        monkeypatch,
    ):
        state = {"converted": False}

        class SourceTokenizer:
            def encode(self, _text, add_special_tokens=True):
                return [10] if add_special_tokens else [10]

            def save_pretrained(self, path):
                state["converted"] = True
                (Path(path) / "tokenizer.json").write_text(
                    json.dumps({"model": {"type": "BPE"}}),
                    encoding="utf-8",
                )

        class ConvertedTokenizer:
            def encode(self, _text, add_special_tokens=True):
                return [2, 10] if add_special_tokens else [10]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **_kwargs):
                assert Path(path) == tmp_path
                return ConvertedTokenizer() if state["converted"] else SourceTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        frame = engine_builder._prepare_tokenizer_special_frame(tmp_path)

        assert frame == ([], [])
        assert (tmp_path / "tokenizer.json").is_file()

    def test_prepare_uses_remote_source_contract_for_polluted_snapshot(
        self,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / "tokenizer.json").write_text(
            json.dumps({"model": {"type": "BPE"}}),
            encoding="utf-8",
        )

        class SourceTokenizer:
            def encode(self, _text, add_special_tokens=True):
                return [10] if add_special_tokens else [10]

        class PollutedSnapshotTokenizer:
            def encode(self, _text, add_special_tokens=True):
                return [2, 10] if add_special_tokens else [10]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                if path == "facebook/opt-125m":
                    assert kwargs["local_files_only"] is True
                    return SourceTokenizer()
                assert Path(path) == tmp_path
                return PollutedSnapshotTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        frame = engine_builder._prepare_tokenizer_special_frame(
            tmp_path,
            source_model_id_or_path="facebook/opt-125m",
        )

        assert frame == ([], [])

    def test_tokenizer_generation_does_not_rewrite_sibling_metadata(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_config = tmp_path / "tokenizer_config.json"
        tokenizer_config.write_text('{"add_bos_token": false}', encoding="utf-8")

        class BackendTokenizer:
            def save(self, path):
                Path(path).write_text(
                    json.dumps({"model": {"type": "BPE"}}),
                    encoding="utf-8",
                )

        class FakeTokenizer:
            backend_tokenizer = BackendTokenizer()

            def save_pretrained(self, _path):
                raise AssertionError("save_pretrained would rewrite sibling metadata")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **_kwargs):
                assert Path(path) == tmp_path
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        _ensure_tokenizer_json(tmp_path)

        assert tokenizer_config.read_text(encoding="utf-8") == (
            '{"add_bos_token": false}'
        )
        assert (tmp_path / "tokenizer.json").is_file()

    def test_no_tokenizer_config(self, tmp_path):
        # No tokenizer_config.json and no transformers — should return False
        result = _detect_tokenizer_add_special_tokens(tmp_path)
        assert result is False

    def test_invalid_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("not json{{{")
        # Should not crash, falls through to fallback
        result = _detect_tokenizer_add_special_tokens(tmp_path)
        assert isinstance(result, bool)

    def test_diffusion_detects_tokenizer_subdir(self, tmp_path):
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({"add_eos_token": True}))

        class FakeDiffusionPlugin:
            name = "fake_diffusion"

            def diffusion_tokenizer_add_special_tokens(
                self, model_dir_path, *, detect_tokenizer_add_special_tokens,
            ):
                tok_dir = Path(model_dir_path) / "tokenizer"
                return detect_tokenizer_add_special_tokens(tok_dir)

        assert _diffusion_tokenizer_add_special_tokens_from_plugin(
            FakeDiffusionPlugin(), tmp_path) is True

    def test_diffusion_prefers_tokenizer_2(self, tmp_path):
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer_config.json").write_text(
            json.dumps({"add_eos_token": True}))

        tok2_dir = tmp_path / "tokenizer_2"
        tok2_dir.mkdir()
        (tok2_dir / "tokenizer_config.json").write_text(
            json.dumps({"add_eos_token": False}))

        class FakeDiffusionPlugin:
            name = "fake_diffusion"

            def diffusion_tokenizer_add_special_tokens(
                self, model_dir_path, *, detect_tokenizer_add_special_tokens,
            ):
                model_dir = Path(model_dir_path)
                for subdir in ("tokenizer_2", "tokenizer"):
                    candidate = model_dir / subdir
                    if candidate.is_dir():
                        return detect_tokenizer_add_special_tokens(candidate)
                return detect_tokenizer_add_special_tokens(model_dir)

        assert _diffusion_tokenizer_add_special_tokens_from_plugin(
            FakeDiffusionPlugin(), tmp_path) is False

    def test_diffusion_detects_exact_tokenizer_special_frame(self, tmp_path):
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        class FakeDiffusionPlugin:
            name = "fake_diffusion"

            def diffusion_tokenizer_special_frame(
                self, model_dir_path, *, detect_tokenizer_special_frame,
            ):
                assert Path(model_dir_path) == tmp_path
                return detect_tokenizer_special_frame(tokenizer_dir)

        detector_calls = []

        def detector(path):
            detector_calls.append(Path(path))
            return [], [1]

        assert engine_builder._diffusion_tokenizer_special_frame_from_plugin(
            FakeDiffusionPlugin(), tmp_path, detect_tokenizer_special_frame=detector
        ) == ([], [1])
        assert detector_calls == [tokenizer_dir]


class TestResolveModel:
    """Test _resolve_model with local directories."""

    def test_local_dir_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)

    def test_local_dir_with_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)


class TestGetTrtVersion:
    """Test _get_trt_version helper."""

    def test_returns_string(self):
        result = _get_trt_version()
        assert isinstance(result, str)
        # Should either be a version string or "unknown"
        assert result == "unknown" or "." in result


class TestTrtAbiFromVersion:
    def test_extracts_major_minor(self):
        assert _trt_abi_from_version("10.15.0.6") == "10.15"
        assert _trt_abi_from_version("11.0") == "11.0"

    def test_unknown_returns_empty(self):
        assert _trt_abi_from_version("unknown") == ""


class TestGetGpuName:
    """Test _get_gpu_name helper."""

    def test_returns_active_cuda_device_not_first_physical_gpu(self, monkeypatch):
        """Records the CUDA build device on a heterogeneous GPU host."""
        props = SimpleNamespace(name=b"NVIDIA GB300")
        mock_cudart = SimpleNamespace(
            cudaGetDevice=Mock(return_value=(0, 0)),
            cudaGetDeviceProperties=Mock(return_value=(0, props)),
        )
        mock_smi = SimpleNamespace(
            returncode=0,
            stdout=(
                "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition\n"
                "NVIDIA GB300\n"
            ),
        )
        monkeypatch.setattr(engine_builder, "cudart", mock_cudart, raising=False)
        mock_smi_run = Mock(return_value=mock_smi)
        monkeypatch.setattr("subprocess.run", mock_smi_run)

        assert _get_gpu_name() == "NVIDIA GB300"
        mock_cudart.cudaGetDeviceProperties.assert_called_once_with(0)
        mock_smi_run.assert_not_called()

    def test_uses_programmatically_selected_cuda_device(self, monkeypatch):
        props = SimpleNamespace(name="NVIDIA B200\x00")
        mock_cudart = SimpleNamespace(
            cudaGetDevice=Mock(return_value=(0, 2)),
            cudaGetDeviceProperties=Mock(return_value=(0, props)),
        )
        monkeypatch.setattr(engine_builder, "cudart", mock_cudart)

        assert _get_gpu_name() == "NVIDIA B200"
        mock_cudart.cudaGetDeviceProperties.assert_called_once_with(2)

    def test_returns_empty_when_cuda_runtime_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(engine_builder, "cudart", None)
        assert _get_gpu_name() == ""

    def test_returns_empty_when_current_device_query_fails(self, monkeypatch):
        mock_cudart = SimpleNamespace(
            cudaGetDevice=Mock(return_value=(1, 0)),
            cudaGetDeviceProperties=Mock(),
        )
        monkeypatch.setattr(engine_builder, "cudart", mock_cudart)

        assert _get_gpu_name() == ""
        mock_cudart.cudaGetDeviceProperties.assert_not_called()

    def test_returns_empty_when_device_properties_query_fails(self, monkeypatch):
        mock_cudart = SimpleNamespace(
            cudaGetDevice=Mock(return_value=(0, 0)),
            cudaGetDeviceProperties=Mock(return_value=(1, None)),
        )
        monkeypatch.setattr(engine_builder, "cudart", mock_cudart)

        assert _get_gpu_name() == ""

    def test_returns_string(self):
        result = _get_gpu_name()
        assert isinstance(result, str)


class TestHfAllowPatterns:
    """Test that _HF_ALLOW_PATTERNS contains essential patterns."""

    def test_contains_config(self):
        assert "config.json" in _HF_ALLOW_PATTERNS

    def test_contains_safetensors(self):
        assert "model.safetensors" in _HF_ALLOW_PATTERNS

    def test_contains_tokenizer(self):
        assert "tokenizer.json" in _HF_ALLOW_PATTERNS

    def test_contains_wordpiece_vocab(self):
        assert "vocab.txt" in _HF_ALLOW_PATTERNS

    def test_contains_processor_config(self):
        assert "processor_config.json" in _HF_ALLOW_PATTERNS

    def test_contains_sharded(self):
        assert "model-*.safetensors" in _HF_ALLOW_PATTERNS

    def test_contains_sentencepiece_variants(self):
        assert "*.model" in _HF_ALLOW_PATTERNS
        assert "*.spm" in _HF_ALLOW_PATTERNS

    def test_contains_remote_code(self):
        assert "*.py" in _HF_ALLOW_PATTERNS

    def test_contains_diffusers_component_dirs(self):
        shared = set(_HF_ALLOW_PATTERNS)
        family_owned = set(family_hf_allow_patterns())
        for pattern in (
            "text_encoder/**",
            "text_encoder_2/**",
            "transformer/**",
            "vae/**",
            "tokenizer_2/**",
        ):
            assert pattern not in shared
            assert pattern in family_owned


class TestEnsureTokenizerJson:
    """Test _ensure_tokenizer_json skips when tokenizer.json exists."""

    def test_skips_if_exists(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        # Should not raise or modify
        _ensure_tokenizer_json(tmp_path)
        assert (tmp_path / "tokenizer.json").read_text() == "{}"

    def test_rebuilds_undersized_wordpiece_from_complete_vocab(
        self,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 6}))
        (tmp_path / "vocab.txt").write_text(
            "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n"
        )
        (tmp_path / "tokenizer.json").write_text(json.dumps({
            "model": {
                "type": "WordPiece",
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "[MASK]": 4,
                },
            },
        }))

        class FakeTokenizer:
            @staticmethod
            def save_pretrained(path):
                (Path(path) / "tokenizer.json").write_text(json.dumps({
                    "model": {
                        "type": "WordPiece",
                        "vocab": {
                            "[PAD]": 0,
                            "[UNK]": 1,
                            "[CLS]": 2,
                            "[SEP]": 3,
                            "[MASK]": 4,
                            "hello": 5,
                        },
                    },
                }))

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, use_fast=False):
                assert Path(path) == tmp_path
                assert use_fast is False
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        _ensure_tokenizer_json(tmp_path)

        tokenizer = json.loads((tmp_path / "tokenizer.json").read_text())
        assert tokenizer["model"]["vocab"]["hello"] == 5

    def test_missing_fast_tokenizer_delegates_to_family_plugin(
        self,
        tmp_path,
        monkeypatch,
    ):
        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, use_fast=False):
                assert Path(path) == tmp_path
                assert use_fast is False
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        captured = {}

        class FakePlugin:
            def ensure_tokenizer_json(self, model_dir, *, previous_error=None):
                captured["model_dir"] = Path(model_dir)
                captured["previous_error"] = previous_error
                (Path(model_dir) / "tokenizer.json").write_text("{}")
                return True

        _ensure_tokenizer_json(tmp_path, plugin=FakePlugin())

        assert captured["model_dir"] == tmp_path
        assert "slow tokenizer conversion failed" in captured["previous_error"]
        assert (tmp_path / "tokenizer.json").exists()


def test_tokenizer_json_bundle_override_is_family_owned(tmp_path):
    captured = {}

    class FakePlugin:
        name = "fake"

        def tokenizer_json_bundle_override(self, model_dir):
            captured["model_dir"] = Path(model_dir)
            return b'{"pre_tokenizer": "hf"}'

    assert _tokenizer_json_bundle_override_from_plugin(
        FakePlugin(),
        tmp_path,
    ) == b'{"pre_tokenizer": "hf"}'
    assert captured["model_dir"] == tmp_path


class TestBuildBundleErrors:
    """Test build_bundle error handling."""

    def test_unsupported_model_type(self, tmp_path):
        config = {
            "model_type": "totally_unsupported_model_xyz",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": 32,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        with pytest.raises(ValueError, match="No family plugin"):
            build_bundle(str(tmp_path), str(tmp_path / "out.trtfb"))
