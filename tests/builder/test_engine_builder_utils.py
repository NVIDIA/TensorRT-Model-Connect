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
