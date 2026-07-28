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
import os
import stat
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
        build_bundle,
    )
    from tensorrt_model_connect.families import family_hf_allow_patterns
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


def _native_tokenizer_payload(
    model_type,
    *,
    model_fields=None,
    **document_fields,
):
    if model_type == "BPE":
        model = {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    elif model_type == "WordPiece":
        model = {"type": "WordPiece", "vocab": {"[UNK]": 0}}
    else:
        model = {"type": "Unigram", "vocab": [["<unk>", -1.0]]}
    model.update(model_fields or {})
    return {"model": model, **document_fields}


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
                assert trust_remote_code is False
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
                assert trust_remote_code is False
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
                assert trust_remote_code is False
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
                    json.dumps(
                        {
                            "model": {
                                "type": "BPE",
                                "vocab": {"hello": 0},
                                "merges": [],
                            }
                        }
                    ),
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
            json.dumps(
                {
                    "model": {
                        "type": "BPE",
                        "vocab": {"hello": 0},
                        "merges": [],
                    }
                }
            ),
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
                    json.dumps(
                        {
                            "model": {
                                "type": "BPE",
                                "vocab": {"hello": 0},
                                "merges": [],
                            }
                        }
                    ),
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

    def test_explicit_trust_remote_code_reaches_tokenizer(self, tmp_path, monkeypatch):
        captured = []

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1, 2] if add_special_tokens else [2]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=False):
                assert Path(path) == tmp_path
                captured.append(trust_remote_code)
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        assert _detect_tokenizer_special_frame(
            tmp_path,
            trust_remote_code=True,
        ) == ([1], [])
        assert captured == [True]

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

    def test_diffusion_probe_forwards_explicit_remote_code_trust(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        captured = []

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [1, 2] if add_special_tokens else [2]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, trust_remote_code=False):
                assert Path(path) == tokenizer_dir
                captured.append(trust_remote_code)
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        class FakeDiffusionPlugin:
            def diffusion_tokenizer_special_frame(
                self,
                model_dir_path,
                *,
                detect_tokenizer_special_frame,
            ):
                assert Path(model_dir_path) == tmp_path
                return detect_tokenizer_special_frame(tokenizer_dir)

        assert engine_builder._diffusion_tokenizer_special_frame_from_plugin(
            FakeDiffusionPlugin(),
            tmp_path,
            trust_remote_code=True,
        ) == ([1], [])
        assert captured == [True]


class TestResolveModel:
    """Test _resolve_model with local directories."""

    def test_local_dir_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)

    def test_local_dir_with_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text("{}")
        assert _resolve_model(str(tmp_path)) == str(tmp_path)

    def test_native_builder_forwards_remote_code_trust_to_bundle(
        self,
        tmp_path,
        monkeypatch,
    ):
        captured = []
        monkeypatch.setattr(
            engine_builder,
            "_resolve_model",
            lambda _model, **_kwargs: str(tmp_path),
        )
        monkeypatch.setattr(
            engine_builder,
            "build_bundle",
            lambda *_args, **kwargs: captured.append(kwargs["trust_remote_code"]),
        )

        engine_builder._build_native_impl(
            "example/model",
            str(tmp_path / "out.trtfb"),
            trust_remote_code=True,
        )

        assert captured == [True]

    @pytest.mark.parametrize(
        "invalid_trust",
        ("false", 1, None),
        ids=("string-false", "integer-one", "none"),
    )
    def test_native_builder_rejects_non_boolean_remote_code_trust_before_resolution(
        self,
        monkeypatch,
        invalid_trust,
    ):
        resolver = Mock()
        monkeypatch.setattr(engine_builder, "_resolve_model", resolver)

        with pytest.raises(TypeError, match="trust_remote_code must be a bool"):
            engine_builder._build_native_impl(
                "example/model",
                "out.trtfb",
                trust_remote_code=invalid_trust,
            )

        resolver.assert_not_called()


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
        payload = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0},
                "merges": [],
            }
        }
        (tmp_path / "tokenizer.json").write_text(json.dumps(payload))
        # Should not raise or modify
        _ensure_tokenizer_json(tmp_path)
        assert json.loads((tmp_path / "tokenizer.json").read_text()) == payload

    @pytest.mark.parametrize(
        "payload",
        (
            {"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}},
            {"model": {"type": "WordPiece", "vocab": {"[UNK]": 0}}},
            {"model": {"type": "Unigram", "vocab": [["<unk>", -1.0]]}},
            {"model": {"vocab": {"a": 0}, "merges": []}},
            {
                "model": {
                    "vocab": {"[UNK]": 0},
                    "continuing_subword_prefix": "##",
                }
            },
            {"model": {"vocab": [["<unk>", -1.0]]}},
        ),
        ids=(
            "bpe",
            "wordpiece",
            "unigram",
            "legacy-bpe",
            "legacy-wordpiece",
            "legacy-unigram",
        ),
    )
    def test_native_compatibility_accepts_supported_minimal_models(
        self,
        tmp_path,
        payload,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(payload))

        assert engine_builder._native_tokenizer_json_error(tokenizer_path) is None

    def test_native_compatibility_distinguishes_absent_and_null_model_type(
        self,
        tmp_path,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        payload = {"model": {"vocab": {"a": 0}, "merges": []}}
        tokenizer_path.write_text(json.dumps(payload))
        assert engine_builder._native_tokenizer_json_error(tokenizer_path) is None

        payload["model"]["type"] = None
        tokenizer_path.write_text(json.dumps(payload))
        assert (
            "model.type must be a string when present"
            in engine_builder._native_tokenizer_json_error(tokenizer_path)
        )

    @pytest.mark.parametrize(
        "path_kind",
        ("fifo", "directory", "device"),
    )
    def test_native_compatibility_rejects_nonregular_paths_without_reading(
        self,
        tmp_path,
        path_kind,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        if path_kind == "fifo":
            os.mkfifo(tokenizer_path)
        elif path_kind == "directory":
            tokenizer_path.mkdir()
        else:
            tokenizer_path.symlink_to("/dev/null")

        assert (
            engine_builder._native_tokenizer_json_error(tokenizer_path)
            == "tokenizer.json must resolve to a regular file"
        )

    def test_native_compatibility_accepts_symlink_to_regular_file(
        self,
        tmp_path,
    ):
        target_path = tmp_path / "stored-tokenizer.json"
        target_path.write_text(
            json.dumps(_native_tokenizer_payload("BPE")),
            encoding="utf-8",
        )
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.symlink_to(target_path.name)

        assert engine_builder._native_tokenizer_json_error(tokenizer_path) is None

    def test_native_compatibility_reports_invalid_utf8_without_escaping(
        self,
        tmp_path,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_bytes(b"\xff\xfe")

        assert (
            "cannot decode tokenizer.json as UTF-8"
            in engine_builder._native_tokenizer_json_error(tokenizer_path)
        )

    def test_native_compatibility_reports_deep_json_without_recursion_escape(
        self,
        tmp_path,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(
            '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
            f'"ignored":{"[" * 20000}0{"]" * 20000}' + "}",
            encoding="utf-8",
        )

        error = engine_builder._native_tokenizer_json_error(tokenizer_path)

        assert error is not None
        assert "nesting" in error

    @pytest.mark.parametrize("model_type", ("BPE", "WordPiece"))
    def test_native_compatibility_rejects_negative_required_added_token_ids(
        self,
        tmp_path,
        model_type,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(
            json.dumps(
                _native_tokenizer_payload(
                    model_type,
                    added_tokens=[{"content": "<s>", "id": -1}],
                )
            ),
            encoding="utf-8",
        )

        assert (
            "added_tokens[0].id must be non-negative"
            in engine_builder._native_tokenizer_json_error(tokenizer_path)
        )

    def test_native_compatibility_preserves_optional_unigram_negative_id(
        self,
        tmp_path,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(
            json.dumps(
                _native_tokenizer_payload(
                    "Unigram",
                    added_tokens=[{"content": "<s>", "id": -1}],
                )
            ),
            encoding="utf-8",
        )

        assert engine_builder._native_tokenizer_json_error(tokenizer_path) is None

    @pytest.mark.parametrize(
        ("raw", "message"),
        (
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"ignored":NaN}',
                "invalid tokenizer.json",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"ignored":Infinity}',
                "invalid tokenizer.json",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"ignored":-Infinity}',
                "invalid tokenizer.json",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"ignored":1e400}',
                "JSON-overflow",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                f'"ignored":{10**309}' + "}",
                "native JSON number envelope",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"ignored":"\\ud800"}',
                "unpaired UTF-16 surrogate",
            ),
            (
                '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
                '"\\udc00":"ignored"}',
                "key with an unpaired UTF-16 surrogate",
            ),
        ),
        ids=(
            "nan",
            "infinity",
            "negative-infinity",
            "float-parse-overflow",
            "integer-parse-overflow",
            "lone-high-surrogate",
            "lone-low-surrogate-key",
        ),
    )
    def test_native_compatibility_rejects_values_native_json_cannot_parse(
        self,
        tmp_path,
        raw,
        message,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(raw)

        assert message in engine_builder._native_tokenizer_json_error(tokenizer_path)

    @pytest.mark.parametrize(
        ("payload", "message"),
        (
            (
                _native_tokenizer_payload(
                    "BPE", model_fields={"byte_fallback": "false"}
                ),
                "byte_fallback must be a boolean",
            ),
            (
                _native_tokenizer_payload("BPE", added_tokens=[{"id": 1}]),
                "content is required",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    added_tokens=[{"content": "<s>", "id": 1, "special": 1}],
                ),
                "special must be a boolean",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    added_tokens=[{"content": "<s>", "id": 2**31 - 1}],
                ),
                "contiguous native vocabulary allocation bound",
            ),
            (
                _native_tokenizer_payload("BPE", pre_tokenizer=[]),
                "pre_tokenizer must be an object",
            ),
            (
                _native_tokenizer_payload(
                    "BPE", pre_tokenizer={"type": None}
                ),
                "pre_tokenizer.type must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    pre_tokenizer={
                        "type": "Split",
                        "pattern": {"Regex": "ignored"},
                    },
                ),
                "direct Split requires",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    pre_tokenizer={
                        "type": "Sequence",
                        "pretokenizers": [
                            {
                                "type": "Split",
                                "pattern": {
                                    "Regex": r"[^\r\n]\p{N}{1,３}"
                                },
                            }
                        ],
                    },
                ),
                "std::stoi",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    normalizer={
                        "type": "Replace",
                        "pattern": {"String": 1},
                    },
                ),
                "pattern.String must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    decoder={
                        "type": "Sequence",
                        "decoders": [
                            {"type": "Replace", "pattern": None, "content": 1}
                        ],
                    },
                ),
                "content must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    decoder={
                        "type": "Sequence",
                        "decoders": [
                            {"type": "Strip", "content": " ", "start": 2**31}
                        ],
                    },
                ),
                "start must be a signed 32-bit integer",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    post_processor={
                        "type": "TemplateProcessing",
                        "single": [{"SpecialToken": {"id": 1}}],
                    },
                ),
                "SpecialToken.id must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    post_processor={
                        "type": "TemplateProcessing",
                        "single": {
                            "entry": {"SpecialToken": {"id": 1}},
                        },
                    },
                ),
                "SpecialToken.id must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "BPE",
                    post_processor={
                        "type": "RobertaProcessing",
                        "cls": ["<s>", "0"],
                    },
                ),
                "cls[1] must be a signed 32-bit integer",
            ),
            (
                _native_tokenizer_payload(
                    "WordPiece", model_fields={"unk_token": None}
                ),
                "unk_token must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "WordPiece",
                    model_fields={"max_input_chars_per_word": 2**31},
                ),
                "max_input_chars_per_word must be a signed 32-bit integer",
            ),
            (
                _native_tokenizer_payload(
                    "WordPiece",
                    normalizer={
                        "type": "BertNormalizer",
                        "lowercase": 1,
                    },
                ),
                "lowercase must be a boolean",
            ),
            (
                _native_tokenizer_payload(
                    "WordPiece",
                    normalizer={
                        "type": "Sequence",
                        "normalizers": [{"type": None}],
                    },
                ),
                "normalizers[0].type must be a string",
            ),
            (
                _native_tokenizer_payload(
                    "WordPiece",
                    post_processor={
                        "type": "BertProcessing",
                        "sep": ["[SEP]", None],
                    },
                ),
                "sep[1] must be a signed 32-bit integer",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram", model_fields={"vocab": [["<unk>", 10**100]]}
                ),
                "finite float32",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram", model_fields={"unk_id": True}
                ),
                "signed 32-bit index",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram",
                    added_tokens=[{"content": "<s>", "id": 2**31 - 1}],
                ),
                "contiguous native vocabulary allocation bound",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram", added_tokens=[{"id": "1"}]
                ),
                "added_tokens[0].id must be a signed 32-bit integer",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram",
                    pre_tokenizer={
                        "type": "Metaspace",
                        "add_prefix_space": "true",
                    },
                ),
                "add_prefix_space must be a boolean",
            ),
            (
                _native_tokenizer_payload(
                    "Unigram",
                    post_processor={
                        "type": "TemplateProcessing",
                        "single": [{"SpecialToken": None}],
                    },
                ),
                "SpecialToken must be an object",
            ),
        ),
        ids=(
            "bpe-byte-fallback",
            "bpe-added-token-required-content",
            "bpe-added-token-special",
            "bpe-added-token-allocation",
            "bpe-pretokenizer-object",
            "bpe-pretokenizer-type",
            "bpe-direct-split",
            "bpe-ascii-stoi",
            "bpe-normalizer-replace",
            "bpe-decoder-replace",
            "bpe-decoder-strip",
            "bpe-template-token",
            "bpe-template-object-values",
            "bpe-roberta-id",
            "wordpiece-unk-token",
            "wordpiece-max-chars",
            "wordpiece-normalizer-bool",
            "wordpiece-sequence-child",
            "wordpiece-postprocessor-id",
            "unigram-score-overflow",
            "unigram-unk-id",
            "unigram-added-token-allocation",
            "unigram-added-token-id",
            "unigram-metaspace-bool",
            "unigram-template-token",
        ),
    )
    def test_native_compatibility_rejects_consumed_runtime_field_mismatches(
        self,
        tmp_path,
        payload,
        message,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(payload))

        assert message in engine_builder._native_tokenizer_json_error(tokenizer_path)

    @pytest.mark.parametrize(
        "payload",
        (
            {
                **_native_tokenizer_payload("BPE"),
                "ignored": 1e100,
            },
            {
                **_native_tokenizer_payload("BPE"),
                "ignored": "\ud83d\ude00",
            },
            _native_tokenizer_payload(
                "BPE",
                pre_tokenizer={
                    "type": "Sequence",
                    "pretokenizers": [
                        {"type": "Split"},
                        {"type": "Split", "pattern": None},
                        {
                            "type": "Split",
                            "pattern": {"String": 1},
                        },
                        {
                            "type": "Split",
                            "pattern": {"Regex": r"\p{N}{1,3}"},
                        },
                        None,
                    ],
                },
            ),
            _native_tokenizer_payload(
                "BPE",
                pre_tokenizer={
                    "type": "Sequence",
                    "pretokenizers": [
                        {
                            "type": "Split",
                            "pattern": {"Regex": r"\p{N}{1,３}"},
                        },
                    ],
                },
            ),
            _native_tokenizer_payload(
                "BPE",
                normalizer={
                    "type": "Replace",
                    "pattern": None,
                    "content": None,
                },
                decoder={
                    "type": "Sequence",
                    "decoders": [
                        {"type": "Replace", "pattern": None, "content": ""},
                        {"type": "Strip", "content": "x", "start": None},
                    ],
                },
            ),
            _native_tokenizer_payload(
                "BPE",
                normalizer={
                    "type": "Sequence",
                    "normalizers": [{"type": "Prepend"}, None],
                },
                post_processor={
                    "type": "Sequence",
                    "processors": [
                        {"type": "TemplateProcessing", "single": None},
                        None,
                    ],
                },
            ),
            _native_tokenizer_payload(
                "BPE",
                post_processor={
                    "type": "TemplateProcessing",
                    "single": [
                        {
                            "Sequence": {"id": "A"},
                            "SpecialToken": None,
                        }
                    ],
                },
            ),
            _native_tokenizer_payload(
                "WordPiece",
                normalizer={"type": "Sequence", "normalizers": None},
                post_processor={
                    "type": "RobertaProcessing",
                    "cls": None,
                    "sep": [],
                },
            ),
            _native_tokenizer_payload(
                "Unigram",
                added_tokens=[{"content": "", "id": 2**31 - 1}],
                normalizer={"type": "Sequence", "normalizers": None},
                pre_tokenizer={
                    "type": "Sequence",
                    "pretokenizers": [
                        {"type": "Metaspace", "add_prefix_space": True},
                        None,
                    ],
                },
                post_processor={
                    "type": "TemplateProcessing",
                    "single": {
                        "entry": {"SpecialToken": {"id": 1}},
                    },
                },
            ),
        ),
        ids=(
            "ignored-large-finite-double",
            "valid-surrogate-pair",
            "bpe-first-consumed-split-regex",
            "bpe-non-qwen-regex-skips-digit-group",
            "bpe-replace-and-strip-short-circuits",
            "bpe-first-prepend-and-template",
            "template-sequence-priority",
            "wordpiece-null-and-short-sections",
            "unigram-object-template-is-ignored",
        ),
    )
    def test_native_compatibility_allows_native_short_circuit_shapes(
        self,
        tmp_path,
        payload,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(payload))

        assert engine_builder._native_tokenizer_json_error(tokenizer_path) is None

    @pytest.mark.parametrize(
        ("raw", "message"),
        (
            ("{}", "object-valued model"),
            ("{", "invalid tokenizer.json"),
            (
                json.dumps({"model": {"type": "WordLevel", "vocab": {"a": 0}}}),
                "unsupported tokenizer model.type",
            ),
            (
                json.dumps({"model": {"type": "BPE", "vocab": {"a": 1}, "merges": []}}),
                "must cover 0..0",
            ),
            (
                json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}}}),
                "model.merges must be an array",
            ),
            (
                json.dumps({"model": {"type": "Unigram", "vocab": [["a", True]]}}),
                "string token and numeric score",
            ),
        ),
        ids=(
            "missing-model",
            "invalid-json",
            "unknown-type",
            "noncontiguous-vocab",
            "missing-merges",
            "invalid-unigram-score",
        ),
    )
    def test_native_compatibility_rejects_runtime_invalid_models(
        self,
        tmp_path,
        raw,
        message,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(raw)

        assert message in engine_builder._native_tokenizer_json_error(tokenizer_path)

    @pytest.mark.parametrize(
        "invalid_trust",
        ("false", 1, None),
        ids=("string-false", "integer-one", "none"),
    )
    def test_rejects_non_boolean_remote_code_trust_before_existing_file_shortcut(
        self,
        tmp_path,
        invalid_trust,
    ):
        (tmp_path / "tokenizer.json").write_text(json.dumps({
            "model": {
                "type": "BPE",
                "vocab": {"a": 0},
                "merges": [],
            },
        }))

        with pytest.raises(TypeError, match="trust_remote_code must be a bool"):
            _ensure_tokenizer_json(
                tmp_path,
                trust_remote_code=invalid_trust,
            )

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
            def from_pretrained(
                path,
                use_fast=False,
                trust_remote_code=True,
            ):
                assert Path(path) == tmp_path
                assert use_fast is False
                assert trust_remote_code is False
                return FakeTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        _ensure_tokenizer_json(tmp_path)

        tokenizer = json.loads((tmp_path / "tokenizer.json").read_text())
        assert tokenizer["model"]["vocab"]["hello"] == 5
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_missing_fast_tokenizer_delegates_to_family_plugin(
        self,
        tmp_path,
        monkeypatch,
    ):
        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(
                path,
                use_fast=False,
                trust_remote_code=True,
            ):
                assert Path(path) == tmp_path
                assert use_fast is False
                assert trust_remote_code is True
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        captured = {}

        class FakePlugin:
            def ensure_tokenizer_json(
                self,
                model_dir,
                *,
                previous_error=None,
                trust_remote_code=False,
            ):
                captured["model_dir"] = Path(model_dir)
                captured["previous_error"] = previous_error
                captured["trust_remote_code"] = trust_remote_code
                (Path(model_dir) / "tokenizer.json").write_text(json.dumps({
                    "model": {
                        "type": "Unigram",
                        "vocab": [["<unk>", -1.0]],
                    },
                }))
                return True

        _ensure_tokenizer_json(
            tmp_path,
            plugin=FakePlugin(),
            trust_remote_code=True,
        )

        assert captured["model_dir"] == tmp_path
        assert "slow tokenizer conversion failed" in captured["previous_error"]
        assert captured["trust_remote_code"] is True
        assert (tmp_path / "tokenizer.json").exists()

    @pytest.mark.parametrize(
        "symlink_kind",
        ("relative", "absolute-to-temporary-file"),
    )
    def test_generated_symlink_is_rejected_before_family_fallback(
        self,
        tmp_path,
        monkeypatch,
        symlink_kind,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"quarantined before both attempts"}'
        tokenizer_path.write_bytes(original)

        class SymlinkTokenizer:
            @staticmethod
            def save_pretrained(path):
                generated_dir = Path(path)
                real_path = generated_dir / "real-tokenizer.json"
                real_path.write_text(
                    json.dumps(_native_tokenizer_payload("BPE")),
                    encoding="utf-8",
                )
                link_target = (
                    real_path.name
                    if symlink_kind == "relative"
                    else real_path.resolve()
                )
                (generated_dir / "tokenizer.json").symlink_to(link_target)

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                assert not tokenizer_path.is_symlink()
                return SymlinkTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        captured = {}

        class FamilyFallback:
            @staticmethod
            def ensure_tokenizer_json(model_dir, *, previous_error=None):
                assert not tokenizer_path.exists()
                assert not tokenizer_path.is_symlink()
                captured["previous_error"] = previous_error
                tokenizer_path.write_text(
                    json.dumps(_native_tokenizer_payload("Unigram")),
                    encoding="utf-8",
                )
                return True

        _ensure_tokenizer_json(tmp_path, plugin=FamilyFallback())

        assert "regular, non-symlink file" in captured["previous_error"]
        assert tokenizer_path.is_file()
        assert not tokenizer_path.is_symlink()
        assert json.loads(tokenizer_path.read_text())["model"]["type"] == "Unigram"
        assert not (tmp_path / "real-tokenizer.json").exists()
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_empty_generated_file_is_rejected_and_original_is_restored(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"restore after empty generation"}'
        tokenizer_path.write_bytes(original)

        class EmptyTokenizer:
            @staticmethod
            def save_pretrained(path):
                (Path(path) / "tokenizer.json").write_bytes(b"")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                return EmptyTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        with pytest.raises(RuntimeError, match="non-empty regular file"):
            _ensure_tokenizer_json(tmp_path)

        assert tokenizer_path.read_bytes() == original
        assert not tokenizer_path.is_symlink()
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_deep_existing_json_reaches_normal_repair_failure_and_rollback(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = (
            '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
            f'"ignored":{"[" * 20000}0{"]" * 20000}' + "}"
        ).encode()
        tokenizer_path.write_bytes(original)
        monkeypatch.setattr(
            engine_builder,
            "_generate_standard_tokenizer_json_transactionally",
            lambda *args, **kwargs: "conversion unavailable",
        )

        with pytest.raises(RuntimeError, match="refusing to write a bundle"):
            _ensure_tokenizer_json(tmp_path)

        assert tokenizer_path.read_bytes() == original
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_standard_cleanup_failure_preserves_original_at_recovery_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"standard cleanup recovery"}'
        tokenizer_path.write_bytes(original)

        class FailingAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("conversion unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FailingAutoTokenizer),
        )
        monkeypatch.setattr(
            engine_builder,
            "_remove_tokenizer_candidate",
            lambda _path: (_ for _ in ()).throw(
                OSError("deterministic cleanup failure")
            ),
        )

        error = engine_builder._generate_standard_tokenizer_json_transactionally(
            tmp_path,
            tokenizer_path,
            trust_remote_code=False,
        )

        assert error is not None
        assert "original tokenizer.json is preserved at" in error
        recovery_paths = list(
            tmp_path.glob(
                ".trtmc-tokenizer-recovery-*/original-tokenizer.json"
            )
        )
        assert len(recovery_paths) == 1
        assert recovery_paths[0].read_bytes() == original
        assert not tokenizer_path.exists()

    @pytest.mark.parametrize(
        "partial_cleanup",
        (False, True),
        ids=("complete-cleanup", "partial-cleanup"),
    )
    def test_standard_committed_repair_cleanup_is_best_effort(
        self,
        tmp_path,
        monkeypatch,
        capsys,
        partial_cleanup,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(
            '{"malformed":"discard only after commit"}',
            encoding="utf-8",
        )

        class ValidTokenizer:
            @staticmethod
            def save_pretrained(path):
                (Path(path) / "tokenizer.json").write_text(
                    json.dumps(_native_tokenizer_payload("BPE")),
                    encoding="utf-8",
                )

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return ValidTokenizer()

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        if partial_cleanup:
            real_rmtree = engine_builder.shutil.rmtree

            def fail_after_partial_recovery_cleanup(path, *args, **kwargs):
                cleanup_path = Path(path)
                if cleanup_path.name.startswith(
                    ".trtmc-tokenizer-recovery-"
                ):
                    (
                        cleanup_path / "original-tokenizer.json"
                    ).unlink()
                    (cleanup_path / "cleanup-residue").write_text(
                        "partial",
                        encoding="utf-8",
                    )
                    raise OSError("deterministic partial cleanup failure")
                return real_rmtree(path, *args, **kwargs)

            monkeypatch.setattr(
                engine_builder.shutil,
                "rmtree",
                fail_after_partial_recovery_cleanup,
            )

        error = engine_builder._generate_standard_tokenizer_json_transactionally(
            tmp_path,
            tokenizer_path,
            trust_remote_code=False,
        )

        assert error is None
        assert (
            json.loads(tokenizer_path.read_text())["model"]["type"]
            == "BPE"
        )
        stderr = capsys.readouterr().err
        recovery_dirs = list(
            tmp_path.glob(".trtmc-tokenizer-recovery-*")
        )
        if partial_cleanup:
            assert len(recovery_dirs) == 1
            assert (
                recovery_dirs[0] / "cleanup-residue"
            ).read_text() == "partial"
            assert not (
                recovery_dirs[0] / "original-tokenizer.json"
            ).exists()
            assert (
                "recovery directory may contain residual files" in stderr
            )
            assert "original tokenizer.json is preserved" not in stderr
        else:
            assert not recovery_dirs
            assert "cleanup of the previous artifact" not in stderr

    def test_outer_family_committed_repair_survives_partial_cleanup(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_text(
            '{"malformed":"outer family original"}',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            engine_builder,
            "_generate_standard_tokenizer_json_transactionally",
            lambda *args, **kwargs: "conversion unavailable",
        )

        class SuccessfulPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                (Path(model_dir) / "tokenizer.json").write_text(
                    json.dumps(_native_tokenizer_payload("BPE")),
                    encoding="utf-8",
                )
                return True

        real_rmtree = engine_builder.shutil.rmtree

        def fail_after_partial_recovery_cleanup(path, *args, **kwargs):
            cleanup_path = Path(path)
            if cleanup_path.name.startswith(
                ".trtmc-required-tokenizer-recovery-"
            ):
                (cleanup_path / "original-tokenizer.json").unlink()
                (cleanup_path / "cleanup-residue").write_text(
                    "partial",
                    encoding="utf-8",
                )
                raise OSError("deterministic partial cleanup failure")
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(
            engine_builder.shutil,
            "rmtree",
            fail_after_partial_recovery_cleanup,
        )

        _ensure_tokenizer_json(tmp_path, plugin=SuccessfulPlugin())

        assert (
            json.loads(tokenizer_path.read_text())["model"]["type"]
            == "BPE"
        )
        recovery_dirs = list(
            tmp_path.glob(".trtmc-required-tokenizer-recovery-*")
        )
        assert len(recovery_dirs) == 1
        assert (
            recovery_dirs[0] / "cleanup-residue"
        ).read_text() == "partial"
        assert not (
            recovery_dirs[0] / "original-tokenizer.json"
        ).exists()
        stderr = capsys.readouterr().err
        assert "recovery directory may contain residual files" in stderr
        assert "original tokenizer.json is preserved" not in stderr
        assert "family tokenizer hook failed" not in stderr

    def test_outer_cleanup_failure_preserves_original_directory_and_reports_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.mkdir()
        (tokenizer_path / "original.bin").write_bytes(b"directory bytes")
        monkeypatch.setattr(
            engine_builder,
            "_generate_standard_tokenizer_json_transactionally",
            lambda *args, **kwargs: "conversion unavailable",
        )

        class FailingPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                (Path(model_dir) / "tokenizer.json").write_text(
                    '{"malformed":"failed family candidate"}',
                    encoding="utf-8",
                )
                return False

        monkeypatch.setattr(
            engine_builder,
            "_remove_tokenizer_candidate",
            lambda _path: (_ for _ in ()).throw(
                OSError("deterministic cleanup failure")
            ),
        )

        with pytest.raises(
            RuntimeError,
            match="original tokenizer.json is preserved at",
        ):
            _ensure_tokenizer_json(tmp_path, plugin=FailingPlugin())

        recovery_paths = list(
            tmp_path.glob(
                ".trtmc-required-tokenizer-recovery-*/"
                "original-tokenizer.json"
            )
        )
        assert len(recovery_paths) == 1
        assert stat.S_ISDIR(recovery_paths[0].lstat().st_mode)
        assert (
            recovery_paths[0] / "original.bin"
        ).read_bytes() == b"directory bytes"
        assert tokenizer_path.read_text() == (
            '{"malformed":"failed family candidate"}'
        )

    def test_quarantine_move_failure_keeps_original_at_canonical_path(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"canonical bytes survive"}'
        tokenizer_path.write_bytes(original)
        real_replace = engine_builder.os.replace

        def fail_quarantine_move(source, destination):
            if (
                Path(source) == tokenizer_path
                and ".trtmc-required-tokenizer-recovery-"
                in str(destination)
            ):
                raise OSError("deterministic quarantine move failure")
            return real_replace(source, destination)

        monkeypatch.setattr(
            engine_builder.os,
            "replace",
            fail_quarantine_move,
        )

        with pytest.raises(
            RuntimeError,
            match="original remains at its canonical path",
        ):
            _ensure_tokenizer_json(tmp_path)

        assert tokenizer_path.read_bytes() == original
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_family_hook_cannot_approve_incompatible_tokenizer(
        self,
        tmp_path,
        monkeypatch,
    ):
        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        class InvalidPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                (Path(model_dir) / "tokenizer.json").write_text("{}")
                return True

        with pytest.raises(RuntimeError, match="native-compatible tokenizer.json"):
            _ensure_tokenizer_json(tmp_path, plugin=InvalidPlugin())

        assert not (tmp_path / "tokenizer.json").exists()

    @pytest.mark.parametrize(
        "symlink_kind",
        ("relative", "absolute"),
    )
    def test_family_symlink_output_is_rejected_and_original_is_restored(
        self,
        tmp_path,
        monkeypatch,
        symlink_kind,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"restore after family symlink"}'
        tokenizer_path.write_bytes(original)
        source_path = tmp_path / "preexisting-tokenizer-source.json"
        source_payload = _native_tokenizer_payload("BPE")
        source_path.write_text(
            json.dumps(source_payload),
            encoding="utf-8",
        )

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        class SymlinkPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                assert not tokenizer_path.exists()
                link_target = (
                    source_path.name
                    if symlink_kind == "relative"
                    else source_path.resolve()
                )
                tokenizer_path.symlink_to(link_target)
                return True

        with pytest.raises(RuntimeError, match="regular, non-symlink file"):
            _ensure_tokenizer_json(tmp_path, plugin=SymlinkPlugin())

        assert tokenizer_path.read_bytes() == original
        assert not tokenizer_path.is_symlink()
        assert json.loads(source_path.read_text()) == source_payload
        assert not list(tmp_path.glob(".trtmc-*"))

    @pytest.mark.parametrize(
        "failure_mode",
        ("raises", "reports-failure", "invalid-success", "directory-output"),
    )
    def test_family_failure_restores_quarantined_tokenizer_bytes(
        self,
        tmp_path,
        monkeypatch,
        failure_mode,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        original = b'{"malformed":"preserve these exact bytes"}'
        tokenizer_path.write_bytes(original)

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        class FailingPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                assert not tokenizer_path.exists()
                if failure_mode == "directory-output":
                    tokenizer_path.mkdir()
                    (tokenizer_path / "partial.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                else:
                    tokenizer_path.write_text(
                        '{"malformed":"partial family output"}',
                        encoding="utf-8",
                    )
                if failure_mode == "raises":
                    raise RuntimeError("family conversion failed after writing")
                return failure_mode in {"invalid-success", "directory-output"}

        with pytest.raises(RuntimeError, match="refusing to write a bundle"):
            _ensure_tokenizer_json(tmp_path, plugin=FailingPlugin())

        assert tokenizer_path.read_bytes() == original
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_failed_repair_restores_dangling_tokenizer_symlink(
        self,
        tmp_path,
        monkeypatch,
    ):
        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.symlink_to("missing-original-tokenizer.json")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                assert not tokenizer_path.is_symlink()
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        class FailingPlugin:
            @staticmethod
            def ensure_tokenizer_json(model_dir, **kwargs):
                assert not tokenizer_path.exists()
                assert not tokenizer_path.is_symlink()
                return False

        with pytest.raises(RuntimeError, match="refusing to write a bundle"):
            _ensure_tokenizer_json(tmp_path, plugin=FailingPlugin())

        assert tokenizer_path.is_symlink()
        assert tokenizer_path.readlink() == Path(
            "missing-original-tokenizer.json"
        )
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_rejected_file_does_not_shortcut_t5_unigram_fallback(
        self,
        tmp_path,
        monkeypatch,
    ):
        from tensorrt_model_connect.families.t5.tokenizer_json import (
            ensure_tokenizer_json as ensure_t5_tokenizer_json,
        )

        tokenizer_path = tmp_path / "tokenizer.json"
        tokenizer_path.write_bytes(b'{"malformed":"quarantine before T5 hook"}')
        sentencepiece_path = tmp_path / "spiece.model"
        sentencepiece_path.write_bytes(b"fake sentencepiece model")
        calls = {}

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                assert not tokenizer_path.exists()
                raise RuntimeError("standard slow conversion unavailable")

        class FakeSentencePieceProcessor:
            _pieces = ("<unk>", "\u2581hello")
            _scores = (-1.0, -2.0)

            def Load(self, path):
                assert not tokenizer_path.exists()
                calls["sentencepiece_path"] = Path(path)

            def GetPieceSize(self):
                return len(self._pieces)

            def IdToPiece(self, index):
                return self._pieces[index]

            def GetScore(self, index):
                return self._scores[index]

        class FakeUnigram:
            def __init__(self, vocab, unk_id):
                calls["vocab"] = list(vocab)
                calls["unk_id"] = unk_id

        class FakeTokenizer:
            def __init__(self, model):
                self.model = model

            def save(self, path):
                Path(path).write_text(
                    json.dumps({
                        "model": {
                            "type": "Unigram",
                            "vocab": calls["vocab"],
                            "unk_id": calls["unk_id"],
                        },
                    }),
                    encoding="utf-8",
                )

        tokenizers_module = types.ModuleType("tokenizers")
        tokenizers_models_module = types.ModuleType("tokenizers.models")
        tokenizers_models_module.Unigram = FakeUnigram
        tokenizers_module.Tokenizer = FakeTokenizer
        tokenizers_module.models = tokenizers_models_module
        tokenizers_module.decoders = types.SimpleNamespace(
            Metaspace=lambda: object(),
        )
        tokenizers_module.normalizers = types.SimpleNamespace(
            Sequence=lambda items: tuple(items),
            Prepend=lambda **kwargs: kwargs,
            Replace=lambda *args: args,
        )
        tokenizers_module.pre_tokenizers = types.SimpleNamespace(
            Sequence=lambda items: tuple(items),
        )

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )
        monkeypatch.setitem(
            sys.modules,
            "sentencepiece",
            types.SimpleNamespace(
                SentencePieceProcessor=FakeSentencePieceProcessor,
            ),
        )
        monkeypatch.setitem(sys.modules, "tokenizers", tokenizers_module)
        monkeypatch.setitem(
            sys.modules,
            "tokenizers.models",
            tokenizers_models_module,
        )

        _ensure_tokenizer_json(
            tmp_path,
            plugin=types.SimpleNamespace(
                ensure_tokenizer_json=ensure_t5_tokenizer_json,
            ),
        )

        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        assert tokenizer["model"]["type"] == "Unigram"
        assert calls["sentencepiece_path"] == sentencepiece_path
        assert calls["vocab"] == [("<unk>", -1.0), ("\u2581hello", -2.0)]
        assert calls["unk_id"] == 0
        assert not list(tmp_path.glob(".trtmc-*"))

    def test_invalid_existing_tokenizer_fails_closed_when_rebuild_is_unavailable(
        self,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / "tokenizer.json").write_text("{}")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("slow tokenizer unavailable")

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
        )

        with pytest.raises(RuntimeError, match="refusing to write a bundle"):
            _ensure_tokenizer_json(tmp_path)

    def test_diffusion_plugins_validate_existing_tokenizers_through_callback(self):
        repo_root = Path(__file__).resolve().parents[2]
        plugin_paths = (
            "python/tensorrt_model_connect/families/flux/plugin.py",
            "python/tensorrt_model_connect/families/ltx_video/plugin.py",
            "python/tensorrt_model_connect/families/pixart/plugin.py",
            "python/tensorrt_model_connect/families/qwen_image/plugin.py",
            "python/tensorrt_model_connect/families/wan_t2v/plugin.py",
            "python/tensorrt_model_connect/families/z_image/plugin.py",
            (
                "python/tensorrt_model_connect/families/sana_wm/"
                "components/ltx_video/plugin.py"
            ),
        )

        for relative in plugin_paths:
            source = (repo_root / relative).read_text(encoding="utf-8")
            assert "ensure_tokenizer_json(tokenizer_dir)" in source
            assert 'if not (tokenizer_dir / "tokenizer.json").exists()' not in source


class TestBuildBundleErrors:
    """Test build_bundle error handling."""

    @pytest.mark.parametrize(
        "invalid_trust",
        ("false", 1, None),
        ids=("string-false", "integer-one", "none"),
    )
    def test_rejects_non_boolean_remote_code_trust_before_setup(
        self,
        monkeypatch,
        invalid_trust,
    ):
        setup_trt = Mock()
        monkeypatch.setattr(engine_builder, "_setup_trt_import", setup_trt)

        with pytest.raises(TypeError, match="trust_remote_code must be a bool"):
            build_bundle(
                "unused-model-dir",
                "unused.trtfb",
                trust_remote_code=invalid_trust,
            )

        setup_trt.assert_not_called()

    def test_family_without_required_tokenizer_still_builds(
        self,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "image_family",
            "architectures": ["ImageFamilyModel"],
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
        }))

        class ImagePlugin:
            name = "image_family"
            runtime_strategy = "image_family_runtime"
            runtime_capabilities = set()
            requires_tokenizer = False

            @staticmethod
            def load_weights(model_dir, config, *, precision="fp32"):
                return {}

            @staticmethod
            def build_engine(
                config,
                weights,
                max_cache_length,
                *,
                precision="fp32",
                verbose=False,
            ):
                return b"PLAN"

        ensure_tokenizer = Mock(
            side_effect=AssertionError("tokenizer generation must be skipped")
        )
        write_bundle_mock = Mock()
        monkeypatch.setattr(
            engine_builder,
            "_apply_family_builder_capabilities",
            lambda _config: None,
        )
        monkeypatch.setattr(
            engine_builder,
            "find_plugin",
            lambda _config: ImagePlugin(),
        )
        monkeypatch.setattr(
            engine_builder,
            "_ensure_tokenizer_json",
            ensure_tokenizer,
        )
        monkeypatch.setattr(
            engine_builder,
            "_detect_tokenizer_special_frame",
            lambda *_args, **_kwargs: ([], []),
        )
        monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.0")
        monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "")
        monkeypatch.setattr(engine_builder, "write_bundle", write_bundle_mock)
        monkeypatch.setattr(
            engine_builder.trt_compat,
            "resolved_summary",
            lambda: "test TensorRT",
        )

        build_bundle(
            str(tmp_path),
            str(tmp_path / "out.trtfb"),
        )

        ensure_tokenizer.assert_not_called()
        write_bundle_mock.assert_called_once()

    def test_diffusion_repairs_tokenizer_before_reconciling_prebuilt_config(
        self,
        tmp_path,
        monkeypatch,
    ):
        (tmp_path / "model_index.json").write_text(json.dumps({
            "_class_name": "SyntheticDiffusionPipeline",
        }))
        (tmp_path / "tokenizer.json").write_text("{}")
        events = []

        class RepairedTokenizer:
            @staticmethod
            def save_pretrained(path):
                (Path(path) / "tokenizer.json").write_text(json.dumps({
                    "model": {
                        "type": "BPE",
                        "vocab": {"a": 0},
                        "merges": [],
                    },
                }))

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                assert Path(path) == tmp_path
                assert kwargs == {
                    "use_fast": False,
                    "trust_remote_code": False,
                }
                return RepairedTokenizer()

        class DiffusionPlugin:
            name = "synthetic_diffusion"
            runtime_strategy = "diffusion_synthetic"

            @staticmethod
            def load_weights(model_dir, config):
                return {}

            @staticmethod
            def build_components(
                model_dir,
                config,
                weights,
                *,
                verbose=False,
                fp8_scales=None,
            ):
                return {
                    "config_json": json.dumps({
                        "runtime_strategy": "diffusion_synthetic",
                        "tokenizer": {"add_special_tokens": False},
                    }).encode("utf-8"),
                }

            @staticmethod
            def diffusion_bundle_sections(components, **kwargs):
                return []

            @staticmethod
            def diffusion_tokenizer_bundle_sections(
                model_dir_path,
                *,
                ensure_tokenizer_json,
            ):
                events.append("tokenizer_sections")
                ensure_tokenizer_json(Path(model_dir_path))
                tokenizer_path = Path(model_dir_path) / "tokenizer.json"
                return [("tokenizer.json", tokenizer_path.read_bytes())]

            @staticmethod
            def diffusion_tokenizer_special_frame(
                model_dir_path,
                *,
                detect_tokenizer_special_frame,
            ):
                events.append("special_frame")
                tokenizer = json.loads(
                    (Path(model_dir_path) / "tokenizer.json").read_text()
                )
                assert tokenizer["model"]["type"] == "BPE"
                return [11], [12]

            @staticmethod
            def diffusion_bundle_config(config, *, components):
                raise AssertionError(
                    "pre-rendered config_json must stay on its dedicated path"
                )

        write_bundle_mock = Mock()
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(AutoTokenizer=AutoTokenizer),
        )
        monkeypatch.setattr(
            engine_builder,
            "find_diffusion_plugin",
            lambda _pipeline_class: DiffusionPlugin(),
        )
        monkeypatch.setattr(
            engine_builder.trt_compat,
            "resolved_summary",
            lambda: "test TensorRT",
        )
        monkeypatch.setattr(engine_builder, "_get_trt_version", lambda: "10.0")
        monkeypatch.setattr(engine_builder, "_get_gpu_name", lambda: "")
        monkeypatch.setattr(engine_builder, "write_bundle", write_bundle_mock)

        build_bundle(
            str(tmp_path),
            str(tmp_path / "out.trtfb"),
        )

        info = write_bundle_mock.call_args.args[1]
        sections = write_bundle_mock.call_args.args[2]
        section_map = {section.name: section.data for section in sections}
        config = json.loads(section_map["config.json"])
        assert events == ["tokenizer_sections", "special_frame"]
        assert json.loads(section_map["tokenizer.json"])["model"]["type"] == "BPE"
        assert config["tokenizer_add_special_tokens"] == 1
        assert config["tokenizer_special_prefix_ids"] == [11]
        assert config["tokenizer_special_suffix_ids"] == [12]
        assert config["tokenizer"]["add_special_tokens"] is True
        assert info.tokenizer_add_special_tokens is True

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
