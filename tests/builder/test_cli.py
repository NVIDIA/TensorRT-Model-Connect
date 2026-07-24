# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_cli.py — argument parsing edge cases.

Pure Python, no TRT needed. Tests the CLI argument parser without
actually invoking engine builds.

Trace: ARCH-ENG-001, UD-ENG-01
Intent: Validate that the trtmc CLI correctly parses build/inspect/version subcommands and their arguments.
Preconditions: tensorrt_model_connect.build_cli is importable; no TRT or GPU required.
Postconditions: Parsed arguments match expected values for all subcommands, defaults, and edge cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest



class TestBuildArgs:
    def test_build_with_hidden_legacy_args(self, monkeypatch):
        """The real parser retains explicit legacy profile compatibility."""
        import tensorrt_model_connect.build_cli as cli

        captured: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(
            cli,
            "_cmd_build",
            lambda args: captured.setdefault("args", args) and 0,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "trtmc",
                "build",
                "example-org/example-model",
                "-o",
                "/tmp/out.trtfb",
                "--max-cache-length",
                "512",
                "--verbose",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        args = captured["args"]
        assert args.command == "build"
        assert args.model == "example-org/example-model"
        assert args.output == "/tmp/out.trtfb"
        assert args.max_cache_length == 512
        assert args.verbose is True

    def test_model_only_build_parser_uses_automatic_policy(self, monkeypatch):
        """The real parser leaves output and KV policy for automatic resolution."""
        import tensorrt_model_connect.build_cli as cli

        captured: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(
            cli,
            "_cmd_build",
            lambda args: captured.setdefault("args", args) and 0,
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["trtmc", "build", "example-org/long-context-model"],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        args = captured["args"]
        assert args.output is None
        assert args.max_cache_length is None
        assert args.dynamic_kv_cache is None
        assert args.dynamic_kv_profile_rows is None
        assert args.decoder_engine_layout == "split"

    def test_build_model_only_command_does_not_require_output(self, monkeypatch):
        """The real parser accepts a build command without -o."""
        import tensorrt_model_connect.build_cli as cli

        captured: dict[str, argparse.Namespace] = {}
        monkeypatch.setattr(
            cli,
            "_cmd_build",
            lambda args: captured.setdefault("args", args) and 0,
        )
        monkeypatch.setattr(sys, "argv", ["trtmc", "build", "model-dir"])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert captured["args"].output is None

    def test_parse_dynamic_kv_profile_rows(self):
        """Comma-separated dynamic-KV profile rows parse into integer lists."""
        from tensorrt_model_connect.build_cli import _parse_profile_rows

        assert _parse_profile_rows("32,64,128") == [32, 64, 128]
        assert _parse_profile_rows(" 32, 64 ,128 ") == [32, 64, 128]

    def test_parse_dynamic_kv_profile_rows_rejects_empty(self):
        """Empty profile-row strings are rejected with a parser-style error."""
        from tensorrt_model_connect.build_cli import _parse_profile_rows

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_profile_rows(" , ")

    def test_parse_fp32_layers(self):
        from tensorrt_model_connect.build_cli import _parse_layer_indices

        assert _parse_layer_indices("2") == [2]
        assert _parse_layer_indices("1, 3") == [1, 3]

    def test_parse_fp32_layers_rejects_negative_indices(self):
        import argparse
        from tensorrt_model_connect.build_cli import _parse_layer_indices

        with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
            _parse_layer_indices("-1")


class TestMainParser:
    def test_build_help_hides_legacy_method_selector(self, monkeypatch, capsys):
        import tensorrt_model_connect.build_cli as cli

        monkeypatch.setattr(sys, "argv", ["trtmc", "build", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 0
        assert "--method" not in capsys.readouterr().out

    def test_build_accepts_trust_remote_code(self, monkeypatch, tmp_path):
        """The real build parser accepts E2E manifest trust-remote-code commands."""
        import tensorrt_model_connect.build_cli as cli

        captured: dict[str, argparse.Namespace] = {}

        def _fake_cmd_build(args):
            captured["args"] = args
            return 0

        monkeypatch.setattr(cli, "_cmd_build", _fake_cmd_build)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "trtmc",
                "build",
                "example-org/remote-code-model",
                "-o",
                str(tmp_path / "out.trtfb"),
                "--max-cache-length",
                "256",
                "--trust-remote-code",
                "--build-timing-json",
                str(tmp_path / "timing.json"),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert captured["args"].trust_remote_code is True
        assert captured["args"].build_timing_json == str(tmp_path / "timing.json")
        assert not hasattr(captured["args"], "_explicit_public_options")


class TestInspectArgs:
    def test_inspect_parses(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        inspect_p = subparsers.add_parser("inspect")
        inspect_p.add_argument("bundle_path")

        args = parser.parse_args(["inspect", "/path/to/bundle.trtfb"])
        assert args.command == "inspect"
        assert args.bundle_path == "/path/to/bundle.trtfb"


class TestVersionCommand:
    def test_version_exits_zero(self):
        """trtmc version should return 0."""
        from tensorrt_model_connect.build_cli import _cmd_version
        result = _cmd_version(argparse.Namespace())
        assert result == 0


class TestCmdBuildValidation:
    def test_missing_model(self):
        """_cmd_build returns 1 when model is empty."""
        from tensorrt_model_connect.build_cli import _cmd_build
        args = argparse.Namespace(model="", output="out.trtfb", quantize=None, quant_scales=None, quant_calibration_samples=512,
                                  max_cache_length=256, verbose=False, _skip_profile_resolution=True)
        result = _cmd_build(args)
        assert result == 1

    def test_missing_output_is_derived(self, monkeypatch, capsys):
        """_cmd_build derives the bundle name before dispatch."""
        import tensorrt_model_connect.build_cli as cli
        import tensorrt_model_connect.engine_builder as engine_builder

        captured: dict[str, str] = {}

        def _optimized(_model, output, _options, **_kwargs):
            captured["output"] = output
            return object()

        monkeypatch.setattr(cli, "_model_only_native_dynamic_family", lambda _args: None)
        monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", _optimized)
        args = argparse.Namespace(
            model="example-org/Some-Model",
            output="",
            max_cache_length=None,
            dynamic_kv_cache=None,
            dynamic_kv_profile_rows=None,
            decoder_engine_layout="split",
            tensor_parallel_size=1,
            quantize=None,
            quant_scales=None,
            quant_calibration_samples=512,
            verbose=False,
            _skip_profile_resolution=True,
        )

        assert cli._cmd_build(args) == 0
        assert args.output == "some-model.trtfb"
        assert captured["output"] == "some-model.trtfb"
        assert "Output: some-model.trtfb (derived from model)" in capsys.readouterr().err


class TestCmdInspect:
    """Tests for _cmd_inspect with real and invalid bundle files."""

    def test_inspect_valid_bundle(self, tmp_path, capsys):
        """Inspect a real (synthetic) .trtfb bundle and verify output."""
        import json
        import struct

        bundle_path = tmp_path / "test.trtfb"
        header = {
            "model_id": "test-model",
            "model_type": "example_decoder",
            "family": "example_family",
            "trt_version": "10.0.0",
            "gpu_name": "A100",
            "created_at": "2025-01-01T00:00:00Z",
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_cache_length": 256,
            "sections": {
                "engine_plan": {"offset": 0, "size": 100},
            },
        }
        header_json = json.dumps(header).encode("utf-8")

        with open(bundle_path, "wb") as f:
            f.write(b"TRTFB\x00\x01\x00")
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 100)  # fake engine plan

        from tensorrt_model_connect.build_cli import _cmd_inspect
        result = _cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path)))
        assert result == 0

        captured = capsys.readouterr()
        assert "example_decoder" in captured.out
        assert "test-model" in captured.out
        assert "example_family" in captured.out
        assert "engine_plan" in captured.out

    @pytest.mark.dynamic_memory
    def test_inspect_runtime_memory_bundle_reports_only_static_contract(
        self, monkeypatch, tmp_path, capsys
    ):
        import ctypes
        import json
        import struct

        import tensorrt_model_connect.trt_compat as trt_compat

        bundle_path = tmp_path / "dynamic.trtfb"
        header = {
            "model_id": "Qwen/Qwen3-0.6B",
            "model_type": "qwen3",
            "family": "qwen",
            "max_cache_length": 40960,
            "runtime_memory": {
                "contract_version": 1,
                "qualified_model_id": "Qwen/Qwen3-0.6B",
                "qualified_model_revision": "a" * 40,
                "qualified_config_sha256": "b" * 64,
                "qualified_target": "gb300-trt-11.2",
                "qualified_runtime_stack": {
                    "sm": "sm103",
                    "tensorrt": "11.2.0.113",
                    "cuda_runtime": "13.3",
                    "cudnn_backend": "9.20.0",
                    "cudnn_frontend_revision": "c" * 40,
                    "nvrtc": "13.3",
                    "driver": "580.105.08",
                },
                "native_kv_plugin_abi": 2,
                "model_context_limit": 40960,
                "prefill_chunk_limit": 1024,
                "kv_layout": "contiguous_runtime_v1",
                "kv_dtype": "bfloat16",
                "kv_bytes_per_token": 114688,
                "active_kv_profile_limits": [
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    8192,
                    32768,
                    40960,
                ],
                "runtime_owned": True,
            },
            "sections": {},
        }
        payload = json.dumps(header).encode("utf-8")
        bundle_path.write_bytes(b"TRTFB\x00\x01\x00" + struct.pack("<Q", len(payload)) + payload)

        from tensorrt_model_connect.build_cli import _cmd_inspect

        def unexpected_runtime_touch(*_args, **_kwargs):
            pytest.fail("static bundle inspection must not initialize CUDA or load a backend")

        monkeypatch.setattr(trt_compat, "load_module", unexpected_runtime_touch)
        monkeypatch.setattr(ctypes, "CDLL", unexpected_runtime_touch)
        monkeypatch.setattr(subprocess, "run", unexpected_runtime_touch)
        monkeypatch.setattr(subprocess, "Popen", unexpected_runtime_touch)
        monkeypatch.setitem(sys.modules, "tensorrt", None)
        monkeypatch.setitem(sys.modules, "tensorrt_rtx", None)
        monkeypatch.setitem(sys.modules, "cuda", None)

        assert _cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path))) == 0
        output = capsys.readouterr().out
        expected_static_fields = {
            "Runtime KV contract version": "1",
            "Qualified model ID": "Qwen/Qwen3-0.6B",
            "Qualified model revision": "a" * 40,
            "Qualified config fingerprint": "b" * 64,
            "Model context limit": "40960",
            "Prefill chunk limit": "1024",
            "KV layout": "contiguous_runtime_v1",
            "KV bytes per token": "114688",
        }
        for label, value in expected_static_fields.items():
            assert f"{label + ':':<32} {value}" in output
        assert (
            f"{'Active KV profile limits:':<32} 128, 256, 512, 1024, 2048, 8192, 32768, 40960"
        ) in output
        assert "Max cache length:" not in output
        assert "runtime_kv_capacity_tokens" not in output
        assert "post_load_free_bytes" not in output

    def test_list_engine_sections_marks_split_decoder_roles(self, tmp_path):
        """Split decoder bundles should label decode and prefill plans distinctly."""
        import json
        import struct

        bundle_path = tmp_path / "split.trtfb"
        header = {
            "model_id": "test-model",
            "model_type": "example_decoder",
            "family": "example_family",
            "trt_version": "10.0.0",
            "gpu_name": "A100",
            "created_at": "2025-01-01T00:00:00Z",
            "vocab_size": 32000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_cache_length": 256,
            "sections": {
                "engine_plan": {"offset": 0, "size": 100},
                "prefill_engine_plan": {"offset": 100, "size": 200},
            },
        }
        header_json = json.dumps(header).encode("utf-8")

        with open(bundle_path, "wb") as f:
            f.write(b"TRTFB\x00\x01\x00")
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(b"\x00" * 300)

        from tensorrt_model_connect.build_cli import list_engine_sections

        roles = {entry["name"]: entry["role"] for entry in list_engine_sections(str(bundle_path))}
        assert roles["engine_plan"] == "decode"
        assert roles["prefill_engine_plan"] == "prefill"

    def test_inspect_nonexistent_file(self):
        """_cmd_inspect returns 1 for non-existent file."""
        from tensorrt_model_connect.build_cli import _cmd_inspect
        result = _cmd_inspect(argparse.Namespace(
            bundle_path="/nonexistent/path/bundle.trtfb"))
        assert result == 1

    def test_inspect_invalid_magic(self, tmp_path):
        """_cmd_inspect returns 1 for file with wrong magic bytes."""
        bundle_path = tmp_path / "bad.trtfb"
        bundle_path.write_bytes(b"NOT_TRTFB_MAGIC_1234567890")

        from tensorrt_model_connect.build_cli import _cmd_inspect
        result = _cmd_inspect(argparse.Namespace(
            bundle_path=str(bundle_path)))
        assert result == 1

    def test_inspect_empty_bundle_path(self):
        """_cmd_inspect returns 1 when bundle_path is empty."""
        from tensorrt_model_connect.build_cli import _cmd_inspect
        result = _cmd_inspect(argparse.Namespace(bundle_path=""))
        assert result == 1


class TestCmdBuildMocked:
    """Tests for _cmd_build with mocked engine_builder."""

    def test_build_calls_engine_builder_with_correct_args(self, tmp_path):
        """Verify _cmd_build passes model, output, cache length, verbose to build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        captured_kwargs = {}

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       precision="fp32", quantize=None, quant_scales=None, quant_calibration_samples=512, verbose=False, **kwargs):
            captured_kwargs["model_id_or_path"] = model_id_or_path
            captured_kwargs["output_path"] = output_path
            captured_kwargs["max_cache_length"] = max_cache_length
            captured_kwargs["precision"] = precision
            captured_kwargs["verbose"] = verbose

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="/path/to/model",
                output=str(tmp_path / "out.trtfb"),
                max_cache_length=512,
                precision="fp32",
                method="trt",
                quantize=None, quant_scales=None, quant_calibration_samples=512,
                verbose=True,
                _skip_profile_resolution=True,
            )
            with patch(
                "tensorrt_model_connect.runtime_provider.orchestrator.try_build_optimized_runtime",
                return_value=None,
            ):
                result = _cmd_build(args)
            assert result == 0
            assert captured_kwargs["model_id_or_path"] == "/path/to/model"
            assert captured_kwargs["output_path"] == str(
                tmp_path / "out.trtfb")
            assert captured_kwargs["max_cache_length"] == 512
            assert captured_kwargs["verbose"] is True
        finally:
            eb._build_native_impl = original_build

    def test_verbose_flag_propagated(self, tmp_path):
        """Verify verbose=True is forwarded to engine_builder.build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        received_verbose = []

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       precision="fp32", quantize=None, quant_scales=None, quant_calibration_samples=512, verbose=False, **kwargs):
            received_verbose.append(verbose)

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="some-model", output=str(tmp_path / "out.trtfb"),
                max_cache_length=256, precision="fp32", method="trt", quantize=None, quant_scales=None, quant_calibration_samples=512, verbose=True, _skip_profile_resolution=True)
            _cmd_build(args)
            assert received_verbose == [True]

            received_verbose.clear()
            args = argparse.Namespace(
                model="some-model", output=str(tmp_path / "out.trtfb"),
                max_cache_length=256, precision="fp32", method="trt", quantize=None, quant_scales=None, quant_calibration_samples=512, verbose=False, _skip_profile_resolution=True)
            _cmd_build(args)
            assert received_verbose == [False]
        finally:
            eb._build_native_impl = original_build

    def test_max_cache_length_propagated(self, tmp_path):
        """Verify max_cache_length value is forwarded to engine_builder.build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        received_cache = []

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       precision="fp32", quantize=None, quant_scales=None, quant_calibration_samples=512, verbose=False, **kwargs):
            received_cache.append(max_cache_length)

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            for cache_len in [128, 1024, 4096]:
                args = argparse.Namespace(
                    model="some-model",
                    output=str(tmp_path / "out.trtfb"),
                    max_cache_length=cache_len,
                    precision="fp32",
                    method="trt",
                    quantize=None, quant_scales=None,
                    quant_calibration_samples=512,
                    verbose=False,
                    _skip_profile_resolution=True)
                _cmd_build(args)
            assert received_cache == [128, 1024, 4096]
        finally:
            eb._build_native_impl = original_build

    def test_dynamic_kv_cache_propagated(self, tmp_path):
        """Verify dynamic_kv_cache is forwarded to engine_builder.build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        received = []

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       dynamic_kv_cache=False, precision="fp32", quantize=None,
                       quant_scales=None, quant_calibration_samples=512,
                       verbose=False, **kwargs):
            received.append(dynamic_kv_cache)

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="some-model",
                output=str(tmp_path / "out.trtfb"),
                max_cache_length=256,
                dynamic_kv_cache=True,
                precision="fp32",
                quantize=None,
                quant_scales=None,
                quant_calibration_samples=512,
                verbose=False,
                method="trt",
                _skip_profile_resolution=True,
            )
            _cmd_build(args)
            assert received == [True]
        finally:
            eb._build_native_impl = original_build

    def test_dynamic_kv_profile_rows_propagated(self, tmp_path):
        """Verify explicit dynamic-KV profile rows are forwarded to build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        received = []

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       dynamic_kv_profile_rows_override=None, **kwargs):
            received.append(dynamic_kv_profile_rows_override)

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="some-model",
                output=str(tmp_path / "out.trtfb"),
                max_cache_length=256,
                dynamic_kv_cache=True,
                dynamic_kv_profile_rows=[32, 64, 128],
                precision="fp32",
                quantize=None,
                quant_scales=None,
                quant_calibration_samples=512,
                verbose=False,
                fp8=False,
                fp8_scales=None,
                save_fp8_scales=None,
                triattention_stats=None,
                triattention_kv_budget=None,
                triattention_divide_length=128,
                triattention_recent_window=128,
                triattention_score_aggregation="mean",
                triattention_count_prompt_tokens=True,
                triattention_protect_prefill=True,
                triattention_disable_mlr=False,
                triattention_disable_trig=False,
                method="trt",
                _skip_profile_resolution=True,
            )
            _cmd_build(args)
            assert received == [[32, 64, 128]]
        finally:
            eb._build_native_impl = original_build

    def test_decoder_engine_layout_propagated(self, tmp_path):
        """Verify --decoder-engine-layout is forwarded to engine_builder.build()."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        received = []

        def mock_build(model_id_or_path, output_path, max_cache_length, *,
                       decoder_engine_layout="split", **kwargs):
            received.append(decoder_engine_layout)

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="some-model",
                output=str(tmp_path / "out.trtfb"),
                max_cache_length=256,
                decoder_engine_layout="dual_profile",
                dynamic_kv_cache=False,
                dynamic_kv_profile_rows=None,
                precision="fp32",
                quantize=None,
                quant_scales=None,
                quant_calibration_samples=512,
                verbose=False,
                fp8=False,
                fp8_scales=None,
                save_fp8_scales=None,
                triattention_stats=None,
                triattention_kv_budget=None,
                triattention_divide_length=128,
                triattention_recent_window=128,
                triattention_score_aggregation="mean",
                triattention_count_prompt_tokens=True,
                triattention_protect_prefill=True,
                triattention_disable_mlr=False,
                triattention_disable_trig=False,
                method="trt",
                _skip_profile_resolution=True,
            )
            _cmd_build(args)
            assert received == ["dual_profile"]
        finally:
            eb._build_native_impl = original_build

    def test_build_exception_returns_1(self, tmp_path):
        """When engine_builder.build() raises, _cmd_build returns 1."""
        from tensorrt_model_connect.build_cli import _cmd_build
        import tensorrt_model_connect.engine_builder as eb

        def mock_build(*args, **kwargs):
            raise RuntimeError("TRT build failed: out of memory")

        original_build = eb._build_native_impl
        eb._build_native_impl = mock_build
        try:
            args = argparse.Namespace(
                model="some-model",
                output=str(tmp_path / "out.trtfb"),
                max_cache_length=256,
                precision="fp32",
                method="trt",
                verbose=False,
                _skip_profile_resolution=True)
            result = _cmd_build(args)
            assert result == 1
        finally:
            eb._build_native_impl = original_build

    def test_build_reexecs_into_declared_python_profile(self, monkeypatch, tmp_path):
        """Families with declared profiles should re-exec into that Python profile."""
        import tensorrt_model_connect.build_cli as cli
        import tensorrt_model_connect.python_profiles as profile_mod

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            cli,
            "_resolve_build_model_metadata",
            lambda model_ref, method_name: ("/tmp/resolved-model", "example_profile"),
        )
        monkeypatch.setattr(
            cli,
            "_resolve_build_profile_name",
            lambda family_name: "example_profile",
        )
        monkeypatch.setattr(
            profile_mod,
            "resolve_profile_python",
            lambda profile_name, base_python: "/tmp/example-profile/bin/python",
        )

        def _fake_run(cmd, env=None, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = env or {}
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(cli.subprocess, "run", _fake_run)

        args = argparse.Namespace(
            model="example-org/profiled-model",
            output=str(tmp_path / "out.trtfb"),
            max_cache_length=256,
            precision="fp32",
            quantize=None,
            quant_scales=None,
            quant_calibration_samples=512,
            verbose=False,
            method="trt",
            _skip_profile_resolution=False,
            active_python_profile="",
        )

        with patch.object(
            sys,
            "argv",
            [
                "trtmc",
                "build",
                "example-org/profiled-model",
                "-o",
                str(tmp_path / "out.trtfb"),
            ],
        ):
            assert cli._cmd_build(args) == 0

        command = captured["cmd"]
        assert command[0:2] == [
            "/tmp/example-profile/bin/python",
            "-c",
        ]
        assert "sys.path.append" in command[2]
        assert "runpy.run_module" in command[2]
        assert command[3:] == [
            str(Path(cli.__file__).resolve().parent.parent),
            "build",
            "example-org/profiled-model",
            "-o",
            str(tmp_path / "out.trtfb"),
            "--active-python-profile",
            "example_profile",
        ]
        assert all("ACTIVE_PYTHON_PROFILE" not in key for key in captured["env"])

    def test_profile_reexec_bootstrap_imports_explicit_package_root(self, tmp_path):
        """A profile can import TRTMC when a baked editable path is unavailable."""
        import tensorrt_model_connect.build_cli as cli

        package_root = tmp_path / "source"
        package = package_root / "tensorrt_model_connect"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "__main__.py").write_text(
            "import json, sys\nprint(json.dumps(sys.argv))\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                cli._PROFILE_REEXEC_BOOTSTRAP,
                str(package_root),
                "build",
                "example-org/profiled-model",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert json.loads(completed.stdout) == [
            str(package / "__main__.py"),
            "build",
            "example-org/profiled-model",
        ]


class TestFriendlyDownloadErrors:
    """Tests for _raise_friendly_download_error — clear messages for HF failures.

    Trace ID: UT-CLI-03 / ARCH-BUILD-001 / UD-BUILD-ERR-01
    Intent: Verify that HF download failures produce actionable error messages
    Preconditions: No network or HF access needed (mocked exceptions)
    Postconditions: Each HF error type maps to a clear, actionable RuntimeError
    """

    def test_repository_not_found(self):
        """RepositoryNotFoundError → tells user to check repo ID and login."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        class RepositoryNotFoundError(Exception):
            pass

        exc = RepositoryNotFoundError("404 Client Error")
        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            _raise_friendly_download_error("example-org/missing-model", exc)

    def test_gated_repo(self):
        """GatedRepoError → tells user to accept license and login."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        class GatedRepoError(Exception):
            pass

        exc = GatedRepoError("Access to model is restricted")
        with pytest.raises(RuntimeError, match="gated.*license"):
            _raise_friendly_download_error("gated-org/gated-model", exc)

    def test_connection_error(self):
        """ConnectionError → tells user to check network."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        exc = ConnectionError("Name resolution failed")
        with pytest.raises(RuntimeError, match="Network error"):
            _raise_friendly_download_error("example-org/example-model", exc)

    def test_entry_not_found(self):
        """EntryNotFoundError → tells user about missing files."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        class EntryNotFoundError(Exception):
            pass

        exc = EntryNotFoundError("config.json not found")
        with pytest.raises(RuntimeError, match="required files are missing"):
            _raise_friendly_download_error("some/model", exc)

    def test_generic_exception_includes_context(self):
        """Unknown exceptions → includes model ID and original message."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        exc = ValueError("something unexpected")
        with pytest.raises(RuntimeError, match="Failed to download.*something unexpected"):
            _raise_friendly_download_error("org/model", exc)

    def test_original_exception_chained(self):
        """All friendly errors chain the original exception via __cause__."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        class RepositoryNotFoundError(Exception):
            pass

        original = RepositoryNotFoundError("404")
        with pytest.raises(RuntimeError) as exc_info:
            _raise_friendly_download_error("org/model", original)
        assert exc_info.value.__cause__ is original

    def test_resolve_model_wraps_download_error(self, monkeypatch):
        """_resolve_model wraps snapshot_download failures with friendly messages."""
        from tensorrt_model_connect.engine_builder import _resolve_model

        class RepositoryNotFoundError(Exception):
            pass

        class _FakeHuggingFaceHub:
            @staticmethod
            def snapshot_download(*_args, **_kwargs):
                raise RepositoryNotFoundError("404")

        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHuggingFaceHub)

        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            _resolve_model("nonexistent/repo-id")

    def test_disk_error(self):
        """OSError with 'disk' in message → tells user to check disk space."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        exc = OSError("No space left on disk")
        with pytest.raises(RuntimeError, match="Disk error.*disk space"):
            _raise_friendly_download_error("org/model", exc)

    def test_http_error(self):
        """HTTPError → tells user about network issues."""
        from tensorrt_model_connect.engine_builder import _raise_friendly_download_error

        class HTTPError(Exception):
            pass

        exc = HTTPError("503 Service Unavailable")
        with pytest.raises(RuntimeError, match="Network error"):
            _raise_friendly_download_error("org/model", exc)
