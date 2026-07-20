# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-focused tests for CLI control flow in tensorrt_model_connect.build_cli.

Trace: ARCH-ENG-001, UD-ENG-02
Intent: Validate CLI control flow branches including version lookup fallbacks, inspect output formatting, build dispatch, and error handling.
Preconditions: tensorrt_model_connect is importable; uses mocks for TRT/GPU dependencies.
Postconditions: Version resolution follows the correct fallback chain, inspect prints expected output, and build errors propagate cleanly.
"""

from __future__ import annotations

import argparse
import builtins
import json
import struct
import sys
import types
from unittest.mock import patch

import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
import tensorrt_model_connect  # noqa: E402
import tensorrt_model_connect.build_cli as cli  # noqa: E402


def test_get_version_prefers_importlib_metadata():
    """Intent: exercise the fast-path version lookup.
    Preconditions: importlib.metadata.version returns a concrete package version string.
    Postconditions: _get_version returns that exact version string.
    """
    with patch("importlib.metadata.version", return_value="9.9.9"):
        assert cli._get_version() == "9.9.9"


def test_get_version_uses_package_fallback_when_metadata_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """Intent: validate fallback from metadata lookup to package __version__.
    Preconditions: importlib.metadata.version raises, and tensorrt_model_connect.__version__ is set.
    Postconditions: _get_version returns the package-level __version__ value.
    """
    monkeypatch.setattr(tensorrt_model_connect, "__version__", "7.8.9", raising=False)
    with patch("importlib.metadata.version", side_effect=RuntimeError("boom")):
        assert cli._get_version() == "7.8.9"


def test_get_version_uses_literal_default_when_relative_import_fails():
    """Intent: cover the final fallback branch in _get_version.
    Preconditions: metadata lookup raises, and relative import of __version__ raises ImportError.
    Postconditions: _get_version returns the hardcoded literal default.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        package = (globals or {}).get("__package__")
        if "__version__" in (fromlist or ()) and (
            package == "tensorrt_model_connect" or level == 1 or name in ("", "tensorrt_model_connect")
        ):
            raise ImportError("synthetic import failure for __version__")
        return real_import(name, globals, locals, fromlist, level)

    with patch("importlib.metadata.version", side_effect=RuntimeError("boom")):
        with patch("builtins.__import__", side_effect=fake_import):
            assert cli._get_version() == "0.1.0"


def test_cmd_inspect_valid_bundle_without_sections(tmp_path, capsys):
    """Intent: cover inspect output when sections metadata is absent.
    Preconditions: a syntactically valid bundle has required header fields but no "sections" key.
    Postconditions: _cmd_inspect succeeds and does not print a "Sections:" block.
    """
    bundle_path = tmp_path / "minimal.trtfb"
    header = {
        "model_id": "minimal-model",
        "model_type": "example_decoder",
        "family": "example_family",
    }
    payload = json.dumps(header).encode("utf-8")
    with open(bundle_path, "wb") as f:
        f.write(b"TRTFB\x00\x01\x00")
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)

    result = cli._cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path)))
    captured = capsys.readouterr()

    assert result == 0
    assert "Model ID:" in captured.out
    assert "Sections:" not in captured.out


def test_cmd_inspect_returns_error_for_malformed_header_json(tmp_path, capsys):
    """Intent: exercise inspect error handling after header decode.
    Preconditions: bundle magic and header length are valid but header JSON is malformed.
    Postconditions: _cmd_inspect returns non-zero and emits an error to stderr.
    """
    bundle_path = tmp_path / "malformed.trtfb"
    malformed = b'{"model_id": "bad-json"'
    with open(bundle_path, "wb") as f:
        f.write(b"TRTFB\x00\x01\x00")
        f.write(struct.pack("<Q", len(malformed)))
        f.write(malformed)

    result = cli._cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path)))
    captured = capsys.readouterr()

    assert result == 1
    assert "Error:" in captured.err


def test_cmd_version_prints_installed_tensorrt_version(monkeypatch, capsys):
    """Intent: cover TensorRT-installed reporting path in _cmd_version.
    Preconditions: a synthetic tensorrt module exists in sys.modules with __version__.
    Postconditions: _cmd_version prints the synthetic TensorRT version and returns success.
    """
    fake_trt = types.ModuleType("tensorrt")
    fake_trt.__version__ = "99.1.2"
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    result = cli._cmd_version(argparse.Namespace())
    captured = capsys.readouterr()

    assert result == 0
    assert "TensorRT:  99.1.2" in captured.out


def test_main_implicit_build_dispatches_to_build_handler(monkeypatch):
    """Intent: verify bare positional args are rewritten to the build subcommand.
    Preconditions: argv omits an explicit subcommand and contains model/output args.
    Postconditions: main dispatches to _cmd_build with parsed build arguments and exits with its code.
    """
    captured: dict[str, argparse.Namespace] = {}

    def fake_cmd_build(args):
        captured["args"] = args
        return 17

    monkeypatch.setattr(cli, "_cmd_build", fake_cmd_build)

    parsed_build_args = argparse.Namespace(
        command="build",
        model="repo/model",
        output="/tmp/out.trtfb",
        max_cache_length=1024,
        verbose=True,
    )
    parse_args_argv: dict[str, list[str]] = {}

    def fake_parse_args(self, args=None, namespace=None):
        parse_args_argv["value"] = list(args or [])
        return parsed_build_args

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trtmc",
            "repo/model",
            "-o",
            "/tmp/out.trtfb",
            "--max-cache-length",
            "1024",
            "--verbose",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 17
    assert parse_args_argv["value"] == [
        "build",
        "repo/model",
        "-o",
        "/tmp/out.trtfb",
        "--max-cache-length",
        "1024",
        "--verbose",
    ]
    assert captured["args"].command == "build"
    assert captured["args"].model == "repo/model"
    assert captured["args"].output == "/tmp/out.trtfb"
    assert captured["args"].max_cache_length == 1024
    assert captured["args"].verbose is True


def test_main_without_args_prints_help_and_exits_zero(monkeypatch):
    """Intent: cover no-argument CLI behavior.
    Preconditions: argv contains only program name, so no command or build args are provided.
    Postconditions: main prints help and exits with code 0.
    """
    monkeypatch.setattr(sys, "argv", ["trtmc"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0


def test_main_explicit_version_dispatch(monkeypatch):
    """Intent: verify explicit subcommand dispatch path.
    Preconditions: argv specifies "version", and _cmd_version is stubbed to a sentinel exit code.
    Postconditions: main invokes _cmd_version and exits with the sentinel code.
    """
    called = {"count": 0}

    def fake_cmd_version(_args):
        called["count"] += 1
        return 23

    monkeypatch.setattr(cli, "_cmd_version", fake_cmd_version)
    monkeypatch.setattr(sys, "argv", ["trtmc", "version"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert called["count"] == 1
    assert exc.value.code == 23


def test_main_unknown_command_prints_help_and_exits_one(monkeypatch):
    """Intent: hit defensive branch where dispatch lookup returns no handler.
    Preconditions: parse_args is monkeypatched to return an unknown command token.
    Postconditions: main prints help and exits with code 1.
    """
    help_calls = {"count": 0}

    def fake_parse_args(self, args=None, namespace=None):
        return argparse.Namespace(command="unknown-command")

    def fake_print_help(self):
        help_calls["count"] += 1

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    monkeypatch.setattr(argparse.ArgumentParser, "print_help", fake_print_help)
    monkeypatch.setattr(sys, "argv", ["trtmc", "unknown-command"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert help_calls["count"] == 1


def test_auto_select_build_backend_prefers_raw_trt(tmp_path, monkeypatch):
    """Intent: auto backend selection should use the native TRT path.
    Preconditions: a local config matches a raw plugin.
    Postconditions: _auto_select_build_backend returns "trt".
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "synthetic_native"}),
        encoding="utf-8",
    )

    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.engine_defs as engine_defs

    monkeypatch.setattr(engine_builder, "_resolve_model", lambda model_ref: str(model_dir))
    monkeypatch.setattr(engine_builder, "find_plugin", lambda model_type: object())
    monkeypatch.setattr(engine_builder, "find_diffusion_plugin", lambda pipeline_class: None)
    monkeypatch.setattr(engine_defs, "get_engine_def", lambda name: object())

    method, resolved = cli._auto_select_build_backend(str(model_dir))

    assert method == "trt"
    assert resolved == str(model_dir)


def test_native_diffusion_config_without_model_index_uses_claimed_family(
    tmp_path, monkeypatch
):
    """Official native checkpoints route through their diffusion family.

    Wan2.2-TI2V-5B has ``_class_name=WanModel`` in ``config.json`` and no
    ``model_index.json``. Both public CLI discovery passes must retain the
    claimed family so profile resolution and build dispatch agree.
    """
    model_dir = tmp_path / "Wan2.2-TI2V-5B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanModel", "model_type": "ti2v"}),
        encoding="utf-8",
    )

    import tensorrt_model_connect.engine_builder as engine_builder

    plugin = types.SimpleNamespace(
        name="wan2_2_ti2v",
        pipeline_classes=("WanModel", "WanPipeline"),
    )
    monkeypatch.setattr(engine_builder, "_resolve_model", lambda _model_ref: str(model_dir))
    monkeypatch.setattr(engine_builder, "find_plugin", lambda _config: plugin)
    monkeypatch.setattr(engine_builder, "find_diffusion_plugin", lambda _class_name: plugin)

    method, resolved = cli._auto_select_build_backend(str(model_dir))
    metadata_ref, family = cli._resolve_build_model_metadata(str(model_dir), method)

    assert method == "trt"
    assert resolved == str(model_dir)
    assert metadata_ref == str(model_dir)
    assert family == "wan2_2_ti2v"


def test_wan_hf_id_selects_bf16_before_download_and_preserves_source_id(
    tmp_path, monkeypatch
):
    """Wan's model ID is enough to select BF16 and source provenance."""
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"_class_name": "WanModel"}), encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "_resolve_build_model_metadata",
        lambda model_ref, _method: (str(model_dir), "wan2_2_ti2v"),
    )
    monkeypatch.setattr(cli, "_maybe_reexec_build_in_profile", lambda *_args: None)
    monkeypatch.setattr(cli, "_preflight_family_build_dependencies", lambda _family: None)

    import tensorrt_model_connect.engine_builder as engine_builder

    def fake_build(**kwargs):
        captured.update(kwargs)

    # The CLI has already completed optimized-runtime routing at this point,
    # so it enters the native builder directly. Keep the provenance and
    # family-default assertions on that production seam.
    monkeypatch.setattr(engine_builder, "_build_native_impl", fake_build)
    args = argparse.Namespace(
        model="Wan-AI/Wan2.2-TI2V-5B",
        output=str(tmp_path / "wan.trtfb"),
        max_cache_length=512,
        precision=None,
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        _skip_profile_resolution=False,
    )

    assert cli._cmd_build(args) == 0
    assert captured["model_id_or_path"] == str(model_dir)
    assert captured["source_model_ref"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert captured["precision"] == "bf16"


def test_wan_explicit_unsupported_precision_fails_before_download(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli, "_preflight_family_build_dependencies", lambda _family: None)
    monkeypatch.setattr(
        cli,
        "_resolve_build_model_metadata",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("checkpoint download must not start")),
    )
    args = argparse.Namespace(
        model="Wan-AI/Wan2.2-TI2V-5B",
        output=str(tmp_path / "wan.trtfb"),
        precision="fp32",
        verbose=False,
        _skip_profile_resolution=False,
    )

    assert cli._cmd_build(args) == 1
    assert "does not support build precision 'fp32'" in capsys.readouterr().err


def test_wan_missing_torch_preflight_names_model_extra(monkeypatch):
    import tensorrt_model_connect.families as families

    monkeypatch.setattr(
        families.importlib.util,
        "find_spec",
        lambda module: None if module == "torch" else object(),
    )
    with pytest.raises(RuntimeError, match=r"tensorrt-model-connect\[wan\]"):
        cli._preflight_family_build_dependencies("wan2_2_ti2v")

    assert families.family_build_precision("wan2_2_ti2v") == "bf16"
    assert families.family_build_python_modules("wan2_2_ti2v") == ("torch",)
    assert families.family_build_dependency_extra("wan2_2_ti2v") == "wan"


def test_wan_missing_torch_stops_cli_before_checkpoint_download(
    monkeypatch, tmp_path, capsys
):
    import tensorrt_model_connect.families as families

    monkeypatch.setattr(
        families.importlib.util,
        "find_spec",
        lambda module: None if module == "torch" else object(),
    )
    monkeypatch.setattr(
        cli,
        "_resolve_build_model_metadata",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("checkpoint download must not start")
        ),
    )
    args = argparse.Namespace(
        model="Wan-AI/Wan2.2-TI2V-5B",
        output=str(tmp_path / "wan.trtfb"),
        precision=None,
        verbose=False,
        _skip_profile_resolution=False,
    )

    assert cli._cmd_build(args) == 1
    stderr = capsys.readouterr().err
    assert "requires build dependency module(s): torch" in stderr
    assert "tensorrt-model-connect[wan]" in stderr


def test_wan_torch_is_build_extra_not_global_runtime_dependency():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8")
    )
    assert "torch>=2.0" not in pyproject["project"]["dependencies"]
    assert pyproject["project"]["optional-dependencies"]["wan"] == ["torch>=2.0"]


def test_auto_select_build_backend_errors_for_unsupported_native_model(tmp_path, monkeypatch):
    """Intent: auto backend selection should fail clearly when native TRT is unsupported.
    Preconditions: a local config has no native raw or diffusion plugin.
    Postconditions: _auto_select_build_backend raises a RuntimeError mentioning native TRT.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "unsupported"}), encoding="utf-8")

    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.engine_defs as engine_defs

    monkeypatch.setattr(engine_builder, "_resolve_model", lambda model_ref: str(model_dir))
    monkeypatch.setattr(engine_builder, "find_plugin", lambda model_type: None)
    monkeypatch.setattr(engine_builder, "find_diffusion_plugin", lambda pipeline_class: None)
    monkeypatch.setattr(engine_defs, "get_engine_def", lambda name: None)

    with pytest.raises(RuntimeError, match="No native TRT family plugin"):
        cli._auto_select_build_backend(str(model_dir))
