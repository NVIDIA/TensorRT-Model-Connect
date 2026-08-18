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
    bundle_path = tmp_path / "minimal.bundle"
    header = {
        "model_id": "minimal-model",
        "model_type": "example_decoder",
        "family": "example_family",
    }
    payload = json.dumps(header).encode("utf-8")
    with open(bundle_path, "wb") as f:
        f.write(b"BUNDLE\x01\x00")
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
    bundle_path = tmp_path / "malformed.bundle"
    malformed = b'{"model_id": "bad-json"'
    with open(bundle_path, "wb") as f:
        f.write(b"BUNDLE\x01\x00")
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
