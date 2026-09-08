# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tensorrt_model_connect import native_cli


def test_main_executes_the_native_cli_in_the_package_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "tensorrt_model_connect"
    executable = package / "bin" / "trtmc"
    executable.parent.mkdir(parents=True)
    executable.touch()
    executable.chmod(0o755)
    monkeypatch.setattr(native_cli, "__file__", str(package / "native_cli.py"))
    monkeypatch.setattr(sys, "argv", ["trtmc", "version"])

    called: tuple[Path, list[str]] | None = None

    def capture_execv(path: Path, arguments: list[str]) -> None:
        nonlocal called
        called = (path, arguments)

    monkeypatch.setattr(native_cli.os, "execv", capture_execv)
    native_cli.main()

    assert called == (executable, [str(executable), "version"])


def test_main_rejects_an_incomplete_native_product(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "tensorrt_model_connect"
    package.mkdir()
    monkeypatch.setattr(native_cli, "__file__", str(package / "native_cli.py"))

    with pytest.raises(RuntimeError, match="native trtmc executable is missing"):
        native_cli.main()
