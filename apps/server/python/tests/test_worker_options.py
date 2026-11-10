# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import builtins
from pathlib import Path
from typing import Any

import pytest

from trtmc_server.cli import byok_kernel, main, packaged_runtime_root, parser
from trtmc_server.worker import WorkerLoadOptions


def test_runtime_root_is_optional_and_override_is_preserved() -> None:
    assert WorkerLoadOptions().argv() == []
    assert WorkerLoadOptions(runtime_root="/opt/runtime").argv() == [
        "--runtime-root",
        "/opt/runtime",
    ]


def test_server_has_no_command_line_bearer_token() -> None:
    command = parser()
    assert "--api-key" not in command.format_help()
    assert all("--api-key" not in action.option_strings for action in command._actions)


def test_cli_help_does_not_import_optional_serving_dependencies(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def import_without_uvicorn(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "uvicorn":
            raise ModuleNotFoundError("No module named 'uvicorn'", name="uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_uvicorn)
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    assert "Serve text-generation bundles" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="2"):
        main([])
    assert "tensorrt-model-connect[serve]" in capsys.readouterr().err


def test_packaged_runtime_root_is_derived_from_control_plane(tmp_path: Path) -> None:
    control_plane = tmp_path / "trtmc_server/cli.py"
    control_plane.parent.mkdir()
    control_plane.write_text("")
    runtime_root = tmp_path / "tensorrt_model_connect/bin"
    runtime_root.mkdir(parents=True)

    assert packaged_runtime_root(control_plane) is None
    for name in ("trtmc-server", "libtrtmc_runtime.so", "libtrtmc_backend_trt.so"):
        (runtime_root / name).write_text("")

    assert packaged_runtime_root(control_plane) == runtime_root


def test_byok_kernel_needs_all_three_values_and_reaches_the_worker(tmp_path: Path) -> None:
    library = tmp_path / "kernel.so"
    library.write_bytes(b"")

    def given(**values: str) -> argparse.Namespace:
        return argparse.Namespace(**{"byok_library": None, "byok_function": None, "byok_name": None, **values})

    assert byok_kernel(given()) is None
    kernel = byok_kernel(given(byok_library=str(library), byok_function="run", byok_name="family.kernel"))
    assert kernel == (str(library.resolve()), "run", "family.kernel")
    assert WorkerLoadOptions(byok_kernel=kernel).argv() == [
        "--byok-library", str(library.resolve()), "--byok-function", "run", "--byok-name", "family.kernel",
    ]
    with pytest.raises(ValueError, match="go together"):
        byok_kernel(given(byok_library=str(library)))
    with pytest.raises(ValueError, match="must be a file"):
        byok_kernel(given(byok_library=str(tmp_path / "missing.so"), byok_function="run", byok_name="k"))
    with pytest.raises(ValueError, match="byok_kernel"):
        WorkerLoadOptions(byok_kernel=(str(library), "", "k"))
