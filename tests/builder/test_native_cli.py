from __future__ import annotations

from pathlib import Path

import tensorrt_model_connect.native_cli as native_cli


def test_configure_runtime_environment_sets_venv_and_tensorrt_libs(
    monkeypatch, tmp_path: Path
) -> None:
    trt_lib_dir = tmp_path / "tensorrt_libs"
    trt_lib_dir.mkdir()
    monkeypatch.delenv("TRTMC_PYTHON", raising=False)
    monkeypatch.delenv("TRTMC_DISABLE_SOURCE_PYTHONPATH", raising=False)
    monkeypatch.delenv("TRTMC_TRT_LIBRARY_DIR", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(native_cli.sys, "executable", "/tmp/trtmc-venv/bin/python")
    monkeypatch.setattr(native_cli.sys, "prefix", "/tmp/trtmc-venv")
    monkeypatch.setattr(native_cli.sys, "base_prefix", "/usr")
    monkeypatch.setattr(native_cli, "_tensorrt_library_dir", lambda: trt_lib_dir)

    native_cli._configure_runtime_environment()

    assert native_cli.os.environ["TRTMC_PYTHON"] == "/tmp/trtmc-venv/bin/python"
    assert native_cli.os.environ["TRTMC_DISABLE_SOURCE_PYTHONPATH"] == "1"
    assert native_cli.os.environ["VIRTUAL_ENV"] == "/tmp/trtmc-venv"
    assert native_cli.os.environ["TRTMC_TRT_LIBRARY_DIR"] == str(trt_lib_dir)


def test_configure_runtime_environment_preserves_user_overrides(
    monkeypatch, tmp_path: Path
) -> None:
    trt_lib_dir = tmp_path / "tensorrt_libs"
    trt_lib_dir.mkdir()
    monkeypatch.setenv("TRTMC_PYTHON", "/custom/python")
    monkeypatch.setenv("TRTMC_DISABLE_SOURCE_PYTHONPATH", "0")
    monkeypatch.setenv("TRTMC_TRT_LIBRARY_DIR", "/custom/trt")
    monkeypatch.setenv("VIRTUAL_ENV", "/custom/venv")
    monkeypatch.setattr(native_cli.sys, "executable", "/tmp/trtmc-venv/bin/python")
    monkeypatch.setattr(native_cli.sys, "prefix", "/tmp/trtmc-venv")
    monkeypatch.setattr(native_cli.sys, "base_prefix", "/usr")
    monkeypatch.setattr(native_cli, "_tensorrt_library_dir", lambda: trt_lib_dir)

    native_cli._configure_runtime_environment()

    assert native_cli.os.environ["TRTMC_PYTHON"] == "/custom/python"
    assert native_cli.os.environ["TRTMC_DISABLE_SOURCE_PYTHONPATH"] == "0"
    assert native_cli.os.environ["VIRTUAL_ENV"] == "/custom/venv"
    assert native_cli.os.environ["TRTMC_TRT_LIBRARY_DIR"] == "/custom/trt"
