from __future__ import annotations

from pathlib import Path

import tensorrt_model_connect.native_cli as native_cli


def test_existing_executable_returns_first_executable(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    non_executable = tmp_path / "non_executable"
    executable = tmp_path / "trtmc"
    non_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    assert native_cli._existing_executable([missing, non_executable, executable]) == executable


def test_native_binary_candidates_puts_override_first(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "custom-trtmc"
    monkeypatch.setenv("TRTMC_NATIVE_BIN", str(override))

    candidates = native_cli._native_binary_candidates()

    assert candidates[0] == override
