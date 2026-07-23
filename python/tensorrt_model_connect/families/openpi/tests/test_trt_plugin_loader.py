# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect.families.openpi import trt_plugin_loader


class _Registry:
    def __init__(self, creator=None):
        self.creator = creator

    def get_creator(self, name: str, version: str, namespace: str):
        assert name == "OpenPIRopeQK"
        assert version == "1"
        assert namespace == ""
        return self.creator


class _Trt:
    def __init__(self, registry: _Registry):
        self.registry = registry

    def get_plugin_registry(self) -> _Registry:
        return self.registry


def test_preloaded_openpi_creator_never_touches_the_filesystem(monkeypatch) -> None:
    expected = object()
    trt = _Trt(_Registry(expected))
    monkeypatch.setattr(
        trt_plugin_loader,
        "_plugin_library_path",
        lambda: pytest.fail("preloaded creator must not resolve a DSO path"),
    )

    assert trt_plugin_loader.require_openpi_plugin_creator("OpenPIRopeQK", trt=trt) is expected


def test_openpi_plugin_path_honors_explicit_library(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "custom-openpi.so"
    monkeypatch.setenv("TRTMC_OPENPI_TRT_PLUGIN", str(configured))
    monkeypatch.setenv("TRTMC_MODEL_PLUGIN_DIR", str(tmp_path / "ignored"))

    assert trt_plugin_loader._plugin_library_path() == configured


def test_openpi_plugin_path_honors_model_plugin_directory(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "plugins" / "openpi" / "libtrtmc_model_openpi.so"
    configured.parent.mkdir(parents=True)
    configured.write_bytes(b"test")
    monkeypatch.delenv("TRTMC_OPENPI_TRT_PLUGIN", raising=False)
    monkeypatch.setenv("TRTMC_MODEL_PLUGIN_DIR", str(tmp_path / "plugins"))

    assert trt_plugin_loader._plugin_library_path() == configured


def test_openpi_plugin_path_uses_installed_wheel_dso(monkeypatch, tmp_path: Path) -> None:
    module = tmp_path / "tensorrt_model_connect/families/openpi/trt_plugin_loader.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    installed = tmp_path / "tensorrt_model_connect/bin/libtrtmc_model_openpi.so"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"test")
    monkeypatch.delenv("TRTMC_OPENPI_TRT_PLUGIN", raising=False)
    monkeypatch.delenv("TRTMC_MODEL_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(trt_plugin_loader, "__file__", str(module))

    assert trt_plugin_loader._plugin_library_path() == installed


def test_openpi_creator_loads_only_the_exact_build_relative_dso(
    monkeypatch, tmp_path: Path
) -> None:
    plugin_path = tmp_path / "libtrtmc_model_openpi.so"
    plugin_path.write_bytes(b"test")
    expected = object()
    registry = _Registry()
    calls: list[tuple[str, int]] = []

    def fake_cdll(path: str, *, mode: int):
        calls.append((path, mode))
        registry.creator = expected
        return object()

    monkeypatch.setattr(trt_plugin_loader, "_plugin_library_path", lambda: plugin_path)
    monkeypatch.setattr(trt_plugin_loader.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(trt_plugin_loader, "_openpi_plugin_handle", None)

    actual = trt_plugin_loader.require_openpi_plugin_creator("OpenPIRopeQK", trt=_Trt(registry))

    assert actual is expected
    assert calls == [(str(plugin_path), trt_plugin_loader.ctypes.RTLD_GLOBAL)]


def test_openpi_creator_rejects_a_missing_or_symlinked_dso(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.so"
    monkeypatch.setattr(trt_plugin_loader, "_plugin_library_path", lambda: missing)
    with pytest.raises(RuntimeError, match="regular model DSO"):
        trt_plugin_loader.require_openpi_plugin_creator("OpenPIRopeQK", trt=_Trt(_Registry()))

    target = tmp_path / "target.so"
    target.write_bytes(b"test")
    symlink = tmp_path / "plugin.so"
    symlink.symlink_to(target)
    monkeypatch.setattr(trt_plugin_loader, "_plugin_library_path", lambda: symlink)
    with pytest.raises(RuntimeError, match="regular model DSO"):
        trt_plugin_loader.require_openpi_plugin_creator("OpenPIRopeQK", trt=_Trt(_Registry()))
