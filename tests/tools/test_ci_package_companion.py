# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused wheel contracts for the Wan2.2 builder-only companion DSO."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

import _pyproject_backend as pyproject_backend
from tools.ci.package import InstalledWheelValidator, WheelArchiveValidator, WheelPackageManager
from tools.ci.process import CiError


_REPOSITORY = Path(__file__).resolve().parents[2]
_WAN22_ATTRIBUTION_PATHS = (
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/LICENSE.txt",
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/README.trtmc.md",
    "tensorrt_model_connect/families/wan2_2_ti2v/dit_cuda_plugins/third_party/"
    "cudnn_frontend/include/cudnn_frontend/thirdparty/nlohmann/LICENSE.MIT",
)


class _AuditContext:
    def __init__(self, dynamic: str = "") -> None:
        self.dynamic = dynamic

    def output(self, command) -> str:
        if command[0] == "readelf":
            return self.dynamic
        return "platform tag: manylinux_2_39_aarch64"


def _write_wheel(
    path: Path,
    companions: list[str],
    *,
    tensorrt_requirement: str = "tensorrt==11.2.0.113",
    attributions: dict[str, bytes] | None = None,
) -> None:
    package = "tensorrt_model_connect"
    scripts = "tensorrt_model_connect-0.1.0.data/scripts"
    metadata = "tensorrt_model_connect-0.1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            f"{package}/bin/trtmc",
            f"{scripts}/trtmc",
            f"{package}/bin/libtrtmc_core.so",
            f"{scripts}/libtrtmc_core.so",
            f"{package}/bin/libtrtmc_backend_trt.so",
            f"{package}/bin/libtrtmc_model_wan2_2_ti2v.so",
        ):
            archive.writestr(name, b"\x7fELF")
        for companion in companions:
            archive.writestr(f"{package}/bin/{companion}", b"\x7fELF")
        if attributions is None:
            attributions = {
                name: (_REPOSITORY / "python" / name).read_bytes()
                for name in _WAN22_ATTRIBUTION_PATHS
            }
        for name, content in attributions.items():
            archive.writestr(name, content)
        archive.writestr(
            f"{metadata}/METADATA",
            f"Requires-Dist: {tensorrt_requirement}\n"
            "Requires-Dist: apache-tvm-ffi==0.1.12\n",
        )
        archive.writestr(
            f"{metadata}/WHEEL",
            "Tag: py3-none-manylinux_2_39_aarch64\n",
        )


def test_wheel_archive_requires_exactly_one_wan22_builder_companion(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    _write_wheel(
        wheel,
        ["libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"],
    )

    WheelArchiveValidator(_AuditContext(), "manylinux_2_39_aarch64").validate([wheel])


@pytest.mark.parametrize("missing", _WAN22_ATTRIBUTION_PATHS)
def test_wheel_archive_rejects_missing_wan22_vendored_attribution(
    tmp_path: Path,
    missing: str,
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    attributions = {
        name: (_REPOSITORY / "python" / name).read_bytes()
        for name in _WAN22_ATTRIBUTION_PATHS
        if name != missing
    }
    _write_wheel(
        wheel,
        ["libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"],
        attributions=attributions,
    )

    with pytest.raises(
        CiError,
        match=rf"expected exactly one Wan2\.2 vendored attribution file {re.escape(missing)}; found 0",
    ):
        WheelArchiveValidator(_AuditContext(), "manylinux_2_39_aarch64").validate([wheel])


@pytest.mark.parametrize("tampered", _WAN22_ATTRIBUTION_PATHS)
def test_wheel_archive_rejects_tampered_wan22_vendored_attribution(
    tmp_path: Path,
    tampered: str,
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    attributions = {
        name: (_REPOSITORY / "python" / name).read_bytes()
        for name in _WAN22_ATTRIBUTION_PATHS
    }
    attributions[tampered] += b"tampered"
    _write_wheel(
        wheel,
        ["libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"],
        attributions=attributions,
    )

    with pytest.raises(
        CiError,
        match=rf"Wan2\.2 vendored attribution file {re.escape(tampered)} has SHA256",
    ):
        WheelArchiveValidator(_AuditContext(), "manylinux_2_39_aarch64").validate([wheel])


@pytest.mark.parametrize(
    "companions",
    (
        [],
        [
            "libtrtmc_model_wan2_2_ti2v_plugins_trt11_1.so",
            "libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so",
        ],
    ),
)
def test_wheel_archive_rejects_missing_or_ambiguous_wan22_builder_companion(
    tmp_path: Path, companions: list[str]
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    _write_wheel(wheel, companions)

    with pytest.raises(CiError, match="exactly one ABI-tagged Wan2.2 builder companion"):
        WheelArchiveValidator(_AuditContext(), "manylinux_2_39_aarch64").validate([wheel])


def test_wheel_archive_rejects_wan22_companion_dependency_abi_mismatch(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    _write_wheel(
        wheel,
        ["libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"],
        tensorrt_requirement="tensorrt==11.2.0.113",
    )

    with pytest.raises(CiError, match="companion=11.0, Requires-Dist tensorrt=11.2"):
        WheelArchiveValidator(_AuditContext(), "manylinux_2_39_aarch64").validate([wheel])


@pytest.mark.parametrize("tag", ("RPATH", "RUNPATH"))
def test_wheel_archive_rejects_wan22_companion_runtime_search_path(
    tmp_path: Path, tag: str
) -> None:
    wheel = tmp_path / "trtmc-0.1.0-py3-none-manylinux_2_39_aarch64.whl"
    _write_wheel(
        wheel,
        ["libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"],
    )

    with pytest.raises(CiError, match="must not contain DT_RPATH/DT_RUNPATH"):
        WheelArchiveValidator(
            _AuditContext(f"0x000000000000001d ({tag}) Library: [$ORIGIN]"),
            "manylinux_2_39_aarch64",
        ).validate([wheel])


def test_build_backend_strips_only_wan22_companion_rpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    extension = staging / "_native.cpython-312-aarch64-linux-gnu.so"
    companion = staging / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"
    extension.write_bytes(b"extension")
    companion.write_bytes(b"companion")
    rpaths: dict[str, str] = {}

    def original_patch_rpath(root: Path) -> None:
        for path in root.rglob("*.so"):
            rpaths[path.name] = "$ORIGIN"

    conan_build = SimpleNamespace(patch_rpath=original_patch_rpath)

    def remove_rpath(command, **kwargs):
        assert command == ["patchelf", "--remove-rpath", str(companion)]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        rpaths[companion.name] = ""
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pyproject_backend.subprocess, "run", remove_rpath)
    pyproject_backend._install_wan22_wheel_rpath_policy(conan_build)
    conan_build.patch_rpath(staging)

    assert rpaths[extension.name] == "$ORIGIN"
    assert rpaths[companion.name] == ""


@pytest.mark.parametrize(
    ("error", "message"),
    (
        (FileNotFoundError("patchelf"), "patchelf is required"),
        (
            subprocess.CalledProcessError(1, ["patchelf"], stderr="invalid ELF"),
            "failed to remove.*invalid ELF",
        ),
    ),
)
def test_build_backend_fails_closed_when_companion_cannot_be_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    companion = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"
    companion.write_bytes(b"companion")
    conan_build = SimpleNamespace(patch_rpath=lambda _root: None)

    def fail(_command, **_kwargs):
        raise error

    monkeypatch.setattr(pyproject_backend.subprocess, "run", fail)
    pyproject_backend._install_wan22_wheel_rpath_policy(conan_build)

    with pytest.raises(RuntimeError, match=message):
        conan_build.patch_rpath(tmp_path)


def test_build_backend_rejects_malformed_companion_name(tmp_path: Path) -> None:
    malformed = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11.so"
    malformed.write_bytes(b"companion")
    conan_build = SimpleNamespace(patch_rpath=lambda _root: None)
    pyproject_backend._install_wan22_wheel_rpath_policy(conan_build)

    with pytest.raises(RuntimeError, match="malformed Wan2.2 builder companion"):
        conan_build.patch_rpath(tmp_path)


def test_build_backend_rejects_companion_dependency_abi_mismatch(tmp_path: Path) -> None:
    companion = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so"
    companion.write_bytes(b"companion")
    conan_build = SimpleNamespace(patch_rpath=lambda _root: None)
    pyproject_backend._install_wan22_wheel_rpath_policy(conan_build)

    with pytest.raises(
        RuntimeError,
        match="companion=11.0, Requires-Dist tensorrt=11.2",
    ):
        conan_build.patch_rpath(tmp_path)


def test_installed_wheel_validator_requires_the_wan22_builder_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_dir = tmp_path / "site-packages/tensorrt_model_connect"
    native_dir = package_dir / "bin"
    native_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("", encoding="utf-8")
    trtmc = native_dir / "trtmc"
    trtmc.write_bytes(b"\x7fELF")
    (native_dir / "libtrtmc_backend_trt.so").write_bytes(b"\x7fELF")
    companion = native_dir / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"
    companion.write_bytes(b"\x7fELF")

    package = ModuleType("tensorrt_model_connect")
    package.__file__ = str(package_file)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect", package)
    monkeypatch.setattr("tools.ci.package.importlib.resources.files", lambda _name: package_dir)
    monkeypatch.setattr(
        "tools.ci.package.importlib.metadata.requires",
        lambda _name: ["tensorrt==11.2.0.113"],
    )
    monkeypatch.setattr("tools.ci.package.shutil.which", lambda _name: str(trtmc))
    validator = InstalledWheelValidator(tmp_path / "repository")
    validator.validate(tmp_path / "fixture.whl")

    companion.unlink()
    with pytest.raises(CiError, match="exactly one ABI-tagged Wan2.2 builder companion"):
        validator.validate(tmp_path / "fixture.whl")


def test_installed_wheel_validator_rejects_companion_dependency_abi_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "site-packages/tensorrt_model_connect"
    native_dir = package_dir / "bin"
    native_dir.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_text("", encoding="utf-8")
    trtmc = native_dir / "trtmc"
    trtmc.write_bytes(b"\x7fELF")
    (native_dir / "libtrtmc_backend_trt.so").write_bytes(b"\x7fELF")
    (native_dir / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_0.so").write_bytes(b"\x7fELF")

    package = ModuleType("tensorrt_model_connect")
    package.__file__ = str(package_file)
    monkeypatch.setitem(sys.modules, "tensorrt_model_connect", package)
    monkeypatch.setattr("tools.ci.package.importlib.resources.files", lambda _name: package_dir)
    monkeypatch.setattr(
        "tools.ci.package.importlib.metadata.requires",
        lambda _name: ["tensorrt==11.2.0.113"],
    )
    monkeypatch.setattr("tools.ci.package.shutil.which", lambda _name: str(trtmc))

    with pytest.raises(CiError, match="companion=11.0, Requires-Dist tensorrt=11.2"):
        InstalledWheelValidator(tmp_path / "repository").validate(tmp_path / "fixture.whl")


def test_packaged_wan22_companion_rejects_any_runtime_search_path(
    tmp_path: Path,
) -> None:
    companion = tmp_path / "libtrtmc_model_wan2_2_ti2v_plugins_trt11_2.so"
    companion.write_bytes(b"\x7fELF")

    InstalledWheelValidator.require_no_runtime_search_path(
        companion,
        "0x0000000000000001 (NEEDED) Shared library: [libcudnn.so.9]",
    )
    for tag in ("RPATH", "RUNPATH"):
        with pytest.raises(CiError, match="must not contain DT_RPATH/DT_RUNPATH"):
            InstalledWheelValidator.require_no_runtime_search_path(
                companion,
                f"0x000000000000001d ({tag}) Library {tag.lower()}: [/workspace/build]",
            )


@pytest.mark.parametrize("relative", ("build/Release", "build/arch/Release"))
def test_package_manager_accepts_supported_conan_cmake_layouts(
    tmp_path: Path,
    relative: str,
) -> None:
    conan_out = tmp_path / "conan_out"
    cache = conan_out / relative / "CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("", encoding="utf-8")

    manager = WheelPackageManager(SimpleNamespace())

    assert manager._conan_cmake_build_dir(conan_out) == cache.parent


def test_package_manager_rejects_missing_or_ambiguous_conan_cmake_layout(
    tmp_path: Path,
) -> None:
    conan_out = tmp_path / "conan_out"
    manager = WheelPackageManager(SimpleNamespace())

    with pytest.raises(CiError, match="expected exactly one reusable CMakeCache.txt"):
        manager._conan_cmake_build_dir(conan_out)

    for relative in ("build/Release", "build/arch/Release"):
        cache = conan_out / relative / "CMakeCache.txt"
        cache.parent.mkdir(parents=True)
        cache.write_text("", encoding="utf-8")

    with pytest.raises(CiError, match="expected exactly one reusable CMakeCache.txt"):
        manager._conan_cmake_build_dir(conan_out)
