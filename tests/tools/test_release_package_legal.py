# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import os
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

import _pyproject_backend as backend
from tools import model_ci
from tools.ci.package import RELEASE_LEGAL_FILES, WheelArchiveValidator
from tools.ci.process import CiError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SAM2_HOI_NATIVE_PLUGINS = (
    REPOSITORY_ROOT / "python/tensorrt_model_connect/families/sam2_hoi/native_plugins"
)
SAM2_HOI_PAFPN_BN_HELPER = (
    REPOSITORY_ROOT / "python/tensorrt_model_connect/families/sam2_hoi/pafpn_bn_invstd_helper.cu"
)
requires_sam2_hoi = pytest.mark.skipif(
    not SAM2_HOI_NATIVE_PLUGINS.is_dir(),
    reason="SAM2-HOI-owned legal closure is absent from this family projection",
)


class _Context:
    repository = REPOSITORY_ROOT


def _validate_legal_payload(wheel: Path) -> None:
    validator = WheelArchiveValidator(_Context(), "manylinux_2_39_aarch64")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        validator._validate_legal_payload(
            wheel,
            archive,
            names,
            metadata_name,
            archive.read(metadata_name).decode(),
        )


def _synthetic_wheel(
    path: Path,
    *,
    metadata: str,
    omitted: str | None = None,
    stale: str | None = None,
) -> Path:
    dist_info = "tensorrt_model_connect-0.1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        for relative in RELEASE_LEGAL_FILES:
            if relative == omitted:
                continue
            content = b"stale\n" if relative == stale else (REPOSITORY_ROOT / relative).read_bytes()
            archive.writestr(f"{dist_info}/licenses/{relative}", content)
    return path


def _manifest_members(root: Path) -> set[Path]:
    members = {Path("MANIFEST.sha256")}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        _digest, relative = line.split(maxsplit=1)
        members.add(Path(relative.removeprefix("./")))
    return members


def _exact_hiera_release_members() -> set[Path]:
    return {
        path.relative_to(SAM2_HOI_NATIVE_PLUGINS)
        for path in SAM2_HOI_NATIVE_PLUGINS.rglob("*")
        if path.is_file()
    }


def _assert_manifest_integrity(payloads: dict[Path, bytes], manifest: Path, *, root: Path) -> None:
    for line in payloads[manifest].decode("utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        member = root / relative.removeprefix("./")
        assert member in payloads
        assert hashlib.sha256(payloads[member]).hexdigest() == expected


def test_pyproject_declares_pep639_release_legal_files() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]

    assert project["license"] == "Apache-2.0"
    assert tuple(project["license-files"]) == RELEASE_LEGAL_FILES
    assert "ASSET_LICENSES.md" in model_ci.PLATFORM_PROJECTION_EXACT
    assert "ASSET_LICENSES.md" in model_ci.LEGAL_OR_DOC_EXACT


def test_libjpeg_turbo_conan_dependency_has_notice_attribution() -> None:
    dependency = "libjpeg-turbo/2.1.5"
    name, version = dependency.split("/")
    conanfile = (REPOSITORY_ROOT / "conanfile.py").read_text()
    notice = (REPOSITORY_ROOT / "NOTICE").read_text()

    assert f'self.requires("{dependency}")' in conanfile

    separator = "-" * 79
    heading = f"{separator}\n{name}\n{separator}\n"
    _, found, remainder = notice.partition(heading)
    assert found, f"NOTICE is missing the {name} section"
    section, found, _ = remainder.partition(f"\n{separator}\n")
    assert found, f"NOTICE has an unterminated {name} section"

    assert f"{name} {version}" in section
    assert f"https://github.com/libjpeg-turbo/libjpeg-turbo/tree/{version}" in section
    assert "This software is based in part on the work of the Independent JPEG Group." in section
    assert "The authors make NO WARRANTY or representation" in section
    assert "copyright (C) 1991-2020, Thomas G. Lane, Guido Vollbeding" in section
    assert "Permission is hereby granted to use, copy, modify, and distribute" in section
    assert "(1) If any part of the source code for this software is distributed" in section
    assert "(2) If only executable code is distributed" in section
    assert "(3) Permission for use of this software" in section
    assert "Permission is NOT granted for the use of any IJG author's name" in section
    assert "The Modified (3-clause) BSD License" in section
    assert "Copyright (C)2009-2023 D. R. Commander.  All Rights Reserved." in section
    assert "Copyright (C)2015 Viktor Szathm\u00e1ry.  All Rights Reserved." in section
    assert "Redistributions of source code must retain" in section
    assert "Redistributions in binary form must reproduce" in section
    assert "Neither the name of the libjpeg-turbo Project" in section
    assert 'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"' in section
    assert "The zlib License is a subset of" in section
    assert "covers the libjpeg-turbo SIMD extensions" in section
    assert "Copyright (C) 1995-2022 Jean-loup Gailly and Mark Adler" in section
    assert "This software is provided 'as-is', without any express or implied warranty" in section
    assert "Permission is granted to anyone to use this software for any purpose" in section
    assert "1. The origin of this software must not be misrepresented" in section
    assert "2. Altered source versions must be plainly marked as such" in section
    assert "3. This notice may not be removed or altered from any source distribution" in section


@requires_sam2_hoi
def test_sam2_hoi_pytorch_native_operators_have_notice_attribution() -> None:
    notice = (REPOSITORY_ROOT / "NOTICE").read_text()

    assert "LayerNorm, persistent-softmax, sigmoid, and fixed bicubic-upsample" in notice
    assert "e2d141dbde55c2a4370fac5165b0561b6af4798b" in notice
    assert "aten/src/ATen/native/cuda/layer_norm_kernel.cu" in notice
    assert "aten/src/ATen/native/cuda/block_reduce.cuh" in notice
    assert "aten/src/ATen/native/cuda/PersistentSoftmax.cuh" in notice
    assert "aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu" in notice
    assert "aten/src/ATen/native/cuda/UpSample.cuh" in notice


@requires_sam2_hoi
def test_exact_hiera_dependencies_have_complete_notice_attribution() -> None:
    notice = (REPOSITORY_ROOT / "NOTICE").read_text(encoding="utf-8")

    assert "Hiera LayerNorm and Hiera GELU" in notice
    assert "aten/src/ATen/native/cuda/ActivationGeluKernel.cu" in notice
    assert "aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh" in notice
    assert "aten/src/ATen/cuda/detail/UnpackRaw.cuh" in notice
    assert "FlashAttention" in notice
    assert "979702c87a8713a8e0a5e9fee122b90d2ef13be5" in notice
    assert "CUTLASS" in notice
    assert "afa1772203677c5118fcd82537a9c8fefbcc7008" in notice
    assert "BSD 3-Clause License" in notice
    assert "dynamically links libcuBLASLt" in notice
    assert "separately supplied NVIDIA CUDA Toolkit" in notice
    assert "do not redistribute any cuBLASLt binary" in notice


@requires_sam2_hoi
def test_exact_hiera_release_member_closure_is_under_declared_package_roots() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as source:
        package = tomllib.load(source)["tool"]["conan-py-build"]

    assert package["wheel"]["packages"] == ["python/tensorrt_model_connect"]
    assert "python" in package["sdist"]["include"]

    required = _exact_hiera_release_members()
    actual = {
        path.relative_to(SAM2_HOI_NATIVE_PLUGINS)
        for path in SAM2_HOI_NATIVE_PLUGINS.rglob("*")
        if path.is_file()
    }
    assert required == actual
    assert len(required) == 186
    assert len(_manifest_members(SAM2_HOI_NATIVE_PLUGINS / "vendor/flash_attention")) == 15
    assert len(_manifest_members(SAM2_HOI_NATIVE_PLUGINS / "vendor/cutlass")) == 133

    assert SAM2_HOI_PAFPN_BN_HELPER.is_file() and not SAM2_HOI_PAFPN_BN_HELPER.is_symlink()
    assert hashlib.sha256(SAM2_HOI_PAFPN_BN_HELPER.read_bytes()).hexdigest() == (
        "4d0fad825f75412c968764ed2baade5c652963dd956db385c4eed3ce932089c0"
    )


@pytest.mark.slow
@requires_sam2_hoi
def test_built_wheel_and_sdist_contain_exact_hiera_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_override = os.environ.get("TRTMC_RELEASE_WHEEL")
    sdist_override = os.environ.get("TRTMC_RELEASE_SDIST")
    require_artifacts = os.environ.get("TRTMC_REQUIRE_RELEASE_ARCHIVE_TEST") == "1"
    if bool(wheel_override) != bool(sdist_override):
        pytest.fail("TRTMC_RELEASE_WHEEL and TRTMC_RELEASE_SDIST must be supplied together")
    if wheel_override and sdist_override:
        wheel = Path(wheel_override).resolve(strict=True)
        sdist = Path(sdist_override).resolve(strict=True)
    else:
        if importlib.util.find_spec("conan_py_build") is None:
            message = "actual release archive gate requires the pinned conan-py-build backend"
            if require_artifacts:
                pytest.fail(message)
            pytest.skip(message)
        monkeypatch.chdir(REPOSITORY_ROOT)
        wheel_directory = tmp_path / "wheel"
        sdist_directory = tmp_path / "sdist"
        wheel_directory.mkdir()
        sdist_directory.mkdir()
        wheel = wheel_directory / backend.build_wheel(str(wheel_directory))
        sdist = sdist_directory / backend.build_sdist(str(sdist_directory))

    required = _exact_hiera_release_members()
    wheel_root = Path("tensorrt_model_connect/families/sam2_hoi/native_plugins")
    wheel_helper = Path("tensorrt_model_connect/families/sam2_hoi/pafpn_bn_invstd_helper.cu")
    with zipfile.ZipFile(wheel) as archive:
        wheel_payloads = {
            Path(name): archive.read(name) for name in archive.namelist() if not name.endswith("/")
        }
        wheel_names = set(wheel_payloads)
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_payloads = {
            Path(member.name): archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
        sdist_names = set(sdist_payloads)
    sdist_roots = {name.parts[0] for name in sdist_names}
    assert len(sdist_roots) == 1
    sdist_root = Path(next(iter(sdist_roots)))
    sdist_native_root = sdist_root / "python" / wheel_root

    wheel_native_members = {
        name.relative_to(wheel_root) for name in wheel_names if name.is_relative_to(wheel_root)
    }
    sdist_native_members = {
        name.relative_to(sdist_native_root)
        for name in sdist_names
        if name.is_relative_to(sdist_native_root)
    }
    assert wheel_native_members == required
    assert sdist_native_members == required

    for member in required:
        source_payload = (SAM2_HOI_NATIVE_PLUGINS / member).read_bytes()
        wheel_member = wheel_root / member
        assert wheel_payloads[wheel_member] == source_payload
        sdist_member = sdist_root / "python" / wheel_root / member
        assert sdist_payloads[sdist_member] == source_payload

    assert wheel_payloads[wheel_helper] == SAM2_HOI_PAFPN_BN_HELPER.read_bytes()
    sdist_helper = sdist_root / "python" / wheel_helper
    assert sdist_payloads[sdist_helper] == SAM2_HOI_PAFPN_BN_HELPER.read_bytes()

    _assert_manifest_integrity(
        wheel_payloads,
        wheel_root / "vendor/flash_attention/MANIFEST.sha256",
        root=wheel_root / "vendor/flash_attention",
    )
    _assert_manifest_integrity(
        wheel_payloads,
        wheel_root / "vendor/cutlass/MANIFEST.sha256",
        root=wheel_root / "vendor/cutlass",
    )
    _assert_manifest_integrity(
        wheel_payloads,
        wheel_root / "hiera_gelu_bf16_lut_cuda128_exact.MANIFEST.sha256",
        root=wheel_root,
    )

    _validate_legal_payload(wheel)
    for relative in RELEASE_LEGAL_FILES:
        sdist_member = sdist_root / relative
        assert sdist_payloads[sdist_member] == (REPOSITORY_ROOT / relative).read_bytes()
    sdist_metadata = Parser().parsestr(sdist_payloads[sdist_root / "PKG-INFO"].decode())
    assert sdist_metadata["Metadata-Version"] == "2.4"
    assert sdist_metadata["License-Expression"] == "Apache-2.0"
    assert sorted(sdist_metadata.get_all("License-File")) == sorted(RELEASE_LEGAL_FILES)

    cuda_library_prefixes = ("libcuda.so", "libcudart.so", "libcublas.so", "libcublasLt.so")
    assert not any(name.name.startswith(cuda_library_prefixes) for name in wheel_names)
    assert not any(name.name.startswith(cuda_library_prefixes) for name in sdist_names)


@requires_sam2_hoi
def test_exact_hiera_sources_exclude_cuda_dsos_and_gate_build_process_identity() -> None:
    packaged_files = [path for path in SAM2_HOI_NATIVE_PLUGINS.rglob("*") if path.is_file()]
    assert not any(
        path.name.startswith(("libcuda.so", "libcudart.so", "libcublas.so", "libcublasLt.so"))
        for path in packaged_files
    )

    cmake = (SAM2_HOI_NATIVE_PLUGINS / "CMakeLists.txt").read_text(encoding="utf-8")
    builder = (
        REPOSITORY_ROOT / "python/tensorrt_model_connect/families/sam2_hoi/native_plugin_builder.py"
    ).read_text(encoding="utf-8")
    loader = (
        REPOSITORY_ROOT / "python/tensorrt_model_connect/families/sam2_hoi/source_export.py"
    ).read_text(encoding="utf-8")
    normalized_loader = " ".join(loader.split())
    assert "CUDA::cublasLt" in cmake
    assert '"sha256": _file_sha256(expected_path)' in builder
    assert 'Path("/proc/self/maps")' in builder
    assert "_verify_loaded_cublaslt(path, allow_unloaded=True)" in loader
    assert "does not attest the separately deployed C++ inference process" in normalized_loader
    assert "external qualification gate" in normalized_loader
    dependency_load = loader.index("dependency_handle = ctypes.CDLL")
    plugin_load = loader.index("handle = ctypes.CDLL(str(path)")
    assert dependency_load < plugin_load


@requires_sam2_hoi
def test_sam2_hoi_tracker_position_has_sam2_notice_attribution() -> None:
    from tensorrt_model_connect.families.sam2_hoi import (
        native_image_builder,
        source_package,
    )

    builder = (
        REPOSITORY_ROOT / Path(native_image_builder.__file__).relative_to(REPOSITORY_ROOT)
    ).read_text()
    assert "def _add_position_encoding_sine(" in builder
    assert "ElementWiseOperation.POW" in builder
    assert "ElementWiseOperation.DIV" in builder
    assert "UnaryOperation.SIN" in builder
    assert "UnaryOperation.COS" in builder

    with (REPOSITORY_ROOT / "tests/e2e/models/sam2_hoi/MODEL.toml").open("rb") as source:
        archive_sha256 = tomllib.load(source)["model_source_package"]["sha256"]

    notice = (REPOSITORY_ROOT / "NOTICE").read_text()
    separator = "-" * 79
    heading = f"{separator}\nSAM 2\n{separator}\n"
    _, found, remainder = notice.partition(heading)
    assert found, "NOTICE is missing the SAM 2 section"
    section, found, _ = remainder.partition(f"\n{separator}\n")
    assert found, "NOTICE has an unterminated SAM 2 section"

    assert "sam2/modeling/position_encoding.py" in section
    assert "https://github.com/facebookresearch/sam2" in section
    assert source_package.SOURCE_COMMIT in section
    assert archive_sha256 in section
    assert "b51404718c0d38f381293c8e5e00a15d129651b7f09b1158002d8974a30967b5" in section
    assert "Copyright (c) Meta Platforms, Inc. and affiliates." in section
    assert "Apache License, Version 2.0" in section


def test_python_only_editable_wheel_contains_release_legal_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    filename = backend.build_editable(str(tmp_path), {"py-only": "true"})
    wheel = tmp_path / filename

    _validate_legal_payload(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        assert metadata["Metadata-Version"] == "2.4"
        assert metadata["License-Expression"] == "Apache-2.0"
        assert sorted(metadata.get_all("License-File")) == sorted(RELEASE_LEGAL_FILES)


def test_wheel_legal_validator_rejects_missing_file(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata())
    wheel = _synthetic_wheel(
        tmp_path / "missing.whl",
        metadata=metadata,
        omitted="NOTICE",
    )

    with pytest.raises(CiError, match="packaged legal file is missing"):
        _validate_legal_payload(wheel)


def test_wheel_legal_validator_rejects_stale_file(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata())
    wheel = _synthetic_wheel(
        tmp_path / "stale.whl",
        metadata=metadata,
        stale="ASSET_LICENSES.md",
    )

    with pytest.raises(CiError, match="packaged legal file is stale"):
        _validate_legal_payload(wheel)


def test_wheel_legal_validator_rejects_missing_metadata_declaration(tmp_path: Path) -> None:
    metadata = backend._metadata_text(backend._project_metadata()).replace(
        "License-File: NOTICE\n", ""
    )
    wheel = _synthetic_wheel(tmp_path / "metadata.whl", metadata=metadata)

    with pytest.raises(CiError, match="package metadata must declare legal files"):
        _validate_legal_payload(wheel)
