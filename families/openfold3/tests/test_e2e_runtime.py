# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from . import test_e2e as e2e


@pytest.fixture(scope="module")
def qualification_builds(tmp_path_factory):
    root = tmp_path_factory.mktemp("openfold3-builds")
    repository = Path(__file__).resolve().parents[3]
    sources = Path(__file__).with_name("cpp")
    compiler = shutil.which("c++")
    assert compiler, "OpenFold3 runtime regression requires a C++ compiler"
    flags = [
        compiler,
        "-std=c++17",
        f"-I{repository / 'core/runtime/include'}",
        f"-I{repository / 'families/openfold3/include'}",
    ]
    native, wheel = root / "native-build", root / "wheel"
    for build_id, directory in enumerate((native, wheel), start=1):
        directory.mkdir()
        for name in ("core", "backend_trt"):
            library = f"libtrtmc_{name}.so"
            subprocess.run(
                [
                    *flags,
                    "-shared",
                    "-fPIC",
                    f"-DTEST_BUILD_ID={build_id}",
                    str(sources / "fake_build_identity.cpp"),
                    f"-Wl,-soname,{library}",
                    "-o",
                    str(directory / library),
                ],
                check=True,
            )
        subprocess.run(
            [
                *flags,
                "-shared",
                "-fPIC",
                f"-DTEST_BUILD_ID={build_id}",
                str(sources / "fake_qualification_runtime.cpp"),
                f"-L{directory}",
                "-ltrtmc_core",
                "-ldl",
                "-Wl,-soname,libtrtmc_runtime.so",
                "-Wl,-rpath,$ORIGIN",
                "-o",
                str(directory / "libtrtmc_runtime.so"),
            ],
            check=True,
        )
    qualification = native / "openfold3_qualification"
    subprocess.run(
        [
            *flags,
            str(sources / "qualification.cpp"),
            f"-L{native}",
            "-ltrtmc_runtime",
            f"-Wl,-rpath,{native}",
            f"-Wl,-rpath-link,{native}",
            "-o",
            str(qualification),
        ],
        check=True,
    )
    return qualification, wheel


@pytest.mark.parametrize("layout", ("copied", "symlinked"))
def test_qualification_uses_selected_product_build(
    qualification_builds, monkeypatch, tmp_path: Path, layout: str
) -> None:
    qualification, wheel = qualification_builds
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    for name in ("libtrtmc_core.so", "libtrtmc_backend_trt.so", "libtrtmc_runtime.so"):
        if layout == "copied":
            shutil.copy2(wheel / name, runtime_root / name)
        elif name != "libtrtmc_runtime.so":
            (runtime_root / name).symlink_to(wheel / name)
    # The source-built executable's RUNPATH and the inherited environment both
    # point at a different product build from the selected wheel libraries.
    inherited_path = str(qualification.parent)
    monkeypatch.setenv("LD_LIBRARY_PATH", inherited_path)
    request = tmp_path / "query.json"
    request.write_text("{}", encoding="utf-8")

    structure, metadata = e2e._run_native(
        qualification, runtime_root, tmp_path / "model.bundle", request, tmp_path, 0, 30
    )

    assert structure == "data_test\n"
    assert metadata == {"build_id": 2}
    assert os.environ["LD_LIBRARY_PATH"] == inherited_path


def test_qualification_rejects_a_missing_companion_loader(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    (runtime_root / "libtrtmc_core.so").touch()

    with pytest.raises(AssertionError, match="missing its companion loader"):
        e2e._run_native(
            tmp_path / "qualification",
            runtime_root,
            tmp_path / "model.bundle",
            tmp_path / "query.json",
            tmp_path,
            0,
            30,
        )
