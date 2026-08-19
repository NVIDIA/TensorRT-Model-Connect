# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci.package import (
    _validate_archive_private_static_runtime_dso,
    runtime_dso_private_static_policies,
    validate_private_static_runtime_dso,
)
from tools.ci.process import CiError


class _ReadelfContext:
    def __init__(self, *, dynamic: str = "", symbols: str = "") -> None:
        self.dynamic = dynamic
        self.symbols = symbols
        self.commands: list[list[object]] = []

    def output(self, command: list[object]) -> str:
        self.commands.append(command)
        return self.dynamic if "-d" in command else self.symbols


def test_sam2_hoi_release_dso_accepts_private_static_libjpeg(tmp_path: Path) -> None:
    context = _ReadelfContext(
        dynamic="""
 0x0000000000000001 (NEEDED) Shared library: [libtrtmc_core.so]
 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]
""",
        symbols="""
Symbol table '.dynsym' contains 2 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000001000    32 FUNC    GLOBAL DEFAULT   12 trtmc_register_model
""",
    )
    dso = tmp_path / "libtrtmc_model_sam2_hoi.so"
    dso.write_bytes(b"\x7fELFfixture")

    validate_private_static_runtime_dso(context, "fixture wheel", dso, ("jpeg",))

    assert [command[1:3] for command in context.commands] == [
        ["--wide", "-d"],
        ["--wide", "--dyn-syms"],
    ]


def test_sam2_hoi_release_dso_rejects_external_libjpeg_needed(tmp_path: Path) -> None:
    context = _ReadelfContext(dynamic="0x0000000000000001 (NEEDED) Shared library: [libjpeg.so.8]")
    dso = tmp_path / "libtrtmc_model_sam2_hoi.so"
    dso.write_bytes(b"\x7fELFfixture")

    with pytest.raises(CiError, match="external libjpeg DT_NEEDED.*libjpeg.so.8"):
        validate_private_static_runtime_dso(context, "fixture wheel", dso, ("jpeg",))


def test_sam2_hoi_release_dso_rejects_global_default_jpeg_exports(
    tmp_path: Path,
) -> None:
    context = _ReadelfContext(
        symbols="""
Symbol table '.dynsym' contains 2 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
    17: 0000000000002000   128 FUNC    GLOBAL DEFAULT   13 jpeg_std_error@@LIBJPEG_8.0
"""
    )
    dso = tmp_path / "libtrtmc_model_sam2_hoi.so"
    dso.write_bytes(b"\x7fELFfixture")

    with pytest.raises(CiError, match="global/default libjpeg symbols.*jpeg_std_error"):
        validate_private_static_runtime_dso(context, "fixture wheel", dso, ("jpeg",))


def test_wheel_payload_validation_uses_readelf_for_sam2_hoi() -> None:
    context = _ReadelfContext()

    _validate_archive_private_static_runtime_dso(
        context,
        "fixture wheel",
        "libtrtmc_model_fixture.so",
        b"\x7fELFfixture",
        ("jpeg",),
    )

    assert len(context.commands) == 2
    assert all(
        Path(command[-1]).name == "libtrtmc_model_fixture.so" for command in context.commands
    )


def test_private_static_policy_is_discovered_from_runtime_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "src/runtime/models/fixture/MODEL.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        'id = "fixture"\n'
        'runtime_library = "libtrtmc_model_fixture.so"\n'
        'runtime_link_libraries = ["jpeg"]\n'
        'runtime_private_static_libraries = ["jpeg"]\n',
        encoding="utf-8",
    )

    assert runtime_dso_private_static_policies(tmp_path) == {
        "libtrtmc_model_fixture.so": ("jpeg",)
    }


@pytest.mark.parametrize(
    "private_libraries",
    ['["jpeg", "jpeg"]', '["jpeg"]'],
)
def test_private_static_policy_rejects_duplicate_or_unlinked_libraries(
    tmp_path: Path, private_libraries: str
) -> None:
    manifest = tmp_path / "src/runtime/models/fixture/MODEL.toml"
    manifest.parent.mkdir(parents=True)
    linked = "[]" if private_libraries == '["jpeg"]' else '["jpeg"]'
    manifest.write_text(
        'id = "fixture"\n'
        'runtime_library = "libtrtmc_model_fixture.so"\n'
        f"runtime_link_libraries = {linked}\n"
        f"runtime_private_static_libraries = {private_libraries}\n",
        encoding="utf-8",
    )

    with pytest.raises(CiError, match="invalid runtime_private_static_libraries"):
        runtime_dso_private_static_policies(tmp_path)
