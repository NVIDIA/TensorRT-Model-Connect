# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ci.package import (
    _validate_archive_sam2_hoi_release_dso,
    validate_sam2_hoi_release_dso,
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

    validate_sam2_hoi_release_dso(context, "fixture wheel", dso)

    assert [command[1:3] for command in context.commands] == [
        ["--wide", "-d"],
        ["--wide", "--dyn-syms"],
    ]


def test_sam2_hoi_release_dso_rejects_external_libjpeg_needed(tmp_path: Path) -> None:
    context = _ReadelfContext(dynamic="0x0000000000000001 (NEEDED) Shared library: [libjpeg.so.8]")
    dso = tmp_path / "libtrtmc_model_sam2_hoi.so"
    dso.write_bytes(b"\x7fELFfixture")

    with pytest.raises(CiError, match="external libjpeg DT_NEEDED.*libjpeg.so.8"):
        validate_sam2_hoi_release_dso(context, "fixture wheel", dso)


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
        validate_sam2_hoi_release_dso(context, "fixture wheel", dso)


def test_wheel_payload_validation_uses_readelf_for_sam2_hoi() -> None:
    context = _ReadelfContext()

    _validate_archive_sam2_hoi_release_dso(context, "fixture wheel", b"\x7fELFfixture")

    assert len(context.commands) == 2
    assert all(
        Path(command[-1]).name == "libtrtmc_model_sam2_hoi.so" for command in context.commands
    )
