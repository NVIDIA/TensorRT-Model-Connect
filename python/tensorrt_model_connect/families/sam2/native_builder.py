# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic discovery of the opt-in SAM2 native builder.

The supported package location is
``tensorrt_model_connect/bin/sam2_native_builder``.  A deployment that keeps
the CMake install tree separate from the Python package may instead set
``TRTMC_SAM2_NATIVE_BUILDER`` to the builder's absolute path.  Discovery does
not search ``PATH`` or guess a build directory.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Mapping


NATIVE_BUILDER_ENV = "TRTMC_SAM2_NATIVE_BUILDER"
NATIVE_BUILDER_FILENAME = "sam2_native_builder"
NATIVE_BUILDER_INSTALL_COMPONENT = "sam2_native_builder"


class Sam2NativeBuilderError(RuntimeError):
    """The opt-in SAM2 native builder cannot be used safely."""


def _require_regular_executable(path: Path, *, source: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Sam2NativeBuilderError(
            f"SAM2 native builder from {source} is missing: {path}"
        ) from exc
    except OSError as exc:
        raise Sam2NativeBuilderError(
            f"cannot inspect SAM2 native builder from {source}: {path}: {exc}"
        ) from exc

    if stat.S_ISLNK(metadata.st_mode):
        raise Sam2NativeBuilderError(
            f"SAM2 native builder from {source} must not be a symlink: {path}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise Sam2NativeBuilderError(
            f"SAM2 native builder from {source} is not a regular file: {path}"
        )
    if not os.access(path, os.X_OK):
        raise Sam2NativeBuilderError(f"SAM2 native builder from {source} is not executable: {path}")
    return path


def locate_native_builder(
    *,
    environ: Mapping[str, str] | None = None,
    package_root: Path | None = None,
) -> Path:
    """Return the exact usable SAM2 native-builder path or fail closed.

    An explicit environment override is authoritative and must be absolute; an
    invalid override never falls back to a different executable.  Without an
    override, only the installed package-owned binary is accepted.
    """

    environment = os.environ if environ is None else environ
    configured = environment.get(NATIVE_BUILDER_ENV, "")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            raise Sam2NativeBuilderError(
                f"{NATIVE_BUILDER_ENV} must name an absolute path: {configured!r}"
            )
        return _require_regular_executable(candidate, source=NATIVE_BUILDER_ENV)

    root = Path(__file__).resolve().parents[2] if package_root is None else Path(package_root)
    candidate = root / "bin" / NATIVE_BUILDER_FILENAME
    try:
        return _require_regular_executable(candidate, source="the installed Python package")
    except Sam2NativeBuilderError as exc:
        raise Sam2NativeBuilderError(
            f"{exc}. Install the opt-in CMake component "
            f"{NATIVE_BUILDER_INSTALL_COMPONENT!r}, or set {NATIVE_BUILDER_ENV} "
            "to its absolute path"
        ) from exc
