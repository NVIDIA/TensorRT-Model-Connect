# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the installed build command or packaged native CLI."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

from . import build_cli


def _native_executable() -> Path:
    return Path(__file__).resolve().parent / "bin" / "trtmc"


def main(argv: Sequence[str] | None = None) -> int | None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "build":
        return build_cli.main(arguments)

    executable = _native_executable()
    if not executable.is_file():
        raise FileNotFoundError(f"packaged native trtmc does not exist: {executable}")
    os.execv(str(executable), [str(executable), *arguments])
    return None


if __name__ == "__main__":
    raise SystemExit(main())
