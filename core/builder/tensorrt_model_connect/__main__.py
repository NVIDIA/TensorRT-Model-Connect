# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build with Python; replace this process with the packaged native runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "build":
        from .build_cli import main as build_main

        return build_main(arguments)

    native = Path(__file__).resolve().parent / "bin" / "trtmc"
    if not arguments or arguments in (["--help"], ["-h"], ["help"]):
        print("Build: trtmc build MODEL -o model.bundle [OPTIONS]", flush=True)
        print("Build options: trtmc build --help\n", flush=True)
        arguments = ["help"]
    if arguments[0] not in {"help", "version", "inspect"} and not any(
        argument == "--runtime-root" or argument.startswith("--runtime-root=")
        for argument in arguments
    ):
        arguments.extend(("--runtime-root", str(native.parent)))
    try:
        os.execv(str(native), [str(native), *arguments])
    except OSError as error:
        print(f"Error: cannot execute packaged trtmc: {error}", file=sys.stderr)
        return 1
    raise AssertionError("execv returned without replacing the process")


if __name__ == "__main__":
    raise SystemExit(main())
