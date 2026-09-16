# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build or prepare inputs with Python; execute the packaged native runtime."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    from . import family_cli

    root_help = not arguments or arguments in (["--help"], ["-h"], ["help"])
    declarations = family_cli.discover() if root_help else {}
    if not root_help and re.fullmatch(r"[a-z][a-z0-9_]*", arguments[0]):
        declaration = family_cli.load_family_cli(arguments[0])
        if declaration is not None:
            declarations[arguments[0]] = declaration
    if arguments and arguments[0] in declarations:
        return family_cli.main(arguments, declarations)
    if arguments and arguments[0] in {"build", "prepare-structure"}:
        from .build_cli import main as build_main

        return build_main(arguments)

    native = Path(__file__).resolve().parent / "bin" / "trtmc"
    if root_help:
        for family, declaration in declarations.items():
            names = ", ".join(command["name"] for command in declaration["commands"])
            print(f"Family commands: trtmc {family} {{{names}}} --help", flush=True)
        print("Build: trtmc build MODEL -o model.bundle [OPTIONS]", flush=True)
        print("Build options: trtmc build --help\n", flush=True)
        print("Prepare: trtmc prepare-structure MODEL --input REQUEST -o prepared.request\n", flush=True)
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
