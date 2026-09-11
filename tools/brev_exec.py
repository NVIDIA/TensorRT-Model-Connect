# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one remote application through Brev without retrying its exit code."""

from __future__ import annotations

import argparse
import secrets
import shlex
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from tools.ci.process import CiError


def remote_wrapper(command: Sequence[str], result_file: Path, marker: str) -> str:
    """Cache the application result remotely while returning success to Brev."""
    if not command:
        raise ValueError("remote application command must not be empty")
    if not result_file.is_absolute():
        raise ValueError("remote result file must be absolute")
    if not marker or "\n" in marker:
        raise ValueError("remote result marker must be one non-empty line")

    rendered = shlex.join(command)
    quoted_result = shlex.quote(str(result_file))
    quoted_marker = shlex.quote(marker)
    return "\n".join(
        (
            "set -u",
            f"result_file={quoted_result}",
            'if [ -s "$result_file" ]; then',
            '  status="$(cat -- "$result_file")"',
            "else",
            "  set +e",
            f"  {rendered}",
            "  status=$?",
            "  set -e",
            '  temporary="${result_file}.tmp.$$"',
            '  printf \'%s\\n\' "$status" > "$temporary"',
            '  mv -- "$temporary" "$result_file"',
            "fi",
            "case \"$status\" in ''|*[!0-9]*) status=255 ;; esac",
            f"printf '%s%s\\n' {quoted_marker} \"$status\"",
            "exit 0",
        )
    )


def parse_remote_status(lines: Iterable[str], marker: str) -> int:
    """Extract one consistent cached application result from Brev output."""
    statuses: list[int] = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith(marker):
            continue
        value = line.removeprefix(marker)
        if not value.isdigit() or not 0 <= int(value) <= 255:
            raise CiError(f"invalid remote application result: {value!r}")
        statuses.append(int(value))
    if not statuses:
        raise CiError("Brev completed without a remote application result")
    if len(set(statuses)) != 1:
        raise CiError(f"Brev returned inconsistent application results: {statuses}")
    return statuses[0]


def execute(instance: str, command: Sequence[str], log: Path, result_file: Path) -> int:
    """Stream a Brev execution and return the cached application exit code."""
    marker = f"__TRTMC_REMOTE_EXIT_{secrets.token_hex(16)}__="
    wrapper = remote_wrapper(command, result_file, marker)
    log.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            ["brev", "exec", instance, wrapper],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise CiError("Brev output stream was not created")
        for line in process.stdout:
            lines.append(line)
            output.write(line)
            output.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise CiError(f"Brev transport failed after retry handling (exit {return_code})")
    return parse_remote_status(lines, marker)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and preserve the remote application's real conclusion."""
    arguments = _parser().parse_args(argv)
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    try:
        return execute(arguments.instance, command, arguments.log, arguments.result_file)
    except (CiError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
