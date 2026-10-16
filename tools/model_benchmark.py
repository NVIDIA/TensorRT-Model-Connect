#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the AIPerf-based TRTMC vs native qualification from a repository checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Sequence


REPOSITORY = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    aiperf = commands.add_parser(
        "aiperf", help="Run the AIPerf-based TRTMC vs native qualification (apps/aiperf_qual run-all)"
    )
    aiperf.add_argument("--environment", type=Path, required=True, help="machine environment file")
    aiperf.add_argument(
        "--aiperf-python", type=Path, required=True, help="interpreter of the AIPerf environment (setup.sh)"
    )
    aiperf.add_argument("--out-root", type=Path, required=True)
    aiperf.add_argument("--model", action="append", default=[], help="profiles (default: the machine's list)")
    aiperf.add_argument("--shard", help="INDEX/COUNT")
    aiperf.add_argument("--rerun", action="store_true")
    return value


def aiperf_command(arguments: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """The ``trtmc-aiperf-qual run-all`` invocation and its PYTHONPATH for this checkout."""
    command = [str(arguments.aiperf_python), "-m", "trtmc_aiperf_qual", "run-all",
               "--environment", str(arguments.environment), "--out-root", str(arguments.out_root)]
    for model in arguments.model:
        command += ["--profile", model]
    if arguments.shard:
        command += ["--shard", arguments.shard]
    if arguments.rerun:
        command.append("--rerun")
    roots = ("apps/aiperf_qual", "apps/perf_serving", "core/builder", "apps/benchmark", ".")
    return command, {"PYTHONPATH": ":".join(str(REPOSITORY / root) for root in roots)}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    command, env = aiperf_command(arguments)
    return subprocess.run(command, env={**os.environ, **env}, cwd=REPOSITORY / "apps/aiperf_qual").returncode


if __name__ == "__main__":
    raise SystemExit(main())
