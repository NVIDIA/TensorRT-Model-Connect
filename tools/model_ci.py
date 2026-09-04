#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and select physical model-family ownership."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

try:
    from . import test_impact
except ImportError:
    import test_impact


def matrix(families: Sequence[str]) -> list[dict[str, str]]:
    return [{"family": family} for family in families]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    impact = commands.add_parser("impact")
    impact.add_argument("--base", required=True)
    impact.add_argument("--head", default="HEAD")
    commands.add_parser("all")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        test_impact.validate(repo)
        if args.command == "validate":
            families = test_impact.inventory(repo)
            payload = {"valid": True, "families": list(families)}
        elif args.command == "impact":
            files = test_impact.changed_files(repo, args.base, args.head)
            impact = test_impact.classify(repo, files)
            payload = {**asdict(impact), "matrix": matrix(impact.families)}
        else:
            families = test_impact.inventory(repo)
            payload = {"families": list(families), "matrix": matrix(families)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"model-ci: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
