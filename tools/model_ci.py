#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and select physical model-family ownership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import test_impact


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        test_impact.validate(repo)
        families = test_impact.inventory(repo)
        payload = {"valid": True, "families": list(families)}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print(f"model-ci: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
