#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine every family's support.json, unchanged, into one versioned file.

Families that still declare support in Python are not listed until they move
to support.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


def _rule_files(repository: Path) -> list[Path]:
    return sorted(
        path
        for path in (repository / "families").glob("*/support.json")
        if _ID.fullmatch(path.parent.name)
    )


def build_export(repository: Path = REPO, commit: str | None = None) -> dict[str, Any]:
    families: dict[str, Any] = {}
    combined = hashlib.sha256()
    for path in _rule_files(repository):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        families[path.parent.name] = {"sha256": digest, "rules": json.loads(raw)}
        combined.update(f"{path.parent.name}:{digest}\n".encode())
    return {
        "schema_version": SCHEMA_VERSION,
        "rules_version": {"commit": commit, "sha256": combined.hexdigest()},
        "families": families,
    }


def _head_commit(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=REPO)
    parser.add_argument("--commit", help="source commit to record; defaults to git HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export = build_export(args.repository, args.commit or _head_commit(args.repository))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
