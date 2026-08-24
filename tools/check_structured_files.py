#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse every tracked JSON, TOML, and YAML file without executing repository code."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml


_STRUCTURED_SUFFIXES = frozenset({".json", ".toml", ".yaml", ".yml"})


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object repeats a key."""


def _reject_duplicate_json_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def validate_file(path: Path) -> str | None:
    """Return a concise parse failure for one supported file, otherwise None."""

    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in {".yaml", ".yml"}:
            list(yaml.safe_load_all(text))
        else:
            return None
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        detail = str(error).splitlines()[0] if str(error) else type(error).__name__
        return f"{path}: {detail}"
    return None


def tracked_structured_files(repository: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(detail or "git ls-files failed")
    names = result.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    return sorted(
        repository / name
        for name in names
        if name and Path(name).suffix in _STRUCTURED_SUFFIXES
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    repository = _parse_args(argv).repo_root.resolve()
    try:
        files = tracked_structured_files(repository)
    except RuntimeError as error:
        print(f"structured-file check failed: {error}", file=sys.stderr)
        return 2
    failures = [failure for path in files if (failure := validate_file(path))]
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print(f"structured files: {len(files)} checked, {len(failures)} failed", file=sys.stderr)
        return 1
    print(f"structured files: {len(files)} checked, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
