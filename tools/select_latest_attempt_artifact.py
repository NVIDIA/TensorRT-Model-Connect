# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the newest valid GitHub artifact from one workflow run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--max-attempt", type=int, required=True)
    parser.add_argument("--required-glob", required=True)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args()


def select_latest_attempt(
    parts_dir: Path,
    artifact_prefix: str,
    max_attempt: int,
    required_glob: str,
) -> tuple[int, Path, list[Path]]:
    if not parts_dir.is_dir():
        raise ValueError(f"artifact parts directory does not exist: {parts_dir}")
    if not artifact_prefix or any(character in artifact_prefix for character in "\r\n"):
        raise ValueError("artifact prefix must be a non-empty single-line string")
    if max_attempt < 1:
        raise ValueError("max attempt must be positive")
    if not required_glob or Path(required_glob).is_absolute():
        raise ValueError("required glob must be a non-empty relative pattern")

    name_pattern = re.compile(re.escape(artifact_prefix) + r"(?P<attempt>[1-9][0-9]*)")
    candidates: list[tuple[int, Path, list[Path]]] = []
    for artifact_dir in sorted(path for path in parts_dir.iterdir() if path.is_dir()):
        match = name_pattern.fullmatch(artifact_dir.name)
        if match is None:
            raise ValueError(f"unexpected artifact directory: {artifact_dir.name}")
        attempt = int(match.group("attempt"))
        if attempt > max_attempt:
            raise ValueError(
                f"artifact attempt {attempt} exceeds current attempt {max_attempt}: "
                f"{artifact_dir.name}"
            )
        required_files = sorted(
            path for path in artifact_dir.glob(required_glob) if path.is_file()
        )
        if not required_files:
            raise ValueError(
                f"artifact {artifact_dir.name} has no file matching {required_glob!r}"
            )
        candidates.append((attempt, artifact_dir.resolve(), required_files))

    if not candidates:
        raise ValueError(f"no artifacts matched prefix {artifact_prefix!r}")
    return max(candidates, key=lambda candidate: candidate[0])


def main() -> int:
    args = _parse_args()
    attempt, selected_dir, required_files = select_latest_attempt(
        args.parts_dir,
        args.artifact_prefix,
        args.max_attempt,
        args.required_glob,
    )
    payload = {
        "selected_attempt": attempt,
        "selected_dir": str(selected_dir),
        "selected_file": str(required_files[0].resolve()),
        "required_file_count": len(required_files),
    }
    if args.github_output is not None:
        for value in payload.values():
            if any(character in str(value) for character in "\r\n"):
                raise ValueError("artifact selection output contains a newline")
        with args.github_output.open("a", encoding="utf-8") as stream:
            for key, value in payload.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
