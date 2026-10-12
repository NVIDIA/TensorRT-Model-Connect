# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result-local artifact paths, independent of the machine that ran the benchmark."""

from pathlib import Path


def work_directory(out: Path) -> Path:
    """New runs keep their working files together; old runs remain readable in place."""
    work = out / "artifacts"
    return work if work.is_dir() else out


def recorded_path(out: Path, path: Path) -> str:
    resolved = path.resolve()
    root = out.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def resolve_path(out: Path, recorded: str) -> Path:
    """Resolve a result-relative reference; absolute references from old reports are unchanged."""
    path = Path(recorded)
    if path.is_absolute():
        return path
    resolved = (out / path).resolve()
    if not resolved.is_relative_to(out.resolve()):
        raise ValueError(f"artifact path escapes the model result directory: {recorded!r}")
    return resolved
