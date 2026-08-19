# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B model-local E2E plugins."""

from __future__ import annotations

from pathlib import Path


def save_full_stderr(
    stderr: str,
    artifacts_dir: str,
    stage_name: str,
    case_name: str = "",
) -> tuple[str, str | None]:
    """Persist complete stderr while keeping result payloads compact."""

    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    output_dir = Path(artifacts_dir)
    if case_name:
        output_dir /= case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage_name}_stderr.log"
    path.write_text(stderr, encoding="utf-8")
    return truncated, str(path)
