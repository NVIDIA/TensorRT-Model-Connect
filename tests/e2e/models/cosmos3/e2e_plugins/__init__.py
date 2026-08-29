# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-owned E2E plugin helpers."""

from __future__ import annotations

from pathlib import Path


def save_full_stderr(
    stderr: str,
    artifacts_dir: str,
    stage_name: str,
    case_name: str = "",
) -> tuple[str, str | None]:
    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    directory = Path(artifacts_dir) / case_name if case_name else Path(artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stage_name}_stderr.log"
    path.write_text(stderr, encoding="utf-8")
    return truncated, str(path)
