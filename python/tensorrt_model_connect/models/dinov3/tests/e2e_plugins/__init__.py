# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DINOv3-owned E2E helpers."""

from __future__ import annotations

import os
from pathlib import Path


def image_input(case) -> str:
    return str(
        case.inputs.get("image")
        or case.inputs.get("test_image")
        or case.inputs.get("image_path")
        or ""
    )


def resolve_image_path(case, roots: tuple[Path, ...], error: str) -> str:
    image = image_input(case)
    if not image:
        raise ValueError(error)
    path = Path(image)
    if path.is_absolute():
        return str(path)
    return str(next((root / path for root in roots if (root / path).is_file()), path))


def case_artifact_dir(artifacts_dir: str, case_name: str) -> str:
    path = os.path.join(artifacts_dir, case_name) if case_name else artifacts_dir
    os.makedirs(path, exist_ok=True)
    return path


def save_full_stderr(
    stderr: str,
    artifacts_dir: str,
    stage_name: str,
    case_name: str = "",
) -> tuple[str, str | None]:
    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    directory = case_artifact_dir(artifacts_dir, case_name)
    path = os.path.join(directory, f"{stage_name}_stderr.log")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(stderr)
    return truncated, path
