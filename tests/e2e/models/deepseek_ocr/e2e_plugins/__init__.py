# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-local E2E plugin package.

Concrete runner, comparator, and reference implementations are copied into
this package so a model test does not import central concrete E2E strategies.
"""

from __future__ import annotations

import os


def _case_artifact_dir(artifacts_dir: str, case_name: str) -> str:
    if case_name:
        d = os.path.join(artifacts_dir, case_name)
    else:
        d = artifacts_dir
    os.makedirs(d, exist_ok=True)
    return d


def save_full_stderr(stderr: str, artifacts_dir: str, stage_name: str, case_name: str = "") -> tuple:
    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    d = _case_artifact_dir(artifacts_dir, case_name)
    path = os.path.join(d, f"{stage_name}_stderr.log")
    with open(path, "w", encoding="utf-8") as f:
        f.write(stderr)
    return truncated, path
