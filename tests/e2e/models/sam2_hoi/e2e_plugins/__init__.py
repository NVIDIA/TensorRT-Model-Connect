# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 HOI model-owned E2E plugins."""

from __future__ import annotations

from pathlib import Path


def case_artifact_dir(artifacts_dir: str, case_name: str) -> Path:
    root = Path(artifacts_dir or "/tmp/e2e_artifacts")
    path = root / case_name if case_name else root
    path.mkdir(parents=True, exist_ok=True)
    return path
