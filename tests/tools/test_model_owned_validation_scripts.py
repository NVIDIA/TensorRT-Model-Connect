# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for model-owned validation entrypoints.

Trace: ARCH-MODPLUG-001
Intent: keep developer validation scripts aligned with model-local E2E tests.
Preconditions: validation scripts are present in the repository.
Postconditions: family validation runs model-owned pytest nodes and exposes
isolated model-plugin validation.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_validate_family_uses_model_owned_e2e_entrypoint() -> None:
    """Trace: ARCH-MODPLUG-001
    Intent: prevent validate_family.sh from scheduling the shared E2E test node.
    Preconditions: scripts/validate_family.sh exists.
    Postconditions: the script builds python/tensorrt_model_connect/models/<family> node ids.
    """
    text = (REPO_ROOT / "scripts" / "validate_family.sh").read_text(encoding="utf-8")

    assert "tests/test_e2e.py::test_e2e" not in text
    assert (
        "python/tensorrt_model_connect/models/${E2E_FAMILY}/tests/"
        "test_${E2E_FAMILY}_e2e.py"
    ) in text
    assert "--model-plugin-dir" in text
    assert "--isolate-model-plugin" in text
    assert 'project / "python" / "tensorrt_model_connect" / "models"' in text
    assert 'library = f"libtrtmc_model_{model_id}.so"' in text
    assert 'project / "src" / "runtime" / "models"' not in text
    assert "runtime_library" not in text
