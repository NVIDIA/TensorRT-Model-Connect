# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned offline preparation shared by hardware and named E2E tests."""

import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def dlrm_bridge_root(request):
    runtime = os.environ.get("TRTMC_RUNTIME_ROOT")
    if not runtime:
        selected = (
            request.config.getoption("--e2e-model")
            or request.config.getoption("--e2e-testcase")
            or os.environ.get("HSTU_E2E_ROOT")
        )
        if selected:
            pytest.fail("selected DLRMv3 proof requires TRTMC_RUNTIME_ROOT")
        pytest.skip("select a native runtime for HSTU hardware tests")
    # CI isolates Task DSOs but keeps family transport libraries in the native build.
    return Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", runtime))


@pytest.fixture(scope="session")
def dlrm_root(tmp_path_factory, dlrm_bridge_root):
    supplied = os.environ.get("HSTU_E2E_ROOT")
    if supplied:
        return Path(supplied)
    checkpoint = os.environ.get("DLRMV3_CHECKPOINT")
    if not checkpoint:
        staged = os.environ.get("TRTMC_HSTU_MODEL_DIR")
        assert staged, (
            "stage manifest external_files and set TRTMC_HSTU_MODEL_DIR, or set DLRMV3_CHECKPOINT"
        )
        checkpoint = str(Path(staged) / "non_sparse.ckpt")
    assert Path(checkpoint).exists(), "stage the official DLRMv3 checkpoint before offline E2E"
    from .prepare import prepare

    root = tmp_path_factory.mktemp("dlrmv3-prepared")
    prepare(root, checkpoint, os.environ["TRTMC_RUNTIME_ROOT"], repositories=False)
    return root / "output"
