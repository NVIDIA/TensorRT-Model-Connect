# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU isolation fixtures for E2E tests.

These fixtures ensure clean GPU state between tests and early termination
on unrecoverable CUDA errors.
"""

import pytest


@pytest.fixture(autouse=True)
def gpu_memory_cleanup():
    """Clear CUDA cache before and after each test.

    Intention:
        When running multiple E2E tests sequentially, GPU memory from previous
        tests may not be freed, causing OOM errors on later tests. This fixture
        ensures each test starts with a clean GPU state.

    Setup:
        Try to import torch and call torch.cuda.empty_cache() before and after
        the test. If torch is not available, this is a no-op.
    """
    _clear_cuda_cache()
    yield
    _clear_cuda_cache()


@pytest.fixture(autouse=True)
def cuda_error_early_quit():
    """Detect unrecoverable CUDA errors and fail fast.

    Intention:
        When a CUDA error occurs (e.g., device-side assert, out of memory that
        corrupts the CUDA context), subsequent tests will also fail with cryptic
        errors. This fixture detects such conditions and skips remaining tests.
    """
    yield
    _check_cuda_health()


def _clear_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


def _check_cuda_health():
    try:
        import torch
        if torch.cuda.is_available():
            # A simple operation to check CUDA context is still valid
            torch.tensor([1.0], device="cuda")
    except RuntimeError as e:
        if "CUDA" in str(e).upper():
            pytest.exit(f"Unrecoverable CUDA error detected: {e}", returncode=1)
    except ImportError:
        pass
