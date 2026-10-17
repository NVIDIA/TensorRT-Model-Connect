# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource configuration for the original FP32 eager family reference."""

import os


def pytest_configure(config):
    # The eager reference's varying allocations fragment a nearly full GPU.
    # Keep its precision/operators unchanged and preserve caller configuration.
    if "PYTORCH_ALLOC_CONF" not in os.environ:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
