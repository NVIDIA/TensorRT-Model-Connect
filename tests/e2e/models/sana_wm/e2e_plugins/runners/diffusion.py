# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inactive generated diffusion runner sidecar for SANA-WM.

SANA-WM owns its world-model runner in ``e2e_plugins/runner.py``. This
generated task sidecar remains inert so generic diffusion behavior is not
misclassified as shared ownership.
"""

from __future__ import annotations

plugin = None
