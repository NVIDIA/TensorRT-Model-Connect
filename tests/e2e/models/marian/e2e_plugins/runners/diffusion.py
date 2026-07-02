# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared diffusion runner placeholder.

Concrete diffusion media execution is owned by
``tests/e2e/models/<family>/e2e_plugins/runners/diffusion.py``. The shared
registry may import this module during fallback discovery, but it must not
register model-family behavior.
"""

from __future__ import annotations

plugin = None
