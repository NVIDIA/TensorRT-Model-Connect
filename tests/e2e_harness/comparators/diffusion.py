# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared diffusion comparator placeholder.

Diffusion media comparison policy is model-owned under
``python/tensorrt_model_connect/models/<family>/e2e_plugins/comparators/diffusion.py``. Generic
metric helpers may remain in sibling helper modules, but this module must not
register concrete comparison behavior.
"""

from __future__ import annotations

plugin = None
