# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared diffusion contract plugin placeholder.

Diffusion media contract verification is model-owned. The shared plugin
package may import this module during fallback discovery, but it must not
register a contract plugin for concrete reference families.
"""

from __future__ import annotations

plugin = None
