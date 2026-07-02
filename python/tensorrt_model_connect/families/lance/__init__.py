# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance family package. Exposes the module-level ``plugin`` for auto-discovery."""
from __future__ import annotations

from .plugin import plugin

__all__ = ["plugin"]
