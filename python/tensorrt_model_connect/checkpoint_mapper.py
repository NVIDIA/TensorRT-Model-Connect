# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable weight container type shared by family-owned builders."""

from __future__ import annotations

__all__ = ["WeightDict"]


class WeightDict(dict):
    """Mapping from logical weight names to arrays used during a family build."""
