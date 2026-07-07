# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inactive segmentation runner sidecar.

This model family does not own segmentation E2E behavior. Semantic segmentation
lives under segformer; prompted segmentation lives under sam and sam3.
"""

from __future__ import annotations

plugin = None
