# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deberta model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class DebertaEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """deberta local comparator for encoder_only_nlp."""

comparator = DebertaEncoderOnlyNlpComparator()
