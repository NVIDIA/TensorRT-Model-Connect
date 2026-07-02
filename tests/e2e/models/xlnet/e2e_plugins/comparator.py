# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xlnet model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class XlnetEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """xlnet local comparator for encoder_only_nlp."""

comparator = XlnetEncoderOnlyNlpComparator()
