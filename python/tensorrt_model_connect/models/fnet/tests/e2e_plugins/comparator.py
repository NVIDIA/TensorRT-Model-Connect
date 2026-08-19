# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fnet model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class FnetEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """fnet local comparator for encoder_only_nlp."""

comparator = FnetEncoderOnlyNlpComparator()
