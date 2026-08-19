# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bark model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text_to_audio import TextToAudioComparator


class BarkTextToAudioComparator(TextToAudioComparator):
    """bark local comparator for text_to_audio."""

comparator = BarkTextToAudioComparator()
