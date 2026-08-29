# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""glmasr model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.speech_to_text import SpeechToTextComparator


class GlmAsrSpeechToTextComparator(SpeechToTextComparator):
    """glmasr local comparator for speech_to_text."""

comparator = GlmAsrSpeechToTextComparator()
