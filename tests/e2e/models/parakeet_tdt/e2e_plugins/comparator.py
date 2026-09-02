# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parakeet_tdt model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.parakeet_tdt_asr import SpeechToTextComparator


class ParakeetTDTSpeechToTextComparator(SpeechToTextComparator):
    """parakeet_tdt local comparator for speech_to_text."""

comparator = ParakeetTDTSpeechToTextComparator()
