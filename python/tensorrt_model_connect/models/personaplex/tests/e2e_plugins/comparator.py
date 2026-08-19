# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""personaplex model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.speech_to_speech import SpeechToSpeechComparator


class PersonaplexSpeechToSpeechComparator(SpeechToSpeechComparator):
    """personaplex local comparator for speech_to_speech."""

comparator = PersonaplexSpeechToSpeechComparator()
