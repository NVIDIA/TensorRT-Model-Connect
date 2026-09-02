# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parakeet_tdt model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.parakeet_tdt_speech import SpeechToTextRunner


class ParakeetTDTSpeechToTextRunner(SpeechToTextRunner):
    """parakeet_tdt local runner for speech_to_text."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_text"

runner = ParakeetTDTSpeechToTextRunner()
