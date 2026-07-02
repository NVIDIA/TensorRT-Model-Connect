# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""whisper model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToTextRunner


class WhisperSpeechToTextRunner(SpeechToTextRunner):
    """whisper local runner for speech_to_text."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_text"

runner = WhisperSpeechToTextRunner()
