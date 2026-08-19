# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""personaplex model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToSpeechRunner


class PersonaplexSpeechToSpeechRunner(SpeechToSpeechRunner):
    """personaplex local runner for speech_to_speech."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_speech"

runner = PersonaplexSpeechToSpeechRunner()
