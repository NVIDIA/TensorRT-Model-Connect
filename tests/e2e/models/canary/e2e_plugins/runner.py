"""canary model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToTextRunner


class CanarySpeechToTextRunner(SpeechToTextRunner):
    """canary local runner for speech_to_text."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_text"

runner = CanarySpeechToTextRunner()
