"""personaplex model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToSpeechRunner


class PersonaplexSpeechToSpeechRunner(SpeechToSpeechRunner):
    """personaplex local runner for speech_to_speech."""

    @property
    def strategy_name(self) -> str:
        return "speech_to_speech"

runner = PersonaplexSpeechToSpeechRunner()
