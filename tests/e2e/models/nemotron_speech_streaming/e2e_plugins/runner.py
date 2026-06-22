"""nemotron_speech_streaming model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToTextRunner


class NemotronSpeechStreamingSpeechToTextRunner(SpeechToTextRunner):
    """nemotron_speech_streaming local runner for speech_to_text."""

runner = NemotronSpeechStreamingSpeechToTextRunner()
