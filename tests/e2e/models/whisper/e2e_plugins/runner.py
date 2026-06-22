"""whisper model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import SpeechToTextRunner


class WhisperSpeechToTextRunner(SpeechToTextRunner):
    """whisper local runner for speech_to_text."""

runner = WhisperSpeechToTextRunner()
