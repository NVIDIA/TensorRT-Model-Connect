"""whisper model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.speech_to_text import SpeechToTextComparator


class WhisperSpeechToTextComparator(SpeechToTextComparator):
    """whisper local comparator for speech_to_text."""

comparator = WhisperSpeechToTextComparator()
