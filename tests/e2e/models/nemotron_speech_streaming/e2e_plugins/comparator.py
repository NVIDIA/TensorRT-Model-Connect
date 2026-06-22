"""nemotron_speech_streaming model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.speech_to_text import SpeechToTextComparator


class NemotronSpeechStreamingSpeechToTextComparator(SpeechToTextComparator):
    """nemotron_speech_streaming local comparator for speech_to_text."""

comparator = NemotronSpeechStreamingSpeechToTextComparator()
