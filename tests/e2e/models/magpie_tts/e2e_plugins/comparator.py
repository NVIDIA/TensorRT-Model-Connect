"""magpie_tts model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.text_to_audio import TextToAudioComparator


class MagpieTtsTextToAudioComparator(TextToAudioComparator):
    """magpie_tts local comparator for text_to_audio."""

comparator = MagpieTtsTextToAudioComparator()
