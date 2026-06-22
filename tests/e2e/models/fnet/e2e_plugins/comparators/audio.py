"""Audio/speech comparator umbrella module.

Re-exports and explicitly registers all audio/speech comparators:
- SpeechToTextComparator (speech_to_text)
- TextToAudioComparator (text_to_audio)
- SpeechToSpeechComparator (speech_to_speech)

Each comparator also lives in its own module for independent auto-discovery,
but this umbrella allows `from e2e_harness.comparators.audio import *`.
"""

from __future__ import annotations

from .speech_to_text import SpeechToTextComparator
from .text_to_audio import TextToAudioComparator
from .speech_to_speech import SpeechToSpeechComparator

__all__ = [
    "SpeechToTextComparator",
    "TextToAudioComparator",
    "SpeechToSpeechComparator",
]

# Primary plugin for auto-discovery (SpeechToTextComparator)
plugin = SpeechToTextComparator()

# Explicit registration of all three comparators
_tta = TextToAudioComparator()
_s2s = SpeechToSpeechComparator()


def _register_extra_comparators() -> None:
    """Register the additional comparators that share this umbrella module."""
    try:
        from ..registry import register_comparator
        register_comparator(_tta)
        register_comparator(_s2s)
    except ImportError:
        pass


_register_extra_comparators()
