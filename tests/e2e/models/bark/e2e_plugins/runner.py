"""bark model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import TextToAudioRunner


class BarkTextToAudioRunner(TextToAudioRunner):
    """bark local runner for text_to_audio."""

runner = BarkTextToAudioRunner()
