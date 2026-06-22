"""magpie_tts model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.audio_speech import TextToAudioRunner


class MagpieTtsTextToAudioRunner(TextToAudioRunner):
    """magpie_tts local runner for text_to_audio."""

runner = MagpieTtsTextToAudioRunner()
