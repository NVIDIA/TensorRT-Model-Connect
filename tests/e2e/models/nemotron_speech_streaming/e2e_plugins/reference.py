"""nemotron_speech_streaming model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class NemotronSpeechStreamingHfTransformersReference(HfTransformersReference):
    """nemotron_speech_streaming local reference for hf_transformers."""

reference = NemotronSpeechStreamingHfTransformersReference()
