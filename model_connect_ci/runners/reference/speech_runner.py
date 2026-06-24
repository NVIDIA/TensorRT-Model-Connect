"""Speech and multimodal reference runner abstractions."""

from __future__ import annotations

from model_connect_ci.runners.reference.text_runner import (
    ReferenceRunRequest,
    ReferenceRunResult,
)


class SpeechReferenceRunner:
    """Placeholder speech reference runner contract."""

    task_strategy = "speech_to_text"

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        raise NotImplementedError("Speech reference execution is provided by tests/e2e_harness")


class MultimodalReferenceRunner:
    """Placeholder multimodal reference runner contract."""

    task_strategy = "vision_language_generation"

    def run(self, request: ReferenceRunRequest) -> ReferenceRunResult:
        raise NotImplementedError("Multimodal reference execution is provided by tests/e2e_harness")
