"""deberta model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class DebertaEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """deberta local runner for encoder_only_nlp."""

runner = DebertaEncoderOnlyNlpRunner()
