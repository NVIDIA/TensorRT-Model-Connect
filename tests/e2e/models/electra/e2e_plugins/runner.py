"""electra model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.encoder_only import EncoderOnlyRunner


class ElectraEncoderOnlyNlpRunner(EncoderOnlyRunner):
    """electra local runner for encoder_only_nlp."""

runner = ElectraEncoderOnlyNlpRunner()
