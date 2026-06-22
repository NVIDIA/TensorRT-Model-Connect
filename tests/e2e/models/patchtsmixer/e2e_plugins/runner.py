"""patchtsmixer model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.neural_operator import NeuralOperatorRunner


class PatchtsmixerNeuralOperatorRunner(NeuralOperatorRunner):
    """patchtsmixer local runner for neural_operator."""

runner = PatchtsmixerNeuralOperatorRunner()
