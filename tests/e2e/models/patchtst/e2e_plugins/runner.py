"""patchtst model-owned E2E runner plugins."""

from __future__ import annotations

from .runners.neural_operator import NeuralOperatorRunner


class PatchtstNeuralOperatorRunner(NeuralOperatorRunner):
    """patchtst local runner for neural_operator."""

runner = PatchtstNeuralOperatorRunner()
