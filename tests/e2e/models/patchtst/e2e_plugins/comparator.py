"""patchtst model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.neural_operator import NeuralOperatorComparator


class PatchtstNeuralOperatorComparator(NeuralOperatorComparator):
    """patchtst local comparator for neural_operator."""

comparator = PatchtstNeuralOperatorComparator()
