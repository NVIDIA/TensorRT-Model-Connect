"""chronos_bolt model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.neural_operator import NeuralOperatorComparator


class ChronosBoltNeuralOperatorComparator(NeuralOperatorComparator):
    """chronos_bolt local comparator for neural_operator."""

comparator = ChronosBoltNeuralOperatorComparator()
