"""patchtsmixer model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.neural_operator import NeuralOperatorComparator


class PatchtsmixerNeuralOperatorComparator(NeuralOperatorComparator):
    """patchtsmixer local comparator for neural_operator."""

comparator = PatchtsmixerNeuralOperatorComparator()
