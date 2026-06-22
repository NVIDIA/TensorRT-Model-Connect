"""modernbert model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class ModernbertEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """modernbert local comparator for encoder_only_nlp."""

comparator = ModernbertEncoderOnlyNlpComparator()
