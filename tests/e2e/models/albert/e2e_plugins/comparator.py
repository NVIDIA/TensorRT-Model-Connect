"""albert model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class AlbertEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """albert local comparator for encoder_only_nlp."""

comparator = AlbertEncoderOnlyNlpComparator()
