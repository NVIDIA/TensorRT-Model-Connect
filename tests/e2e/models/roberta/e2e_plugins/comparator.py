"""roberta model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.encoder_only import EncoderOnlyComparator


class RobertaEncoderOnlyNlpComparator(EncoderOnlyComparator):
    """roberta local comparator for encoder_only_nlp."""

comparator = RobertaEncoderOnlyNlpComparator()
