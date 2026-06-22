"""qwen3_omni model-owned E2E comparator plugins."""

from __future__ import annotations

from .comparators.omni import OmniComparator


class Qwen3OmniMultimodalComparator(OmniComparator):
    """qwen3_omni local comparator for omni_multimodal."""


comparator = Qwen3OmniMultimodalComparator()
