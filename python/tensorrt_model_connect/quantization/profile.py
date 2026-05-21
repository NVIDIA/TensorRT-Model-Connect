"""Quantization profile — per-model quantization configuration.

Combines a QuantFormat, per-layer scales, and exclusion patterns into a
single object that determines which layers get quantized and how.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

from .formats import QuantFormat
from .scales import QuantScaleMap


def _matches_exclude_pattern(weight_name: str, pattern: str) -> bool:
    if fnmatch.fnmatch(weight_name, pattern):
        return True
    if "/" in weight_name:
        _, suffix = weight_name.split("/", 1)
        return fnmatch.fnmatch(suffix, pattern)
    return False


@dataclass
class QuantProfile:
    """Complete quantization configuration for one model build."""

    format: QuantFormat
    scale_map: QuantScaleMap
    exclude_patterns: list[str] = field(default_factory=list)

    def should_quantize(self, weight_name: str) -> bool:
        """Return True if this weight should be quantized.

        A weight is quantized if:
        1. It is NOT matched by any exclude pattern.
        2. It has scales in the scale_map, OR the scale_map is dynamic.
        """
        for pattern in self.exclude_patterns:
            if _matches_exclude_pattern(weight_name, pattern):
                return False
        if self.scale_map.dynamic:
            return True
        return self.scale_map.get(weight_name) is not None
