"""Compatibility shim for the phi4_multimodal family-owned phi4mm_vision_builder implementation."""

import sys as _sys

from .families.phi4_multimodal import phi4mm_vision_builder as _impl

_sys.modules[__name__] = _impl
