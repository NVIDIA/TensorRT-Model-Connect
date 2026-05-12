"""Compatibility shim for the shared phi4mm_vision_builder implementation."""

import sys as _sys

from .families._shared import phi4mm_vision_builder as _impl

_sys.modules[__name__] = _impl
