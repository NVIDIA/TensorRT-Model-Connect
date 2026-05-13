"""Compatibility shim for the pixart family-owned standard_dit_builder implementation."""

import sys as _sys

from .families.pixart import standard_dit_builder as _impl

_sys.modules[__name__] = _impl
