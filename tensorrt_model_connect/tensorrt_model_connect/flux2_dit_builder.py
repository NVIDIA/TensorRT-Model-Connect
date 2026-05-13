"""Compatibility shim for the flux family-owned flux2_dit_builder implementation."""

import sys as _sys

from .families.flux import flux2_dit_builder as _impl

_sys.modules[__name__] = _impl
