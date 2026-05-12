"""Compatibility shim for the shared ltx_dit_builder implementation."""

import sys as _sys

from .families._shared import ltx_dit_builder as _impl

_sys.modules[__name__] = _impl
