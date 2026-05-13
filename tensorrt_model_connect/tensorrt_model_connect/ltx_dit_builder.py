"""Compatibility shim for the ltx_video family-owned ltx_dit_builder implementation."""

import sys as _sys

from .families.ltx_video import ltx_dit_builder as _impl

_sys.modules[__name__] = _impl
