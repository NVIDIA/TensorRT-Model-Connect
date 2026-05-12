"""Compatibility shim for the shared encodec_builder implementation."""

import sys as _sys

from .families._shared import encodec_builder as _impl

_sys.modules[__name__] = _impl
