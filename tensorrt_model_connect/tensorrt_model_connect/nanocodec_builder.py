"""Compatibility shim for the shared nanocodec_builder implementation."""

import sys as _sys

from .families._shared import nanocodec_builder as _impl

_sys.modules[__name__] = _impl
