"""Compatibility shim for the shared encoder_builder implementation."""

import sys as _sys

from .families._shared import encoder_builder as _impl

_sys.modules[__name__] = _impl
