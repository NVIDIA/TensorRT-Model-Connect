"""Compatibility shim for the shared standard_decoder_builder implementation."""

import sys as _sys

from .families._shared import standard_decoder_builder as _impl

_sys.modules[__name__] = _impl
