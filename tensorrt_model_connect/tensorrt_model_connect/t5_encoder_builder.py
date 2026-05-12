"""Compatibility shim for the shared t5_encoder_builder implementation."""

import sys as _sys

from .families._shared import t5_encoder_builder as _impl

_sys.modules[__name__] = _impl
