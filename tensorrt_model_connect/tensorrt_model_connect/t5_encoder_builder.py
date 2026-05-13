"""Compatibility shim for the flux family-owned t5_encoder_builder implementation."""

import sys as _sys

from .families.flux import t5_encoder_builder as _impl

_sys.modules[__name__] = _impl
