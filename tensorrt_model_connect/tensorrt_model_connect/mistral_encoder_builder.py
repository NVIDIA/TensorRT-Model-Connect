"""Compatibility shim for the flux family-owned mistral_encoder_builder implementation."""

import sys as _sys

from .families.flux import mistral_encoder_builder as _impl

_sys.modules[__name__] = _impl
