"""Compatibility shim for the flux family-owned clip_encoder_builder implementation."""

import sys as _sys

from .families.flux import clip_encoder_builder as _impl

_sys.modules[__name__] = _impl
