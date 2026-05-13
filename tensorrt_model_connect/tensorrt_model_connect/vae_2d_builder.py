"""Compatibility shim for the pixart family-owned vae_2d_builder implementation."""

import sys as _sys

from .families.pixart import vae_2d_builder as _impl

_sys.modules[__name__] = _impl
