"""Compatibility shim for the shared vae_2d_builder implementation."""

import sys as _sys

from .families._shared import vae_2d_builder as _impl

_sys.modules[__name__] = _impl
