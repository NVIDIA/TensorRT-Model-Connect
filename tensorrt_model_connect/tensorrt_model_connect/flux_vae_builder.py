"""Compatibility shim for the shared flux_vae_builder implementation."""

import sys as _sys

from .families._shared import flux_vae_builder as _impl

_sys.modules[__name__] = _impl
