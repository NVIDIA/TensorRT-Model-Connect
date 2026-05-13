"""Compatibility shim for the flux family-owned flux_vae_builder implementation."""

import sys as _sys

from .families.flux import flux_vae_builder as _impl

_sys.modules[__name__] = _impl
