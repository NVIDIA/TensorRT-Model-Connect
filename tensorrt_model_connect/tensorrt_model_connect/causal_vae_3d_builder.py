"""Compatibility shim for the shared causal_vae_3d_builder implementation."""

import sys as _sys

from .families._shared import causal_vae_3d_builder as _impl

_sys.modules[__name__] = _impl
