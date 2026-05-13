"""Compatibility shim for the wan_t2v family-owned causal_vae_3d_builder implementation."""

import sys as _sys

from .families.wan_t2v import causal_vae_3d_builder as _impl

_sys.modules[__name__] = _impl
