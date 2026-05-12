"""Compatibility shim for the shared ltx_vae_builder implementation."""

import sys as _sys

from .families._shared import ltx_vae_builder as _impl

_sys.modules[__name__] = _impl
