"""Compatibility shim for the shared z_image_dit_builder implementation."""

import sys as _sys

from .families._shared import z_image_dit_builder as _impl

_sys.modules[__name__] = _impl
