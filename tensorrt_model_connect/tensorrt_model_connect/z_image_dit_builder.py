"""Compatibility shim for the z_image family-owned z_image_dit_builder implementation."""

import sys as _sys

from .families.z_image import z_image_dit_builder as _impl

_sys.modules[__name__] = _impl
