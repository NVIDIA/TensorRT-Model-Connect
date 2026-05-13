"""Compatibility shim for the z_image family-owned qwen3_encoder_builder implementation."""

import sys as _sys

from .families.z_image import qwen3_encoder_builder as _impl

_sys.modules[__name__] = _impl
