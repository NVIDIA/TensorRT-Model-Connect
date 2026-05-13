"""Compatibility shim for the fnet family-owned fnet_encoder_builder implementation."""

import sys as _sys

from .families.fnet import fnet_encoder_builder as _impl

_sys.modules[__name__] = _impl
