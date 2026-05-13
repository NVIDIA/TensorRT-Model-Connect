"""Compatibility shim for the bark family-owned encodec_builder implementation."""

import sys as _sys

from .families.bark import encodec_builder as _impl

_sys.modules[__name__] = _impl
