"""Compatibility shim for the xlnet family-owned xlnet_builder implementation."""

import sys as _sys

from .families.xlnet import xlnet_builder as _impl

_sys.modules[__name__] = _impl
