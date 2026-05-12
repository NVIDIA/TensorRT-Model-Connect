"""Compatibility shim for the shared xlnet_builder implementation."""

import sys as _sys

from .families._shared import xlnet_builder as _impl

_sys.modules[__name__] = _impl
