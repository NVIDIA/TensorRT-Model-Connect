"""Compatibility shim for the ConvBERT family-owned builder."""

import sys as _sys

from .families.convbert import builder as _impl

_sys.modules[__name__] = _impl
