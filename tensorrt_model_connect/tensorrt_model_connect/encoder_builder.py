"""Compatibility shim for the bert family-owned encoder_builder implementation."""

import sys as _sys

from .families.bert import encoder_builder as _impl

_sys.modules[__name__] = _impl
