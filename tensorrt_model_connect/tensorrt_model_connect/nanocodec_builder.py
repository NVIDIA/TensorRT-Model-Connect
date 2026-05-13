"""Compatibility shim for the magpie_tts family-owned nanocodec_builder implementation."""

import sys as _sys

from .families.magpie_tts import nanocodec_builder as _impl

_sys.modules[__name__] = _impl
