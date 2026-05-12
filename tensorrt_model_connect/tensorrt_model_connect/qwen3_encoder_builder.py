"""Compatibility shim for the shared qwen3_encoder_builder implementation."""

import sys as _sys

from .families._shared import qwen3_encoder_builder as _impl

_sys.modules[__name__] = _impl
