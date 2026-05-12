"""Compatibility shim for the shared qwen_vl_vision_builder implementation."""

import sys as _sys

from .families._shared import qwen_vl_vision_builder as _impl

_sys.modules[__name__] = _impl
