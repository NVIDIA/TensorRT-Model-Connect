"""Compatibility shim for the qwen_vl family-owned qwen_vl_vision_builder implementation."""

import sys as _sys

from .families.qwen_vl import qwen_vl_vision_builder as _impl

_sys.modules[__name__] = _impl
