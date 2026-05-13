"""Compatibility shim for the qwen_vl family-owned onnx_vision_builder implementation."""

import sys as _sys

from .families.qwen_vl import onnx_vision_builder as _impl

_sys.modules[__name__] = _impl
