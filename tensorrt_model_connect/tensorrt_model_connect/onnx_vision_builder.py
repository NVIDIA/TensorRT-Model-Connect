"""Compatibility shim for the shared onnx_vision_builder implementation."""

import sys as _sys

from .families._shared import onnx_vision_builder as _impl

_sys.modules[__name__] = _impl
