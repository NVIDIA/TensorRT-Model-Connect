"""Compatibility shim for the Qwen-VL family-owned vision encoder builder."""

import sys as _sys

from . import qwen_vl_vision_builder as _impl

_sys.modules[__name__] = _impl
