"""Compatibility shim for the internvl family-owned internvit_vision_builder implementation."""

import sys as _sys

from .families.internvl import internvit_vision_builder as _impl

_sys.modules[__name__] = _impl
