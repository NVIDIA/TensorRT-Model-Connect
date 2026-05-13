"""Compatibility shim for the llama family-owned standard_decoder_builder implementation."""

import sys as _sys

from .families.llama import standard_decoder_builder as _impl

_sys.modules[__name__] = _impl
