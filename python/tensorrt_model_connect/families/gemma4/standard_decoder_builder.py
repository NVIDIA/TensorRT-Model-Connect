"""Compatibility shim — Gemma-4 reuses the Gemma text decoder builder.

See ``families/gemma4/text_decoder_builder.py`` for the full rationale.
This module exists so the family directory matches the layout of other
families (qwen_vl, phi4_multimodal) and so the plugin can resolve
``from .standard_decoder_builder import build_standard_decoder_engine``
without an indirection.
"""

from __future__ import annotations

from ..gemma.standard_decoder_builder import (
    _apply_norm,
    _mark_debug_output,
    build_standard_decoder_engine,
)

__all__ = [
    "_apply_norm",
    "_mark_debug_output",
    "build_standard_decoder_engine",
]
