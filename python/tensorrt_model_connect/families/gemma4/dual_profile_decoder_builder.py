"""Compatibility shim — Gemma-4 reuses the Gemma dual-profile builder.

The Gemma-4 text decoder is architecturally identical to Gemma-2 /
Gemma-3 at the layer level, so we lean on the existing dual-profile
graph (prefill + decode optimization profiles sharing one engine).

VL prefill paths (``embed_input=True``) currently stay on the single-
profile ``standard_decoder_builder`` because the dual-profile builder
does not yet support the input_embed conditional.
"""

from __future__ import annotations

from ..gemma.dual_profile_decoder_builder import (
    build_dual_profile_decoder_engine,
)

__all__ = ["build_dual_profile_decoder_engine"]
