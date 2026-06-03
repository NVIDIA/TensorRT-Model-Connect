"""Gemma-4 text decoder builder.

This module is a thin compatibility shim: Gemma-4's text tower is
architecturally identical to a Gemma-2 / Gemma-3 decoder (RMSNorm with
the +1.0 gamma convention, sqrt(hidden) embedding scale, 4 norms per
layer, gelu_pytorch_tanh-activated SwiGLU MLP, attention logit softcap,
and final logit softcap), so we reuse the existing Gemma decoder builder
verbatim.

If Gemma-4 introduces text-tower-only deviations (e.g. a new rotary base,
sliding-window cadence, MoE blocks, etc.) they would land here as
Gemma-4-specific helpers wrapping the Gemma builder.

Tensor contract matches the standard decoder:
  Inputs:  token_id, position_id, attention_mask, cache_k_i, cache_v_i
           (+ input_embed / use_input_embed when embed_input=True)
  Outputs: logits, present_k_i, present_v_i
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
