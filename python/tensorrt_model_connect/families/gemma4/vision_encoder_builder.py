"""Gemma-4 vision encoder TRT engine builder — SCAFFOLD.

This module is a placeholder. The Gemma-4 vision tower is most likely a
SigLIP-2 variant (consistent with Gemma-3) but the released config + HF
modeling code need to be cross-checked before we wire a builder.

Until then ``build_gemma4_vision_engine`` raises NotImplementedError so
that any caller (the harness, or follow-up bring-up tests) gets a
clearly actionable error instead of silently producing a broken engine.

OPEN QUESTIONS (resolve before implementing):
  * Patch size (Gemma-3 used 14). Verify ``vision_config.patch_size``.
  * Fixed image size (Gemma-3 used 896). Confirm whether Gemma-4 uses a
    different default and whether multiple resolutions are required.
  * Vision tower architecture:
      - SigLIP / SigLIP-2: standard ViT with mean-pool head?
      - Position embeddings: learned vs RoPE?
      - Normalization: LayerNorm or RMSNorm?
  * Multi-modal projector layout:
      - Linear-only ("mm_input_projection_weight") or full MLP?
      - Output dim must equal text decoder hidden_size — verify.
  * Image normalization stats: SigLIP uses mean=0.5 / std=0.5; verify.
  * Soft-token expansion: Gemma-3 expands one image into N "soft tokens";
    confirm the soft-token count and that the C++ runtime preprocessor
    can place them correctly.

When implementing, mirror the Qwen-VL vision builder structure
(``families/qwen_vl/qwen_vl_vision_builder.py``) but skip the
window-index / merge-group permutation if the Gemma-4 tower does not
need it.

Engine I/O (proposed, subject to verification):
  Input:  pixel_values [3, H, W] float32
  Output: image_features [num_soft_tokens, text_hidden_size] float32
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


def build_gemma4_vision_engine(
    vision_config: dict,
    vision_weights: "WeightDict",
    *,
    fixed_image_size: int = 896,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine for the Gemma-4 vision encoder.

    See module docstring for the list of open architectural questions.
    """
    raise NotImplementedError(
        "Gemma-4 vision encoder builder is not yet implemented. Resolve "
        "the open questions listed in "
        "families/gemma4/vision_encoder_builder.py before wiring the "
        "TRT graph; mirror families/qwen_vl/qwen_vl_vision_builder.py "
        "for structure. Inputs received: "
        f"vision_config keys={sorted(vision_config.keys())}, "
        f"num_weight_tensors={len(vision_weights)}, "
        f"fixed_image_size={fixed_image_size}, precision={precision}.")
