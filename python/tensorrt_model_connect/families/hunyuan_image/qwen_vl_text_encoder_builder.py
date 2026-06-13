"""Qwen2.5-VL text encoder builder for HunyuanImage-2.1 (scaffold).

HunyuanImage-2.1 uses Qwen2.5-VL-Instruct as a second text encoder
alongside byT5 (see ``byt5_encoder_builder``). The Qwen2.5-VL encoder
provides semantic / scene understanding while byT5 provides byte-level
literal text conditioning (typography, glyphs).

Per Tencent's reference implementation, HunyuanImage uses only the
**language model** path of Qwen2.5-VL for T2I -- the vision tower is
disabled for text-to-image generation. The encoder output consumed by
the DiT is the LM's ``last_hidden_state`` at a chosen layer (or several
layers concatenated).

----------------------------------------------------------------------------
The codebase already has a Qwen2.5-VL LM-only text encoder builder used
by ``qwen_image``: ``families.qwen_image.qwen25_vl_text_encoder_builder``.
That builder is a good base because:
  * Same LM architecture (Qwen2.5-VL-7B-Instruct).
  * Same I/O contract (``input_ids`` + ``attention_mask`` -> last_hidden).
  * Same precision boundary (bf16 internal, fp32 IO).

This scaffold delegates to the qwen_image builder. The only HunyuanImage-
specific knob is **which hidden states to extract**: Tencent's pipeline
sometimes concatenates outputs from multiple decoder layers (akin to FLUX.2
+ Mistral 3). The gap list below tracks what needs to be confirmed on GPU.

----------------------------------------------------------------------------
GAPS — fill in when ``tencent/HunyuanImage-2.1`` is fetched on GPU host:

  1. Confirm Qwen2.5-VL variant (7B vs 3B) by reading
     ``text_encoder_2/config.json`` (or whatever the second encoder
     directory is named -- see plugin.load_weights for the contract).

  2. Confirm hidden state extraction layer(s):
        - Single-layer: last hidden state at ``model.norm`` output
          (matches qwen_image; ``apply_final_norm=True``).
        - Multi-layer: a la FLUX.2/Mistral 3 the reference may pull
          intermediate layer activations and concatenate. Tencent's
          published code in ``HunyuanImage-2.1/hyimage/diffusion/`` is
          the source of truth.

  3. Verify ``max_seq_len`` used at inference (qwen_image clamps to
     256, FLUX.2 + Mistral 3 uses 512).

  4. Decide whether HunyuanImage applies a learned text projection
     (``cross_attention_dim``) inside the DiT or in a separate text
     projection pass.

Until those are confirmed, this module exposes
``build_qwen_vl_text_encoder_engine`` as a thin shim that just calls
qwen_image's ``build_qwen25vl_text_encoder_engine``.
"""
from __future__ import annotations

from pathlib import Path


DEFAULT_QWEN_VL_MAX_SEQ_LEN = 256


def load_qwen_vl_text_encoder_weights(
    text_encoder_dir: str,
    *,
    max_seq_len: int = DEFAULT_QWEN_VL_MAX_SEQ_LEN,
    apply_final_norm: bool = True,
):
    """Load Qwen2.5-VL LM text encoder weights from a diffusers-format dir.

    Returns ``(Qwen25VLTextEncoderConfig, WeightDict)`` -- the exact tuple
    shape that ``qwen_image.qwen25_vl_text_encoder_builder.build_qwen25vl_text_encoder_engine``
    consumes.

    GAP: see module docstring -- multi-layer extraction would require a
    different loader (one that keeps all decoder layer outputs).
    """
    from ..qwen_image.qwen25_vl_text_encoder_builder import (
        load_qwen25vl_text_encoder_weights,
    )

    return load_qwen25vl_text_encoder_weights(
        Path(text_encoder_dir),
        max_seq_len=max_seq_len,
        apply_final_norm=apply_final_norm,
    )


def build_qwen_vl_text_encoder_engine(
    text_cfg,
    text_w,
    out_plan_path,
    *,
    verbose: bool = False,
) -> None:
    """Build the Qwen2.5-VL LM text encoder TRT engine.

    Delegates to ``qwen_image.qwen25_vl_text_encoder_builder.build_qwen25vl_text_encoder_engine``
    with ``enable_image_inputs=False`` (HunyuanImage is pure T2I; no
    image-conditioning at the text encoder level).

    GAPS:
      * Multi-layer hidden state extraction not yet wired (see module
        docstring). Add a new builder variant if Tencent's reference
        confirms multi-layer concatenation is used.
    """
    from ..qwen_image.qwen25_vl_text_encoder_builder import (
        build_qwen25vl_text_encoder_engine,
    )

    return build_qwen25vl_text_encoder_engine(
        text_cfg,
        text_w,
        out_plan_path,
        enable_image_inputs=False,
        verbose=verbose,
    )
