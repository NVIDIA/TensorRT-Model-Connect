"""Gemma-4 audio encoder TRT engine builder — SCAFFOLD.

This module is a placeholder. From the model card, the Gemma-4 audio
tower (model_type=``gemma4_audio``) is described as a 12-layer
conv-subsampling encoder with hidden_size=1024 and 8 attention heads.
The exact subsampling stack, transformer layer layout, and projector
shape need to be verified from the released ``modeling_gemma4.py``
before we wire a builder.

Until then ``build_gemma4_audio_engine`` raises NotImplementedError so
that callers get a clearly actionable error instead of a silently
broken engine.

OPEN QUESTIONS (resolve before implementing):
  * Input feature representation:
      - Raw waveform [num_samples] vs log-mel spectrogram
        [num_frames, num_mels]?
      - Sample rate (16 kHz typical), frame stride, num_mels.
  * Conv subsampling stack:
      - Number of conv layers, kernel sizes, strides — the 12-layer
        encoder count in the model card may refer to *transformer*
        layers stacked AFTER 2-3 conv-subsampling layers.
      - Activation between conv blocks (GLU / ReLU / GELU).
  * Transformer block layout:
      - Norm placement (pre-norm vs post-norm).
      - Whether per-head q/k norms are used (Gemma-3 introduced these).
      - Position encoding: RoPE / relative / sinusoidal absolute?
  * Audio projector:
      - Linear-only or full MLP?
      - Output dim must equal text decoder hidden_size.
  * Soft-token expansion: how many text "soft tokens" represent one
    audio chunk? Determines ``num_audio_pad_tokens`` in the VL config.

When implementing, the closest existing precedent is the speech
sub-engine in ``families/bark`` (if present) or the Phi-4-multimodal
audio path (``families/phi4_multimodal``) — both predate Gemma-4 but
share the conv-subsample-then-transformer template.

Engine I/O (proposed, subject to verification):
  Input:  audio_features [num_frames, num_mels] float32
          OR raw_audio [num_samples] float32
  Output: audio_features [num_soft_tokens, text_hidden_size] float32
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


def build_gemma4_audio_engine(
    audio_config: dict,
    audio_weights: "WeightDict",
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine for the Gemma-4 audio encoder.

    See module docstring for the list of open architectural questions.
    """
    raise NotImplementedError(
        "Gemma-4 audio encoder builder is not yet implemented. Resolve "
        "the open questions listed in "
        "families/gemma4/audio_encoder_builder.py before wiring the "
        "TRT graph. Inputs received: "
        f"audio_config keys={sorted(audio_config.keys())}, "
        f"num_weight_tensors={len(audio_weights)}, "
        f"precision={precision}.")
