# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry of the RVQ depth decoder.

The local language model. Within one audio frame it predicts the seven
residual codebooks from the global model's hidden state and the codes sampled
so far. From the reference (diffusers v0.40.0,
``models/transformers/minimax_music3_rvq_depth_decoder.py``, Apache-2.0,
(c) The MiniMax Team and The HuggingFace Team)::

    h = inputs_embeds + pos_embedding[:steps]
    four times:
        h = h + attn(rms_norm(h))                    causal, no rope
        n = rms_norm(h)
        h = h + down(silu(gate(n)) * up(n))
    return rms_norm(h)

Three things distinguish it from the pipeline's other transformer. There is no
rotary embedding -- position enters once, as a learned table added to the
input. The sequence is at most sixteen steps, so attention is negligible next
to the projections. And the head dimension is 256, four times the DiT's, which
comes out of dividing 4096 by sixteen heads rather than from any config field.

``audio_embeddings``, ``projection`` and the seven ``audio_heads`` are not part
of this forward; the pipeline applies them around it. Their shapes live in
:mod:`.checkpoint`.
"""

from __future__ import annotations

HIDDEN_SIZE = 4096
NUM_LAYERS = 4
NUM_ATTENTION_HEADS = 16
INTERMEDIATE_SIZE = 6144
AUDIO_VOCAB_SIZE = 1024
NUM_CODEBOOKS = 8
MAX_POSITION_EMBEDDINGS = 16
RMS_NORM_EPS = 1e-6

#: The depth decoder predicts every codebook but the first, which the global
#: language model supplies.
NUM_RESIDUAL_CODEBOOKS = NUM_CODEBOOKS - 1


def head_dim() -> int:
    """Attention head width, which the config does not state directly."""

    if HIDDEN_SIZE % NUM_ATTENTION_HEADS:
        raise ValueError(
            f"hidden size {HIDDEN_SIZE} is not divisible by "
            f"{NUM_ATTENTION_HEADS} heads"
        )
    return HIDDEN_SIZE // NUM_ATTENTION_HEADS


def attention_scale() -> float:
    """The 1/sqrt(head_dim) applied before the softmax."""

    return head_dim() ** -0.5


def embedding_rows() -> int:
    """Rows of ``audio_embeddings``: one block of codes per residual codebook."""

    return AUDIO_VOCAB_SIZE * NUM_RESIDUAL_CODEBOOKS


def code_offset(codebook: int) -> int:
    """Row offset of ``codebook`` in the shared embedding table.

    The reference offsets residual codebook ``i`` by ``i * audio_vocab_size``,
    so the seven tables live end to end in one matrix.
    """

    if not 0 <= codebook < NUM_RESIDUAL_CODEBOOKS:
        raise ValueError(
            f"codebook must be in [0, {NUM_RESIDUAL_CODEBOOKS}), got {codebook}"
        )
    return codebook * AUDIO_VOCAB_SIZE


def steps_for(codes_sampled: int) -> int:
    """Depth-sequence length after ``codes_sampled`` codes.

    The sequence starts with the global hidden state, then grows by one for
    each code sampled, so predicting all seven codebooks needs eight steps at
    most -- well inside the sixteen the position table allows.
    """

    if not 0 <= codes_sampled <= NUM_RESIDUAL_CODEBOOKS:
        raise ValueError(
            f"codes_sampled must be in [0, {NUM_RESIDUAL_CODEBOOKS}], "
            f"got {codes_sampled}"
        )
    return 1 + codes_sampled


def causal_mask(steps: int):
    """Additive mask: zero where a step may attend, -inf where it may not."""

    import numpy as np

    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps > MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"steps {steps} exceeds the position table's "
            f"{MAX_POSITION_EMBEDDINGS}"
        )
    allowed = np.tril(np.ones((steps, steps), dtype=bool))
    mask = np.zeros((steps, steps), dtype=np.float32)
    mask[~allowed] = -np.inf
    return mask


def rms_norm(hidden, weight, eps: float = RMS_NORM_EPS):
    """Root-mean-square normalisation over the last dimension, then scale."""

    import numpy as np

    x = np.asarray(hidden, dtype=np.float32)
    scale = np.asarray(weight, dtype=np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * scale
