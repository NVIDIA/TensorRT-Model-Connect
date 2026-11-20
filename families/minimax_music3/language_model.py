# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry of the global language model.

A Qwen3 decoder, fine-tuned, that emits one semantic code and a hidden state
per audio frame. Thirty-six layers, 399 tensors: an embedding, a head, a final
norm, and eleven per layer.

Three facts a builder needs are not in the config as such:

* The vocabulary is **200000**, not the 151936 a stock Qwen3 carries. The
  16384 semantic codes begin at :data:`.prompt_format.AUDIO_CODE_OFFSET`,
  which is 151675 -- *below* stock Qwen3's ceiling, overlaying its reserved
  special-token block, and running past it into the extension. Assuming the
  codes are appended where Qwen3's vocabulary ends puts them 261 rows out.
* Attention is grouped: thirty-two query heads over eight key/value heads, so
  each key and value head serves four queries. The config states the two counts
  and leaves the ratio implied.
* Qwen3 normalises **per head** before the rotation -- ``q_norm`` and ``k_norm``
  are RMS norms over ``head_dim``, not over the model width. Applying them to
  the flattened projection instead would be wrong by a factor of the head
  count and is the easy mistake here.
"""

from __future__ import annotations

VOCAB_SIZE = 200000
HIDDEN_SIZE = 4096
NUM_HIDDEN_LAYERS = 36
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 12288
RMS_NORM_EPS = 1e-6
ROPE_THETA = 1_000_000.0
MAX_POSITION_EMBEDDINGS = 10240

#: Stock Qwen3's vocabulary, for the comparison the docstring draws.
BASE_QWEN3_VOCAB_SIZE = 151936

#: Tensors per decoder layer: two norms, four projections, two head norms,
#: three MLP matrices.
TENSORS_PER_LAYER = 11


def group_size() -> int:
    """Query heads served by each key/value head."""

    if NUM_ATTENTION_HEADS % NUM_KEY_VALUE_HEADS:
        raise ValueError(
            f"{NUM_ATTENTION_HEADS} query heads do not divide into "
            f"{NUM_KEY_VALUE_HEADS} key/value heads"
        )
    return NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS


def query_width() -> int:
    return NUM_ATTENTION_HEADS * HEAD_DIM


def key_value_width() -> int:
    return NUM_KEY_VALUE_HEADS * HEAD_DIM


def attention_scale() -> float:
    return HEAD_DIM ** -0.5


def total_tensors() -> int:
    """399: embedding, head, final norm, and eleven per layer."""

    return 3 + NUM_HIDDEN_LAYERS * TENSORS_PER_LAYER


def audio_token(code: int) -> int:
    """Vocabulary row of semantic ``code``."""

    from .prompt_format import AUDIO_CODE_OFFSET, SEMANTIC_VOCAB_SIZE

    if not 0 <= code < SEMANTIC_VOCAB_SIZE:
        raise ValueError(
            f"code must be in [0, {SEMANTIC_VOCAB_SIZE}), got {code}"
        )
    token = AUDIO_CODE_OFFSET + code
    if token >= VOCAB_SIZE:
        raise ValueError(f"token {token} is past the vocabulary")
    return token


def rope_tables(seq_len: int, offset: int = 0):
    """Return ``(cos, sin)`` of shape ``(seq_len, head_dim)``.

    Full rotation over the head, unlike the diffusion transformer's partial
    one, and with a theta a hundred times larger.
    """

    import numpy as np

    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if offset + seq_len > MAX_POSITION_EMBEDDINGS:
        raise ValueError(
            f"positions {offset}..{offset + seq_len} exceed the model's "
            f"{MAX_POSITION_EMBEDDINGS}"
        )
    exponents = np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM
    inv_freq = 1.0 / (ROPE_THETA ** exponents)
    positions = np.arange(offset, offset + seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    freqs = np.concatenate((freqs, freqs), axis=-1)
    return (
        np.ascontiguousarray(np.cos(freqs), dtype=np.float32),
        np.ascontiguousarray(np.sin(freqs), dtype=np.float32),
    )


def repeat_key_value(heads):
    """Expand eight key/value heads to thirty-two by repeating each in place."""

    import numpy as np

    array = np.asarray(heads)
    if array.shape[0] != NUM_KEY_VALUE_HEADS:
        raise ValueError(
            f"expected {NUM_KEY_VALUE_HEADS} heads, got {array.shape[0]}"
        )
    return np.repeat(array, group_size(), axis=0)


def kv_cache_shape(max_length: int) -> tuple[int, ...]:
    """Per-layer key or value cache: grouped heads, not query heads."""

    if max_length < 1:
        raise ValueError(f"max_length must be positive, got {max_length}")
    return (1, NUM_KEY_VALUE_HEADS, max_length, HEAD_DIM)
