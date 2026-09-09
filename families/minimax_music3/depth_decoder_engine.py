# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The depth decoder as an engine the runtime can actually drive.

:func:`.depth_decoder_builder.add_depth_decoder` builds the four transformer
blocks: embeddings in, hidden states out. That is the middle of the stage, not
the stage. Three tensors the checkpoint carries were left outside the graph,
and without them the runtime holds a hidden state and cannot turn it into a
code:

``projection.weight``
    Projects the language model's frame hidden state into the depth sequence's
    first position.
``audio_embeddings.weight``
    Embeds the codes already sampled for this frame. Seven ``1024``-row blocks
    live end to end in one matrix, so residual codebook ``i`` is offset by
    ``i * AUDIO_VOCAB_SIZE`` -- see :func:`.depth_decoder.code_offset`.
``audio_heads.{0..6}.weight``
    Seven output projections, one per residual codebook. They are **not**
    interchangeable: position ``j`` is read by head ``j``, so a single stacked
    matmul would silently score every position against the first codebook.

Putting them in the graph rather than in the runtime is what keeps the bundle
self-sufficient. The alternative is exporting a ``7 * 1024 x 4096`` embedding
matrix and seven more of the same size as bundle sections and doing the gather
on the host, which moves 900 MB of weights out of the engine for no gain.

The whole depth sequence is at most eight positions over four layers, so this
runs the prefix again at each step instead of carrying a cache. A cache here
would cost more bookkeeping than the recompute it saves.
"""

from __future__ import annotations

from typing import Any

from .prompt_format import AUDIO_CODE_OFFSET, SEMANTIC_VOCAB_SIZE
from .depth_decoder import (
    NUM_CODEBOOKS,
    AUDIO_VOCAB_SIZE,
    HIDDEN_SIZE,
    NUM_RESIDUAL_CODEBOOKS,
    code_offset,
    embedding_rows,
    steps_for,
)

#: Engine tensor names.
HIDDEN_INPUT = "lm_hidden"
CODES_INPUT = "codes"
LOGITS_OUTPUT = "logits"
#: The per-position hidden states the condition encoder reads. Together
#: with the language model's own hidden state they are the eight streams
#: condition_encoder.NUM_CONDITION_LAYERS counts.
HIDDEN_OUTPUT = "depth_hidden"
#: The whole frame's embedding, which the language model reads back as
#: its next input. Both tables already live in this graph, so computing
#: it here keeps 251 MB of embeddings out of the bundle sections.
FRAME_EMBED_OUTPUT = "frame_embed"

#: Steps the engine is compiled for: the hidden state plus all seven codes.
MAX_STEPS = steps_for(NUM_RESIDUAL_CODEBOOKS)


def _const(network: Any, trt: Any, array, *, rank: int | None = None):
    """Add a constant, optionally with leading ones so a matmul's ranks match.

    TensorRT refuses a matrix multiply whose operands differ in rank -- "inputs
    with the same operation must have same number of dimensions". The activations
    here are (1, steps, width), so a checkpoint's rank-2 weight needs a leading
    batch axis before it can meet them.
    """

    import numpy as np

    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if rank is not None and contiguous.ndim < rank:
        contiguous = contiguous.reshape((1,) * (rank - contiguous.ndim) + contiguous.shape)
    return network.add_constant(contiguous.shape, trt.Weights(contiguous)).get_output(0)


def offset_table():
    """Return the per-position row offsets into the shared embedding table.

    Position ``j + 1`` of the depth sequence holds the code drawn from residual
    codebook ``j``, so the offsets are those of codebooks 0 through 6.
    """

    import numpy as np

    return np.asarray(
        [code_offset(j) for j in range(NUM_RESIDUAL_CODEBOOKS)], dtype=np.int32
    ).reshape(1, NUM_RESIDUAL_CODEBOOKS)


def semantic_slice(embed_tokens):
    """Return the language model rows an audio code can address.

    The reference embeds the frame's semantic code with the *language model's*
    table, at ``code + AUDIO_CODE_OFFSET``. Only the semantic block is ever
    reachable, so the engine carries those 16384 rows rather than all 200000 --
    134 MB instead of 1.6 GB, with nothing lost.
    """

    import numpy as np

    table = np.asarray(embed_tokens)
    first = AUDIO_CODE_OFFSET
    last = first + SEMANTIC_VOCAB_SIZE
    if table.shape[0] < last:
        raise ValueError(
            f"embed_tokens has {table.shape[0]} rows, need at least {last}"
        )
    return np.ascontiguousarray(table[first:last], dtype=np.float32)


def audio_offsets():
    """Row offsets for the residual codes that re-enter the sequence.

    Position ``p`` holds the code drawn at step ``p - 1`` and reads block
    ``p - 2`` -- the reference's ``(index - 1) * audio_vocab_size``. Six codes
    re-enter; the seventh is drawn and never fed back.
    """

    import numpy as np

    return np.asarray(
        [index * AUDIO_VOCAB_SIZE for index in range(NUM_RESIDUAL_CODEBOOKS - 1)],
        dtype=np.int32,
    ).reshape(1, NUM_RESIDUAL_CODEBOOKS - 1)


def add_input_sequence(network: Any, trt: Any, lm_hidden: Any, codes: Any,
                       *, weights: dict, embed_tokens) -> Any:
    """Build the depth sequence the reference builds.

    Eight positions: the language model's hidden state, then its embedding of
    the semantic code, then the six residual codes that re-enter. Every one of
    them goes through ``projection`` -- applying it only to the hidden state,
    as an earlier revision did, leaves the six embeddings in a different space
    from the one position the decoder was trained to read them beside.
    """

    import numpy as np

    semantic = _const(network, trt, semantic_slice(embed_tokens))
    residual_table = np.asarray(weights["audio_embeddings.weight"], dtype=np.float32)
    if residual_table.shape != (embedding_rows(), HIDDEN_SIZE):
        raise ValueError(
            f"audio_embeddings.weight is {residual_table.shape}, expected "
            f"({embedding_rows()}, {HIDDEN_SIZE})"
        )

    # codes[0] is the semantic code and reads the language model's table;
    # codes[1:] are residual codes and read their own blocks.
    semantic_code = network.add_slice(codes, (0, 0), (1, 1), (1, 1)).get_output(0)
    residual_codes = network.add_slice(
        codes, (0, 1), (1, NUM_RESIDUAL_CODEBOOKS - 1), (1, 1)
    ).get_output(0)

    first_embed = network.add_gather(semantic, semantic_code, 0).get_output(0)

    offsets = network.add_constant(
        (1, NUM_RESIDUAL_CODEBOOKS - 1), trt.Weights(np.ascontiguousarray(audio_offsets()))
    ).get_output(0)
    rows = network.add_elementwise(residual_codes, offsets, trt.ElementWiseOperation.SUM)
    residual_embed = network.add_gather(
        _const(network, trt, residual_table), rows.get_output(0), 0
    ).get_output(0)

    stacked = network.add_concatenation([lm_hidden, first_embed, residual_embed])
    stacked.axis = 1

    projection = np.asarray(weights["projection.weight"], dtype=np.float32)
    if projection.shape != (HIDDEN_SIZE, HIDDEN_SIZE):
        raise ValueError(
            f"projection.weight is {projection.shape}, expected "
            f"({HIDDEN_SIZE}, {HIDDEN_SIZE})"
        )
    return network.add_matrix_multiply(
        stacked.get_output(0), trt.MatrixOperation.NONE,
        _const(network, trt, projection, rank=3), trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)


def add_output_heads(network: Any, trt: Any, hidden: Any, *, weights: dict) -> Any:
    """Score each position with its own codebook head and stack the results."""

    import numpy as np

    scored = []
    for codebook in range(NUM_RESIDUAL_CODEBOOKS):
        head = np.asarray(
            weights[f"audio_heads.{codebook}.weight"], dtype=np.float32
        )
        if head.shape != (AUDIO_VOCAB_SIZE, HIDDEN_SIZE):
            raise ValueError(
                f"audio_heads.{codebook}.weight is {head.shape}, expected "
                f"({AUDIO_VOCAB_SIZE}, {HIDDEN_SIZE})"
            )
        # Head j reads position j + 1: position 0 holds the language model's
        # hidden state and predicts nothing, so the seven predicting positions
        # start at one.
        slice_layer = network.add_slice(
            hidden, (0, codebook + 1, 0), (1, 1, HIDDEN_SIZE), (1, 1, 1)
        )
        scored.append(
            network.add_matrix_multiply(
                slice_layer.get_output(0), trt.MatrixOperation.NONE,
                _const(network, trt, head, rank=3), trt.MatrixOperation.TRANSPOSE,
            ).get_output(0)
        )
    concat = network.add_concatenation(scored)
    concat.axis = 1
    return concat.get_output(0)


def expected_io_shapes() -> dict:
    """Return the engine's input and output shapes."""

    return {
        HIDDEN_INPUT: (1, 1, HIDDEN_SIZE),
        CODES_INPUT: (1, NUM_CODEBOOKS),
        LOGITS_OUTPUT: (1, NUM_RESIDUAL_CODEBOOKS, AUDIO_VOCAB_SIZE),
        HIDDEN_OUTPUT: (1, NUM_RESIDUAL_CODEBOOKS, HIDDEN_SIZE),
        FRAME_EMBED_OUTPUT: (1, 1, HIDDEN_SIZE),
    }


def add_hidden_output(network: Any, trt: Any, hidden: Any) -> Any:
    """Return the seven predicting positions' hidden states.

    The states the condition encoder wants are positions 1 through 7 -- the
    same slice the heads read. Position 0 holds the language model's own hidden
    state, which the encoder receives separately as its first stream.
    """

    layer = network.add_slice(
        hidden, (0, 1, 0), (1, NUM_RESIDUAL_CODEBOOKS, HIDDEN_SIZE), (1, 1, 1)
    )
    return layer.get_output(0)


def add_frame_embedding(network: Any, trt: Any, codes: Any, *, weights: dict,
                        embed_tokens) -> Any:
    """Return the embedding the language model reads back for this frame.

    Mirrors ``_embed_audio_frame``: the semantic code through the language
    model's table, plus all seven residual codes through their own blocks,
    summed and scaled by ``num_codebooks ** -0.5``.

    Feeding the sampled token id instead -- as an earlier revision did -- keeps
    only the first of eight codebooks. The language model's state then diverges
    from the first generated frame, and its hidden state correlates with the
    reference's at 0.03.
    """

    import numpy as np

    semantic = _const(network, trt, semantic_slice(embed_tokens))
    residual = _const(
        network, trt, np.asarray(weights["audio_embeddings.weight"], dtype=np.float32)
    )

    semantic_code = network.add_slice(codes, (0, 0), (1, 1), (1, 1)).get_output(0)
    residual_codes = network.add_slice(
        codes, (0, 1), (1, NUM_RESIDUAL_CODEBOOKS), (1, 1)
    ).get_output(0)

    # Every residual codebook reads its own block, this time all seven of them.
    offsets = np.asarray(
        [index * AUDIO_VOCAB_SIZE for index in range(NUM_RESIDUAL_CODEBOOKS)],
        dtype=np.int32,
    ).reshape(1, NUM_RESIDUAL_CODEBOOKS)
    offset_const = network.add_constant(
        offsets.shape, trt.Weights(np.ascontiguousarray(offsets))
    ).get_output(0)
    rows = network.add_elementwise(
        residual_codes, offset_const, trt.ElementWiseOperation.SUM
    )

    first = network.add_gather(semantic, semantic_code, 0).get_output(0)
    extra = network.add_gather(residual, rows.get_output(0), 0).get_output(0)
    summed = network.add_reduce(extra, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)

    total = network.add_elementwise(
        first, summed.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    scale = np.full((1, 1, 1), float(NUM_CODEBOOKS) ** -0.5, dtype=np.float32)
    return network.add_elementwise(
        total,
        network.add_constant(scale.shape, trt.Weights(np.ascontiguousarray(scale))).get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
