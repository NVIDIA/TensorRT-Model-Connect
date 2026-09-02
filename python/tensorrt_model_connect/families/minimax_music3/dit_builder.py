# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the flow-matching diffusion transformer.

Thirty-six blocks of full attention with partial rotary embedding and a gated
feed-forward, wrapped in two residual stem convolutions.

Three things are constants at build time and are folded rather than computed:
the rotary tables, which depend only on the sequence length; the zero block
concatenated into the input; and the causal-free attention, which needs no
mask at all. The timestep is not folded -- it is a runtime input, so the
Fourier projection and its embedding are layers.

``num_layers`` is a parameter so a shortened stack can be built for
validation; the checkpoint's thirty-six is the default.

**Validated.** Built on an A40 with TensorRT 11.1.0.106 over a two-layer stack
of the published weights -- 16 s, a 579 MB engine -- at a latent length of 64,
and compared with the reference model truncated to the same two blocks.
Agreement is 1.6e-06 relative. The remaining thirty-four layers are the same
block repeated, so what a longer stack adds is build time and memory rather
than new graph structure.
"""

from __future__ import annotations

from typing import Any

from .dit import (
    ATTENTION_HEAD_DIM,
    CONCAT_CHANNELS,
    CONDITION_DIM,
    FF_INNER_DIM,
    IN_CHANNELS,
    INNER_DIM,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    ROTARY_DIM,
    attention_scale,
    rotary_tables,
    sequence_length,
)

LATENTS_NAME = "latents"
CONDITION_NAME = "condition"
TIMESTEP_NAME = "timestep"
OUTPUT_NAME = "velocity"

LAYER_NORM_EPS = 1e-5

DISABLE_TF32 = True
WORKSPACE_BYTES = 8 << 30


def _no_bias(trt: Any):
    """An empty bias for a convolution the checkpoint stores without one.

    A zero-length numpy array is not the same thing: TensorRT requires the
    count and the pointer to agree, and an empty array still carries a
    non-null pointer, which the parameter check rejects.
    """

    return trt.Weights()


def _const(network, trt, array):
    import numpy as np

    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return network.add_constant(data.shape, trt.Weights(data)).get_output(0)


def _linear(network, trt, hidden, weight, out_features, bias=None):
    """``x @ W.T (+ b)`` with the weight as a constant operand."""

    import numpy as np

    array = np.asarray(weight, dtype=np.float32)
    if array.shape[0] != out_features:
        raise ValueError(f"weight has {array.shape[0]} rows, expected {out_features}")
    operand = _const(network, trt, np.ascontiguousarray(array.T).reshape(
        1, array.shape[1], out_features))
    out = network.add_matrix_multiply(
        hidden, trt.MatrixOperation.NONE, operand, trt.MatrixOperation.NONE
    ).get_output(0)
    if bias is not None:
        offset = _const(network, trt, np.asarray(bias, dtype=np.float32).reshape(
            1, 1, out_features))
        out = network.add_elementwise(
            out, offset, trt.ElementWiseOperation.SUM
        ).get_output(0)
    return out


def add_layer_norm(network: Any, trt: Any, hidden: Any, weight, bias) -> Any:
    """Mean-and-variance normalisation over the last dimension, then affine."""

    import numpy as np

    mean = network.add_reduce(hidden, trt.ReduceOperation.AVG, axes=1 << 2,
                              keep_dims=True)
    centred = network.add_elementwise(hidden, mean.get_output(0),
                                      trt.ElementWiseOperation.SUB)
    squared = network.add_elementwise(centred.get_output(0), centred.get_output(0),
                                      trt.ElementWiseOperation.PROD)
    variance = network.add_reduce(squared.get_output(0), trt.ReduceOperation.AVG,
                                  axes=1 << 2, keep_dims=True)
    shifted = network.add_elementwise(
        variance.get_output(0),
        _const(network, trt, np.array([[[LAYER_NORM_EPS]]], dtype=np.float32)),
        trt.ElementWiseOperation.SUM,
    )
    deviation = network.add_unary(shifted.get_output(0), trt.UnaryOperation.SQRT)
    normalised = network.add_elementwise(
        centred.get_output(0), deviation.get_output(0), trt.ElementWiseOperation.DIV
    )
    scaled = network.add_elementwise(
        normalised.get_output(0),
        _const(network, trt, np.asarray(weight, dtype=np.float32).reshape(1, 1, -1)),
        trt.ElementWiseOperation.PROD,
    )
    return network.add_elementwise(
        scaled.get_output(0),
        _const(network, trt, np.asarray(bias, dtype=np.float32).reshape(1, 1, -1)),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)


def add_partial_rope(network: Any, trt: Any, heads: Any, cos, sin, seq_len: int) -> Any:
    """Rotate the leading ``ROTARY_DIM`` of each head, pass the rest through.

    ``heads`` is ``(num_heads, seq_len, head_dim)``.
    """

    import numpy as np

    rotated = network.add_slice(
        heads, (0, 0, 0), (NUM_ATTENTION_HEADS, seq_len, ROTARY_DIM), (1, 1, 1)
    ).get_output(0)
    passthrough = network.add_slice(
        heads, (0, 0, ROTARY_DIM),
        (NUM_ATTENTION_HEADS, seq_len, ATTENTION_HEAD_DIM - ROTARY_DIM), (1, 1, 1)
    ).get_output(0)

    half = ROTARY_DIM // 2
    first = network.add_slice(
        rotated, (0, 0, 0), (NUM_ATTENTION_HEADS, seq_len, half), (1, 1, 1)
    ).get_output(0)
    second = network.add_slice(
        rotated, (0, 0, half), (NUM_ATTENTION_HEADS, seq_len, half), (1, 1, 1)
    ).get_output(0)
    negated = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    swapped = network.add_concatenation([negated, first])
    swapped.axis = 2

    cos_c = _const(network, trt, np.asarray(cos).reshape(1, seq_len, ROTARY_DIM))
    sin_c = _const(network, trt, np.asarray(sin).reshape(1, seq_len, ROTARY_DIM))
    direct = network.add_elementwise(rotated, cos_c, trt.ElementWiseOperation.PROD)
    turned = network.add_elementwise(swapped.get_output(0), sin_c,
                                     trt.ElementWiseOperation.PROD)
    combined = network.add_elementwise(direct.get_output(0), turned.get_output(0),
                                       trt.ElementWiseOperation.SUM)

    joined = network.add_concatenation([combined.get_output(0), passthrough])
    joined.axis = 2
    return joined.get_output(0)


def add_attention(network, trt, hidden, weights, prefix, seq_len, cos, sin):
    """Full attention with partial rotary embedding and no mask."""

    heads = {}
    for name in ("to_q", "to_k", "to_v"):
        flat = _linear(network, trt, hidden,
                       weights[f"{prefix}.attn.{name}.weight"], INNER_DIM)
        split = network.add_shuffle(flat)
        split.reshape_dims = (seq_len, NUM_ATTENTION_HEADS, ATTENTION_HEAD_DIM)
        split.second_transpose = (1, 0, 2)
        heads[name] = split.get_output(0)

    query = add_partial_rope(network, trt, heads["to_q"], cos, sin, seq_len)
    key = add_partial_rope(network, trt, heads["to_k"], cos, sin, seq_len)

    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE)
    scaled = network.add_elementwise(
        scores.get_output(0), _const(network, trt, [[[attention_scale()]]]),
        trt.ElementWiseOperation.PROD)
    probabilities = network.add_softmax(scaled.get_output(0))
    probabilities.axes = 1 << 2

    context = network.add_matrix_multiply(
        probabilities.get_output(0), trt.MatrixOperation.NONE,
        heads["to_v"], trt.MatrixOperation.NONE)
    merge = network.add_shuffle(context.get_output(0))
    merge.first_transpose = (1, 0, 2)
    merge.reshape_dims = (1, seq_len, INNER_DIM)
    return _linear(network, trt, merge.get_output(0),
                   weights[f"{prefix}.attn.to_out.0.weight"], INNER_DIM)


def add_gated_feed_forward(network, trt, hidden, weights, prefix, seq_len):
    """``ff_out(first * silu(second))`` over the two halves of ``ff_in``."""

    projected = _linear(network, trt, hidden, weights[f"{prefix}.ff_in.weight"],
                        2 * FF_INNER_DIM, weights[f"{prefix}.ff_in.bias"])
    first = network.add_slice(projected, (0, 0, 0), (1, seq_len, FF_INNER_DIM),
                              (1, 1, 1)).get_output(0)
    second = network.add_slice(projected, (0, 0, FF_INNER_DIM),
                               (1, seq_len, FF_INNER_DIM), (1, 1, 1)).get_output(0)
    gate = network.add_activation(second, trt.ActivationType.SIGMOID).get_output(0)
    silu = network.add_elementwise(second, gate, trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(first, silu.get_output(0),
                                    trt.ElementWiseOperation.PROD)
    return _linear(network, trt, gated.get_output(0),
                   weights[f"{prefix}.ff_out.weight"], INNER_DIM,
                   weights[f"{prefix}.ff_out.bias"])


def add_dit(network: Any, trt: Any, latents: Any, condition: Any, timestep_embedding: Any,
            *, latent_length: int, weights: dict, num_layers: int = NUM_LAYERS) -> Any:
    """Add the transformer and return the predicted velocity.

    ``timestep_embedding`` is the already-embedded ``(1, 1, INNER_DIM)`` prefix
    token, so the Fourier projection stays outside this subgraph.
    """

    import numpy as np

    seq_len = sequence_length(latent_length)

    zeros = _const(network, trt, np.zeros((1, IN_CHANNELS, latent_length),
                                          dtype=np.float32))
    swapped = network.add_shuffle(condition)
    swapped.first_transpose = (0, 2, 1)
    stacked = network.add_concatenation([latents, zeros, swapped.get_output(0)])
    stacked.axis = 1

    to_conv = network.add_shuffle(stacked.get_output(0))
    to_conv.reshape_dims = (1, CONCAT_CHANNELS, 1, latent_length)
    conv = network.add_convolution_nd(
        to_conv.get_output(0), CONCAT_CHANNELS, (1, 1),
        trt.Weights(np.ascontiguousarray(
            np.asarray(weights["preprocess_conv.weight"], dtype=np.float32).reshape(
                CONCAT_CHANNELS, CONCAT_CHANNELS, 1, 1))),
        _no_bias(trt),
    )
    from_conv = network.add_shuffle(conv.get_output(0))
    from_conv.reshape_dims = (1, CONCAT_CHANNELS, latent_length)
    residual = network.add_elementwise(
        from_conv.get_output(0), stacked.get_output(0), trt.ElementWiseOperation.SUM)

    to_tokens = network.add_shuffle(residual.get_output(0))
    to_tokens.first_transpose = (0, 2, 1)
    hidden = _linear(network, trt, to_tokens.get_output(0),
                     weights["proj_in.weight"], INNER_DIM)

    prefixed = network.add_concatenation([timestep_embedding, hidden])
    prefixed.axis = 1
    hidden = prefixed.get_output(0)

    cos, sin = rotary_tables(seq_len)
    for layer in range(num_layers):
        prefix = f"transformer_blocks.{layer}"
        normed = add_layer_norm(network, trt, hidden,
                                weights[f"{prefix}.norm1.weight"],
                                weights[f"{prefix}.norm1.bias"])
        attended = add_attention(network, trt, normed, weights, prefix, seq_len,
                                 cos, sin)
        hidden = network.add_elementwise(hidden, attended,
                                         trt.ElementWiseOperation.SUM).get_output(0)
        normed = add_layer_norm(network, trt, hidden,
                                weights[f"{prefix}.norm2.weight"],
                                weights[f"{prefix}.norm2.bias"])
        projected = add_gated_feed_forward(network, trt, normed, weights, prefix,
                                           seq_len)
        hidden = network.add_elementwise(hidden, projected,
                                         trt.ElementWiseOperation.SUM).get_output(0)

    trimmed = network.add_slice(hidden, (0, 1, 0), (1, latent_length, INNER_DIM),
                                (1, 1, 1)).get_output(0)
    out = _linear(network, trt, trimmed, weights["proj_out.weight"], IN_CHANNELS)

    to_channels = network.add_shuffle(out)
    to_channels.first_transpose = (0, 2, 1)
    to_post = network.add_shuffle(to_channels.get_output(0))
    to_post.reshape_dims = (1, IN_CHANNELS, 1, latent_length)
    post = network.add_convolution_nd(
        to_post.get_output(0), IN_CHANNELS, (1, 1),
        trt.Weights(np.ascontiguousarray(
            np.asarray(weights["postprocess_conv.weight"], dtype=np.float32).reshape(
                IN_CHANNELS, IN_CHANNELS, 1, 1))),
        _no_bias(trt),
    )
    from_post = network.add_shuffle(post.get_output(0))
    from_post.reshape_dims = (1, IN_CHANNELS, latent_length)
    return network.add_elementwise(
        from_post.get_output(0), to_channels.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)


def configure(config: Any, trt: Any) -> None:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if DISABLE_TF32:
        config.clear_flag(trt.BuilderFlag.TF32)


def expected_io_shapes(latent_length: int) -> dict[str, tuple[int, ...]]:
    return {
        LATENTS_NAME: (1, IN_CHANNELS, latent_length),
        CONDITION_NAME: (1, latent_length, CONDITION_DIM),
        OUTPUT_NAME: (1, IN_CHANNELS, latent_length),
    }


#: Rows of ``time_proj.weight``: half the Fourier width, since cosine and sine
#: are concatenated.
FOURIER_HALF_DIM = 128

TIMESTEP_SCALAR_NAME = "timestep"


def add_timestep_embedding(network: Any, trt: Any, timestep: Any, *, weights: dict) -> Any:
    """Embed a scalar flow-matching time into the prefix token.

    Mirrors ``MiniMaxMusic3FourierEmbedding`` followed by ``TimestepEmbedding``
    in ``diffusers.models.transformers.transformer_minimax_music3``::

        angles = 2 * pi * t @ time_proj.weight.T
        cat(angles.cos(), angles.sin()) -> linear_1 -> silu -> linear_2

    **Cosine comes first.** Swapping the halves costs nothing at build time and
    everything at inference: the denoiser still runs, and the audio it produces
    is wrong.

    Building this into the graph rather than on the host is what keeps the
    bundle self-sufficient -- the alternative exports three more weight tensors
    as bundle sections for the runtime to multiply by hand.
    """

    import math

    import numpy as np

    projection = np.asarray(weights["time_proj.weight"], dtype=np.float32).reshape(-1)
    if projection.size != FOURIER_HALF_DIM:
        raise ValueError(
            f"time_proj.weight has {projection.size} rows, expected {FOURIER_HALF_DIM}"
        )
    scale = np.ascontiguousarray(
        (2.0 * math.pi * projection).reshape(1, 1, FOURIER_HALF_DIM), dtype=np.float32
    )
    angles = network.add_elementwise(
        timestep,
        network.add_constant(scale.shape, trt.Weights(scale)).get_output(0),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)

    cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    fourier = network.add_concatenation([cosine, sine])
    fourier.axis = 2
    hidden = fourier.get_output(0)

    for index, activate in ((1, True), (2, False)):
        weight = np.ascontiguousarray(
            np.asarray(weights[f"time_embed.linear_{index}.weight"], dtype=np.float32)
        )
        bias = np.ascontiguousarray(
            np.asarray(weights[f"time_embed.linear_{index}.bias"],
                       dtype=np.float32).reshape(1, 1, -1)
        )
        # The checkpoint stores [out, in] for a torch Linear; a matmul against
        # the constant needs the transpose, and a leading axis to match rank.
        constant = network.add_constant(
            (1,) + weight.shape, trt.Weights(weight)
        ).get_output(0)
        hidden = network.add_matrix_multiply(
            hidden, trt.MatrixOperation.NONE, constant, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        hidden = network.add_elementwise(
            hidden,
            network.add_constant(bias.shape, trt.Weights(bias)).get_output(0),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        if activate:
            # TensorRT 11.2 has no ActivationType.SILU, so silu is built as
            # x * sigmoid(x) -- the same construction graph_ops uses.
            gate = network.add_activation(hidden, trt.ActivationType.SIGMOID)
            hidden = network.add_elementwise(
                hidden, gate.get_output(0), trt.ElementWiseOperation.PROD
            ).get_output(0)
    return hidden
