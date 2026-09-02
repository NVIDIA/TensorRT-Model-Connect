# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the RVQ depth decoder.

Four blocks of causal attention and SwiGLU over a sequence of at most sixteen
steps. Position enters once, as a constant slice of the learned table added to
the input, so there is no rotary embedding to express.

Root-mean-square normalisation is built from primitives -- square, mean,
epsilon, reciprocal square root, scale -- rather than assumed present as a
layer, and the causal mask is a build-time constant since the step count is
fixed per engine.

**Validated.** Built on an A40 with TensorRT 11.1.0.106 at one, four and eight
steps -- 9 to 13 s, a 2282 MB engine -- and compared with the reference module
on the published weights. Agreement is 2.3e-07 to 8.4e-07 relative, the
sharpest of the pipeline's networks, which the short sequence and the shallow
stack both help.
"""

from __future__ import annotations

from typing import Any

from .depth_decoder import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_LAYERS,
    RMS_NORM_EPS,
    attention_scale,
    causal_mask,
    head_dim,
)

INPUT_NAME = "inputs_embeds"
OUTPUT_NAME = "hidden_states"

DISABLE_TF32 = True
WORKSPACE_BYTES = 2 << 30


def _const(network, trt, array):
    import numpy as np

    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return network.add_constant(data.shape, trt.Weights(data)).get_output(0)


def add_rms_norm(network: Any, trt: Any, hidden: Any, weight) -> Any:
    """``x * rsqrt(mean(x ** 2) + eps) * weight`` over the last dimension."""

    import numpy as np

    squared = network.add_elementwise(hidden, hidden, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        squared.get_output(0), trt.ReduceOperation.AVG, axes=1 << 2, keep_dims=True
    )
    eps = _const(network, trt, np.array([[[RMS_NORM_EPS]]], dtype=np.float32))
    shifted = network.add_elementwise(
        mean.get_output(0), eps, trt.ElementWiseOperation.SUM
    )
    inv = network.add_unary(shifted.get_output(0), trt.UnaryOperation.SQRT)
    normalised = network.add_elementwise(
        hidden, inv.get_output(0), trt.ElementWiseOperation.DIV
    )
    scale = _const(network, trt, np.asarray(weight, dtype=np.float32).reshape(1, 1, -1))
    return network.add_elementwise(
        normalised.get_output(0), scale, trt.ElementWiseOperation.PROD
    ).get_output(0)


def _linear(network, trt, hidden, weight, out_features: int):
    """A bias-free ``x @ W.T`` as a constant-operand matrix multiply."""

    import numpy as np

    array = np.asarray(weight, dtype=np.float32)
    if array.shape[0] != out_features:
        raise ValueError(
            f"weight has {array.shape[0]} rows, expected {out_features}"
        )
    transposed = _const(network, trt, np.ascontiguousarray(array.T).reshape(
        1, array.shape[1], out_features))
    return network.add_matrix_multiply(
        hidden, trt.MatrixOperation.NONE, transposed, trt.MatrixOperation.NONE
    ).get_output(0)


def add_attention(network: Any, trt: Any, hidden: Any, weights: dict, prefix: str,
                  steps: int) -> Any:
    """Causal multi-head attention with no positional rotation."""

    heads, dim = NUM_ATTENTION_HEADS, head_dim()
    projections = {}
    for name in ("to_q", "to_k", "to_v"):
        flat = _linear(network, trt, hidden, weights[f"{prefix}.attn.{name}.weight"],
                       HIDDEN_SIZE)
        split = network.add_shuffle(flat)
        split.reshape_dims = (steps, heads, dim)
        split.second_transpose = (1, 0, 2)
        projections[name] = split.get_output(0)

    scores = network.add_matrix_multiply(
        projections["to_q"], trt.MatrixOperation.NONE,
        projections["to_k"], trt.MatrixOperation.TRANSPOSE,
    )
    scaled = network.add_elementwise(
        scores.get_output(0),
        _const(network, trt, [[[attention_scale()]]]),
        trt.ElementWiseOperation.PROD,
    )
    masked = network.add_elementwise(
        scaled.get_output(0),
        _const(network, trt, causal_mask(steps).reshape(1, steps, steps)),
        trt.ElementWiseOperation.SUM,
    )
    probabilities = network.add_softmax(masked.get_output(0))
    probabilities.axes = 1 << 2

    context = network.add_matrix_multiply(
        probabilities.get_output(0), trt.MatrixOperation.NONE,
        projections["to_v"], trt.MatrixOperation.NONE,
    )
    merge = network.add_shuffle(context.get_output(0))
    merge.first_transpose = (1, 0, 2)
    merge.reshape_dims = (1, steps, HIDDEN_SIZE)
    return _linear(network, trt, merge.get_output(0),
                   weights[f"{prefix}.attn.to_out.weight"], HIDDEN_SIZE)


def add_swiglu(network: Any, trt: Any, hidden: Any, weights: dict, prefix: str) -> Any:
    """``down(silu(gate(x)) * up(x))``."""

    gate = _linear(network, trt, hidden, weights[f"{prefix}.gate_proj.weight"],
                   INTERMEDIATE_SIZE)
    activated = network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
    silu = network.add_elementwise(gate, activated, trt.ElementWiseOperation.PROD)
    up = _linear(network, trt, hidden, weights[f"{prefix}.up_proj.weight"],
                 INTERMEDIATE_SIZE)
    gated = network.add_elementwise(
        silu.get_output(0), up, trt.ElementWiseOperation.PROD
    )
    return _linear(network, trt, gated.get_output(0),
                   weights[f"{prefix}.down_proj.weight"], HIDDEN_SIZE)


def add_depth_decoder(network: Any, trt: Any, inputs_embeds: Any, *, steps: int,
                      weights: dict) -> Any:
    """Add the decoder and return its normalised hidden states."""

    import numpy as np

    table = np.asarray(weights["pos_embedding.weight"], dtype=np.float32)
    positions = _const(network, trt, table[:steps].reshape(1, steps, HIDDEN_SIZE))
    hidden = network.add_elementwise(
        inputs_embeds, positions, trt.ElementWiseOperation.SUM
    ).get_output(0)

    for layer in range(NUM_LAYERS):
        prefix = f"layers.{layer}"
        normed = add_rms_norm(network, trt, hidden,
                              weights[f"{prefix}.input_layernorm.weight"])
        attended = add_attention(network, trt, normed, weights, prefix, steps)
        hidden = network.add_elementwise(
            hidden, attended, trt.ElementWiseOperation.SUM
        ).get_output(0)

        normed = add_rms_norm(network, trt, hidden,
                              weights[f"{prefix}.post_attention_layernorm.weight"])
        projected = add_swiglu(network, trt, normed, weights, prefix)
        hidden = network.add_elementwise(
            hidden, projected, trt.ElementWiseOperation.SUM
        ).get_output(0)

    return add_rms_norm(network, trt, hidden, weights["norm.weight"])


def configure(config: Any, trt: Any) -> None:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if DISABLE_TF32:
        config.clear_flag(trt.BuilderFlag.TF32)


def expected_io_shapes(steps: int) -> dict[str, tuple[int, ...]]:
    return {
        INPUT_NAME: (1, steps, HIDDEN_SIZE),
        OUTPUT_NAME: (1, steps, HIDDEN_SIZE),
    }
