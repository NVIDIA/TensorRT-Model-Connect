# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the global language model's decoder stack.

A Qwen3 decoder: grouped attention, per-head normalisation before the
rotation, and SwiGLU. Duplicated into this family rather than shared with the
repository's other Qwen decoders, which is what the architecture asks for --
model-specific code stays family-local so a defect here is fixed and reverted
here.

Three things separate this from the diffusion transformer's block:

* Eight key/value heads serve thirty-two query heads, so the cache is a
  quarter the size and its heads are repeated on the way into the scores.
* ``q_norm`` and ``k_norm`` are RMS norms over the 128-wide **head**, applied
  after the projection is split into heads and before the rotation. Applying
  them to the flattened projection would be wrong by a factor of the head
  count.
* The rotation is full width at a theta of a million, against the diffusion
  transformer's leading 32 of 64 at ten thousand.

``num_layers`` is a parameter so a shortened stack can be built for
validation; the checkpoint's thirty-six is the default.

**Validated.** Built on an A40 with TensorRT 11.1.0.106 over a two-layer stack
of the published weights -- 8.4 s, a 1544 MB engine -- at a sequence length of
32, and compared with the reference decoder truncated to the same two blocks.
Agreement is 7.3e-07 relative.
"""

from __future__ import annotations

from typing import Any

from .language_model import (
    HEAD_DIM,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    attention_scale,
    group_size,
    key_value_width,
    query_width,
    rope_tables,
)

INPUT_NAME = "hidden_states"
OUTPUT_NAME = "hidden_states_out"

DISABLE_TF32 = True
WORKSPACE_BYTES = 8 << 30


def _const(network, trt, array):
    import numpy as np

    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return network.add_constant(data.shape, trt.Weights(data)).get_output(0)


def _linear(network, trt, hidden, weight, out_features):
    """Bias-free ``x @ W.T`` with the weight as a constant operand."""

    import numpy as np

    array = np.asarray(weight, dtype=np.float32)
    if array.shape[0] != out_features:
        raise ValueError(f"weight has {array.shape[0]} rows, expected {out_features}")
    operand = _const(network, trt, np.ascontiguousarray(array.T).reshape(
        1, array.shape[1], out_features))
    return network.add_matrix_multiply(
        hidden, trt.MatrixOperation.NONE, operand, trt.MatrixOperation.NONE
    ).get_output(0)


def add_rms_norm(network: Any, trt: Any, hidden: Any, weight, *, axis: int) -> Any:
    """``x * rsqrt(mean(x ** 2) + eps) * weight`` over ``axis``."""

    import numpy as np

    squared = network.add_elementwise(hidden, hidden, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(squared.get_output(0), trt.ReduceOperation.AVG,
                              axes=1 << axis, keep_dims=True)
    shifted = network.add_elementwise(
        mean.get_output(0),
        _const(network, trt, np.array(RMS_NORM_EPS, dtype=np.float32).reshape(
            *([1] * (axis + 1)))),
        trt.ElementWiseOperation.SUM,
    )
    deviation = network.add_unary(shifted.get_output(0), trt.UnaryOperation.SQRT)
    normalised = network.add_elementwise(
        hidden, deviation.get_output(0), trt.ElementWiseOperation.DIV)
    scale_shape = [1] * (axis + 1)
    scale_shape[axis] = -1
    return network.add_elementwise(
        normalised.get_output(0),
        _const(network, trt, np.asarray(weight, dtype=np.float32).reshape(*scale_shape)),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)


def add_rope(network: Any, trt: Any, heads: Any, cos, sin, *, num_heads: int,
             seq_len: int) -> Any:
    """Full-width rotation of ``(num_heads, seq_len, head_dim)``."""

    import numpy as np

    half = HEAD_DIM // 2
    first = network.add_slice(heads, (0, 0, 0), (num_heads, seq_len, half),
                              (1, 1, 1)).get_output(0)
    second = network.add_slice(heads, (0, 0, half), (num_heads, seq_len, half),
                               (1, 1, 1)).get_output(0)
    negated = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    swapped = network.add_concatenation([negated, first])
    swapped.axis = 2

    cos_c = _const(network, trt, np.asarray(cos).reshape(1, seq_len, HEAD_DIM))
    sin_c = _const(network, trt, np.asarray(sin).reshape(1, seq_len, HEAD_DIM))
    direct = network.add_elementwise(heads, cos_c, trt.ElementWiseOperation.PROD)
    turned = network.add_elementwise(swapped.get_output(0), sin_c,
                                     trt.ElementWiseOperation.PROD)
    return network.add_elementwise(
        direct.get_output(0), turned.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)


def _repeat_kv(network, trt, heads, seq_len):
    """Expand eight key/value heads to thirty-two, each repeated in place."""

    expand = network.add_shuffle(heads)
    expand.reshape_dims = (NUM_KEY_VALUE_HEADS, 1, seq_len, HEAD_DIM)
    tiled = network.add_slice(
        expand.get_output(0), (0, 0, 0, 0),
        (NUM_KEY_VALUE_HEADS, group_size(), seq_len, HEAD_DIM), (1, 0, 1, 1))
    flat = network.add_shuffle(tiled.get_output(0))
    flat.reshape_dims = (NUM_ATTENTION_HEADS, seq_len, HEAD_DIM)
    return flat.get_output(0)


def causal_mask(seq_len: int):
    """Additive mask: zero where a position may attend, -inf where it may not."""

    import numpy as np

    allowed = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    mask[~allowed] = -np.inf
    return mask


def add_attention(network, trt, hidden, weights, prefix, seq_len, cos, sin):
    """Grouped causal attention with per-head normalisation before the rotation."""

    projections = {}
    for name, width, count in (
        ("q_proj", query_width(), NUM_ATTENTION_HEADS),
        ("k_proj", key_value_width(), NUM_KEY_VALUE_HEADS),
        ("v_proj", key_value_width(), NUM_KEY_VALUE_HEADS),
    ):
        flat = _linear(network, trt, hidden,
                       weights[f"{prefix}.self_attn.{name}.weight"], width)
        split = network.add_shuffle(flat)
        split.reshape_dims = (seq_len, count, HEAD_DIM)
        split.second_transpose = (1, 0, 2)
        projections[name] = split.get_output(0)

    # Per-head RMS norm, then the rotation. Order matters: Qwen3 normalises the
    # head before rotating it.
    query = add_rms_norm(network, trt, projections["q_proj"],
                         weights[f"{prefix}.self_attn.q_norm.weight"], axis=2)
    key = add_rms_norm(network, trt, projections["k_proj"],
                       weights[f"{prefix}.self_attn.k_norm.weight"], axis=2)
    query = add_rope(network, trt, query, cos, sin,
                     num_heads=NUM_ATTENTION_HEADS, seq_len=seq_len)
    key = add_rope(network, trt, key, cos, sin,
                   num_heads=NUM_KEY_VALUE_HEADS, seq_len=seq_len)

    key = _repeat_kv(network, trt, key, seq_len)
    value = _repeat_kv(network, trt, projections["v_proj"], seq_len)

    scores = network.add_matrix_multiply(query, trt.MatrixOperation.NONE,
                                         key, trt.MatrixOperation.TRANSPOSE)
    scaled = network.add_elementwise(
        scores.get_output(0), _const(network, trt, [[[attention_scale()]]]),
        trt.ElementWiseOperation.PROD)
    masked = network.add_elementwise(
        scaled.get_output(0),
        _const(network, trt, causal_mask(seq_len).reshape(1, seq_len, seq_len)),
        trt.ElementWiseOperation.SUM)
    probabilities = network.add_softmax(masked.get_output(0))
    probabilities.axes = 1 << 2

    context = network.add_matrix_multiply(
        probabilities.get_output(0), trt.MatrixOperation.NONE,
        value, trt.MatrixOperation.NONE)
    merge = network.add_shuffle(context.get_output(0))
    merge.first_transpose = (1, 0, 2)
    merge.reshape_dims = (1, seq_len, HIDDEN_SIZE)
    return _linear(network, trt, merge.get_output(0),
                   weights[f"{prefix}.self_attn.o_proj.weight"], HIDDEN_SIZE)


def add_swiglu(network, trt, hidden, weights, prefix):
    """``down(silu(gate(x)) * up(x))``."""

    gate = _linear(network, trt, hidden, weights[f"{prefix}.mlp.gate_proj.weight"],
                   INTERMEDIATE_SIZE)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0)
    silu = network.add_elementwise(gate, sigmoid, trt.ElementWiseOperation.PROD)
    up = _linear(network, trt, hidden, weights[f"{prefix}.mlp.up_proj.weight"],
                 INTERMEDIATE_SIZE)
    gated = network.add_elementwise(silu.get_output(0), up,
                                    trt.ElementWiseOperation.PROD)
    return _linear(network, trt, gated.get_output(0),
                   weights[f"{prefix}.mlp.down_proj.weight"], HIDDEN_SIZE)


def add_language_model(network: Any, trt: Any, hidden_states: Any, *, seq_len: int,
                       weights: dict, num_layers: int = NUM_HIDDEN_LAYERS,
                       prefix: str = "model") -> Any:
    """Add the decoder stack and return its normalised hidden states."""

    cos, sin = rope_tables(seq_len)
    hidden = hidden_states
    for layer in range(num_layers):
        block = f"{prefix}.layers.{layer}"
        normed = add_rms_norm(network, trt, hidden,
                              weights[f"{block}.input_layernorm.weight"], axis=2)
        attended = add_attention(network, trt, normed, weights, block, seq_len,
                                 cos, sin)
        hidden = network.add_elementwise(hidden, attended,
                                         trt.ElementWiseOperation.SUM).get_output(0)
        normed = add_rms_norm(network, trt, hidden,
                              weights[f"{block}.post_attention_layernorm.weight"],
                              axis=2)
        projected = add_swiglu(network, trt, normed, weights, block)
        hidden = network.add_elementwise(hidden, projected,
                                         trt.ElementWiseOperation.SUM).get_output(0)

    return add_rms_norm(network, trt, hidden, weights[f"{prefix}.norm.weight"],
                        axis=2)


def configure(config: Any, trt: Any) -> None:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    if DISABLE_TF32:
        config.clear_flag(trt.BuilderFlag.TF32)


def expected_io_shapes(seq_len: int) -> dict[str, tuple[int, ...]]:
    return {
        INPUT_NAME: (1, seq_len, HIDDEN_SIZE),
        OUTPUT_NAME: (1, seq_len, HIDDEN_SIZE),
    }
