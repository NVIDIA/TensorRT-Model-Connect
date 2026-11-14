# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-modal fusion and the text enhancer.

Two blocks run before the deformable attention in every encoder layer.

``GroundingDinoBiMultiHeadAttention`` is a bidirectional cross-attention: one
score matrix ``vision_query @ text_key^T`` is softmaxed along the text axis to
update the vision tokens and, transposed, along the vision axis to update the
text tokens. Four heads at an inner width of 1024, so the head dimension is
256 - wider than the model width of 256 per head would suggest.

Three rules read from the reference, each shape-invariant:

* **the residual is taken from the normalised value, not the input.** The
  reference rebinds ``vision_features = self.layer_norm_vision(...)`` and then
  adds to that. Ordinary pre-norm, which keeps the original tensor as the
  residual, is the natural reading and is wrong;
* the update is gated by a learned **per-channel** vector, ``vision_param``
  and ``text_param``, not a scalar;
* only the vision query carries the ``head_dim ** -0.5`` scale.

The reference also subtracts a global scalar maximum and clamps to +/-50000
before each softmax. Both are numerical-stability steps: a softmax is
invariant to a constant shift, and the clamp never binds at these magnitudes,
so neither is reproduced here.

The text enhancer that follows is post-norm self-attention over the text
tokens, with the same posed-query rule the rest of this model uses - q and k
see the position embeddings, v does not. Its ``layer_norm_before`` and
``layer_norm_after`` are **both** applied after their residual add; the names
describe position in the block, not pre- versus post-norm.

Two of its rules were settled by measuring the reference's own attention
weights, because reading the call chain gives the wrong answer:

* **the phrase block mask marks what is allowed.** The call site hands the
  enhancer ``~text_self_attention_masks`` and then applies
  ``(1 - mask) * dtype.min``, which reads as though within-phrase attention is
  masked out. Dumping the reference attention shows the opposite: every row
  attends exactly to its own phrase and nowhere else. The negations cancel
  somewhere in the chain; the measured pattern is what this reproduces;
* **the sine position embedding uses the per-phrase position ids**, the same
  ``[0, 0, 1, 0, 0, 1, 2, 0, 0]`` the text tower uses - not the plain token
  index. Using ``arange`` scores 0.9993 with a maximum error of 0.47, which is
  the kind of near-miss an absolute cosine gate would wave through.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-5


def _heads(network, x, length, heads, head_dim):
    """(1, length, heads * head_dim) -> (heads, length, head_dim)."""
    shuffle = network.add_shuffle(x)
    shuffle.reshape_dims = (length, heads, head_dim)
    shuffle.second_transpose = (1, 0, 2)
    return shuffle.get_output(0)


def _unheads(network, x, length, heads, head_dim):
    """(heads, length, head_dim) -> (1, length, heads * head_dim)."""
    shuffle = network.add_shuffle(x)
    shuffle.first_transpose = (1, 0, 2)
    shuffle.reshape_dims = (1, length, heads * head_dim)
    return shuffle.get_output(0)


def build_fusion(network, vision, text, weights, prefix, heads, dtype=np.float32,
                 vision_mask=None, text_mask=None):
    """One bidirectional vision/text fusion block.

    ``vision_mask`` and ``text_mask`` are additive terms already shaped to
    broadcast over ``(heads, query, key)``.
    """
    vision_len = int(vision.shape[1])
    text_len = int(text.shape[1])
    width = int(vision.shape[2])

    normed_vision = g.add_layer_norm(
        network, vision, weights[f"{prefix}.layer_norm_vision.weight"],
        weights[f"{prefix}.layer_norm_vision.bias"], _LAYER_NORM_EPS, dtype=dtype)
    normed_text = g.add_layer_norm(
        network, text, weights[f"{prefix}.layer_norm_text.weight"],
        weights[f"{prefix}.layer_norm_text.bias"], _LAYER_NORM_EPS, dtype=dtype)

    inner = int(np.asarray(weights[f"{prefix}.attn.vision_proj.weight"]).shape[0])
    head_dim = inner // heads
    scale = float(head_dim) ** -0.5

    def project(source, name, length):
        out = g.add_linear(network, source, weights[f"{prefix}.attn.{name}.weight"],
                           weights[f"{prefix}.attn.{name}.bias"], dtype=dtype)
        return _heads(network, out, length, heads, head_dim)

    vision_query = project(normed_vision, "vision_proj", vision_len)
    text_key = project(normed_text, "text_proj", text_len)
    vision_value = project(normed_vision, "values_vision_proj", vision_len)
    text_value = project(normed_text, "values_text_proj", text_len)

    # Only the query is scaled.
    vision_query = network.add_elementwise(
        vision_query, g.add_constant(network, (1, 1, 1), np.full((1,), scale), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)

    scores = network.add_matrix_multiply(
        vision_query, trt.MatrixOperation.NONE,
        text_key, trt.MatrixOperation.TRANSPOSE).get_output(0)

    def attend(matrix, mask, axis_len, value, length):
        if mask is not None:
            matrix = network.add_elementwise(
                matrix, mask, trt.ElementWiseOperation.SUM).get_output(0)
        softmax = network.add_softmax(matrix)
        softmax.axes = 1 << 2
        context = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE,
            value, trt.MatrixOperation.NONE).get_output(0)
        return _unheads(network, context, length, heads, head_dim)

    # Vision tokens attend over text; text tokens attend over vision.
    vision_delta = attend(scores, text_mask, text_len, text_value, vision_len)
    transposed = network.add_shuffle(scores)
    transposed.first_transpose = (0, 2, 1)
    text_delta = attend(transposed.get_output(0), vision_mask, vision_len,
                        vision_value, text_len)

    vision_delta = g.add_linear(
        network, vision_delta, weights[f"{prefix}.attn.out_vision_proj.weight"],
        weights[f"{prefix}.attn.out_vision_proj.bias"], dtype=dtype)
    text_delta = g.add_linear(
        network, text_delta, weights[f"{prefix}.attn.out_text_proj.weight"],
        weights[f"{prefix}.attn.out_text_proj.bias"], dtype=dtype)

    def gate(normed, delta, name, width_out):
        param = np.asarray(weights[f"{prefix}.{name}"], dtype=dtype).reshape(1, 1, width_out)
        scaled = network.add_elementwise(
            delta, g.add_constant(network, param.shape, param, dtype),
            trt.ElementWiseOperation.PROD).get_output(0)
        # The residual is the NORMALISED tensor, not the block input.
        return g.add_sum(network, normed, scaled)

    return (gate(normed_vision, vision_delta, "vision_param", width),
            gate(normed_text, text_delta, "text_param", width))


def build_text_enhancer(network, text, weights, prefix, heads, dtype=np.float32,
                        mask=None, position=None):
    """Post-norm self-attention over the text tokens."""
    length = int(text.shape[1])
    width = int(text.shape[2])
    head_dim = width // heads

    posed = text
    if position is not None:
        posed = network.add_elementwise(
            text, position, trt.ElementWiseOperation.SUM).get_output(0)

    def project(source, name):
        out = g.add_linear(network, source, weights[f"{prefix}.self_attn.{name}.weight"],
                           weights[f"{prefix}.self_attn.{name}.bias"], dtype=dtype)
        return _heads(network, out, length, heads, head_dim)

    # q and k are posed; v is not.
    query, key, value = project(posed, "query"), project(posed, "key"), project(text, "value")
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, g.add_constant(network, (1, 1, 1), np.full((1,), 1.0 / math.sqrt(head_dim)),
                               dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)
    if mask is not None:
        scores = network.add_elementwise(
            scores, mask, trt.ElementWiseOperation.SUM).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value,
        trt.MatrixOperation.NONE).get_output(0)
    attended = g.add_linear(
        network, _unheads(network, context, length, heads, head_dim),
        weights[f"{prefix}.self_attn.out_proj.weight"],
        weights[f"{prefix}.self_attn.out_proj.bias"], dtype=dtype)

    hidden = g.add_layer_norm(
        network, g.add_sum(network, text, attended),
        weights[f"{prefix}.layer_norm_before.weight"],
        weights[f"{prefix}.layer_norm_before.bias"], _LAYER_NORM_EPS, dtype=dtype)

    inner = g.add_relu(network, g.add_linear(
        network, hidden, weights[f"{prefix}.fc1.weight"],
        weights[f"{prefix}.fc1.bias"], dtype=dtype))
    inner = g.add_linear(network, inner, weights[f"{prefix}.fc2.weight"],
                         weights[f"{prefix}.fc2.bias"], dtype=dtype)
    return g.add_layer_norm(
        network, g.add_sum(network, hidden, inner),
        weights[f"{prefix}.layer_norm_after.weight"],
        weights[f"{prefix}.layer_norm_after.bias"], _LAYER_NORM_EPS, dtype=dtype)
