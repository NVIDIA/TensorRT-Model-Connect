# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 TensorRT graph of the privacy-filter token classifier.

One request is one sequence of ``T`` token ids (``input_ids [T]``, dynamic) and
yields ``logits [T, labels]``. Per layer:

* RMSNorm, then Q/K/V projections with bias; interleaved-pair rotary
  embedding from host tables (YaRN gain included); Q and K each scaled by
  ``head_dim ** -0.25``.
* Grouped-query attention over the bidirectional band ``|i - j| <= window``
  (additive mask of the float32 minimum outside it). Each head appends its sink
  logit as one extra softmax column, which is then dropped.
* RMSNorm, router logits with bias, top-k over the raw logits and a softmax over
  the selected values. Every expert is evaluated with one GEMM; its clamped
  SwiGLU output ``(up + 1) * gate * sigmoid(alpha * gate)`` goes through the
  per-expert down projection, and each token's selected outputs are gathered
  and summed with the routing weights.

The reference divides the routing weights by ``k`` and multiplies the expert
sum by ``k``; both are exact powers-of-two rescalings in float32, so the graph
omits them.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .weights import LayerWeights, ModelWeights

INPUT_NAME = "input_ids"
OUTPUT_NAME = "logits"


class _Graph:
    """Small builder over one strongly typed float32 network; keeps every constant alive."""

    def __init__(self, network: trt.INetworkDefinition):
        self.network = network
        self._arrays: list[np.ndarray] = []

    def constant(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.ascontiguousarray(values)
        if array.dtype not in (np.float32, np.int32, np.int64):
            raise TypeError(f"unsupported constant dtype {array.dtype}")
        self._arrays.append(array)
        return self.network.add_constant(
            tuple(shape or array.shape), trt.Weights(array)
        ).get_output(0)

    def scalar(self, value: float, rank: int) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value, dtype=np.float32))

    def shape_of(self, *parts) -> trt.ITensor:
        """Concatenate int64 shape pieces: ints become constants, tensors are used as given."""
        tensors = [
            self.constant(np.array([part], dtype=np.int64)) if isinstance(part, int) else part
            for part in parts
        ]
        layer = self.network.add_concatenation(tensors)
        layer.axis = 0
        return layer.get_output(0)

    def elementwise(self, left, right, op) -> trt.ITensor:
        return self.network.add_elementwise(left, right, op).get_output(0)

    def add(self, left, right):
        return self.elementwise(left, right, trt.ElementWiseOperation.SUM)

    def sub(self, left, right):
        return self.elementwise(left, right, trt.ElementWiseOperation.SUB)

    def mul(self, left, right):
        return self.elementwise(left, right, trt.ElementWiseOperation.PROD)

    def reshape(self, tensor, dims, first=None, second=None) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        if first is not None:
            layer.first_transpose = first
        if dims is not None:
            layer.reshape_dims = dims
        if second is not None:
            layer.second_transpose = second
        return layer.get_output(0)

    def slice(self, tensor, start, shape: trt.ITensor, stride) -> trt.ITensor:
        layer = self.network.add_slice(tensor, start, (1,) * len(start), stride)
        layer.set_input(2, shape)
        return layer.get_output(0)

    def linear(self, tensor, weight: np.ndarray, bias: np.ndarray) -> trt.ITensor:
        product = self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, self.constant(weight), trt.MatrixOperation.NONE
        ).get_output(0)
        return self.add(product, self.constant(bias, (1, bias.shape[0])))

    def rms_norm(self, tensor, weight: np.ndarray, eps: float) -> trt.ITensor:
        mean = self.network.add_reduce(
            self.mul(tensor, tensor), trt.ReduceOperation.AVG, 1 << 1, True
        ).get_output(0)
        root = self.network.add_unary(self.add(mean, self.scalar(eps, 2)), trt.UnaryOperation.SQRT)
        inverse = self.network.add_unary(root.get_output(0), trt.UnaryOperation.RECIP).get_output(0)
        return self.mul(self.mul(tensor, inverse), self.constant(weight, (1, weight.shape[0])))


def _rotate(graph: _Graph, tensor, heads: int, length, cos, sin, half: int) -> trt.ITensor:
    """Interleaved-pair rotary embedding of ``tensor [heads, T, 2 * half]``."""
    shape = graph.shape_of(heads, length, half)
    even = graph.slice(tensor, (0, 0, 0), shape, (1, 1, 2))
    odd = graph.slice(tensor, (0, 0, 1), shape, (1, 1, 2))
    first = graph.sub(graph.mul(even, cos), graph.mul(odd, sin))
    second = graph.add(graph.mul(odd, cos), graph.mul(even, sin))
    pair = graph.network.add_concatenation(
        [graph.reshape(first, (0, 0, 0, 1)), graph.reshape(second, (0, 0, 0, 1))]
    )
    pair.axis = 3
    return graph.reshape(pair.get_output(0), (0, 0, 2 * half))


def _attention(
    graph: _Graph, config: ModelConfig, layer: LayerWeights, hidden, length, cos, sin, band
) -> trt.ITensor:
    heads, kv_heads, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    scale = graph.scalar(float(np.float32(head_dim**-0.25)), 3)

    def project(weight, bias, count):
        rows = graph.linear(hidden, weight, bias)  # [T, count * head_dim]
        return graph.reshape(rows, (0, count, head_dim), second=(1, 0, 2))  # [count, T, head_dim]

    query = project(layer.q_weight, layer.q_bias, heads)
    key = project(layer.k_weight, layer.k_bias, kv_heads)
    value = project(layer.v_weight, layer.v_bias, kv_heads)
    query = graph.mul(_rotate(graph, query, heads, length, cos, sin, head_dim // 2), scale)
    key = graph.mul(_rotate(graph, key, kv_heads, length, cos, sin, head_dim // 2), scale)

    grouped = graph.reshape(query, (kv_heads, -1, head_dim))  # [kv, group * T, head_dim]
    scores = graph.network.add_matrix_multiply(
        grouped, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = graph.add(graph.reshape(scores, (heads, -1, 0)), band)  # [heads, T, T]
    first_column = graph.slice(scores, (0, 0, 0), graph.shape_of(heads, length, 1), (1, 1, 1))
    sinks = graph.add(
        graph.mul(first_column, graph.scalar(0.0, 3)), graph.constant(layer.sinks, (heads, 1, 1))
    )
    with_sinks = graph.network.add_concatenation([scores, sinks])
    with_sinks.axis = 2
    softmax = graph.network.add_softmax(with_sinks.get_output(0))
    softmax.axes = 1 << 2
    probs = graph.slice(
        softmax.get_output(0), (0, 0, 0), graph.shape_of(heads, length, length), (1, 1, 1)
    )
    context = graph.network.add_matrix_multiply(
        graph.reshape(probs, (kv_heads, -1, 0)),
        trt.MatrixOperation.NONE,
        value,
        trt.MatrixOperation.NONE,
    ).get_output(0)  # [kv, group * T, head_dim]
    context = graph.reshape(
        context, (heads, -1, head_dim), second=(1, 0, 2)
    )  # [T, heads, head_dim]
    context = graph.reshape(context, (0, heads * head_dim))
    return graph.linear(context, layer.o_weight, layer.o_bias)


def _experts(
    graph: _Graph, config: ModelConfig, layer: LayerWeights, hidden, length
) -> trt.ITensor:
    experts, inner, top_k = (
        config.num_local_experts,
        config.intermediate_size,
        config.num_experts_per_tok,
    )
    width = config.hidden_size
    router = graph.linear(hidden, layer.router_weight, layer.router_bias)  # [T, experts]
    top = graph.network.add_topk(router, trt.TopKOperation.MAX, top_k, 1 << 1)
    routing = graph.network.add_softmax(top.get_output(0))
    routing.axes = 1 << 1

    gate_up = graph.network.add_matrix_multiply(
        hidden,
        trt.MatrixOperation.NONE,
        graph.constant(layer.gate_up_weight),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    gate_up = graph.add(
        graph.reshape(gate_up, (0, experts, 2 * inner)),
        graph.constant(layer.gate_up_bias, (1, experts, 2 * inner)),
    )
    half = graph.shape_of(length, experts, inner)
    gate = graph.slice(gate_up, (0, 0, 0), half, (1, 1, 1))
    linear = graph.slice(gate_up, (0, 0, inner), half, (1, 1, 1))
    limit = graph.scalar(config.swiglu_limit, 3)
    gate = graph.elementwise(gate, limit, trt.ElementWiseOperation.MIN)
    linear = graph.elementwise(
        graph.elementwise(linear, limit, trt.ElementWiseOperation.MIN),
        graph.scalar(-config.swiglu_limit, 3),
        trt.ElementWiseOperation.MAX,
    )
    sigmoid = graph.network.add_activation(
        graph.mul(gate, graph.scalar(config.swiglu_alpha, 3)), trt.ActivationType.SIGMOID
    )
    activated = graph.mul(
        graph.add(linear, graph.scalar(1.0, 3)), graph.mul(gate, sigmoid.get_output(0))
    )

    per_expert = graph.reshape(activated, None, first=(1, 0, 2))  # [experts, T, inner]
    down = graph.network.add_matrix_multiply(
        per_expert,
        trt.MatrixOperation.NONE,
        graph.constant(layer.down_weight),
        trt.MatrixOperation.NONE,
    ).get_output(0)
    down = graph.add(
        down, graph.constant(layer.down_bias, (experts, 1, width))
    )  # [experts, T, hidden]
    by_token = graph.reshape(down, None, first=(1, 0, 2))  # [T, experts, hidden]
    indices = graph.reshape(top.get_output(1), (0, top_k, 1))
    gather = graph.network.add_gather_v2(by_token, indices, trt.GatherMode.ND)
    gather.num_elementwise_dims = 1
    weights = graph.reshape(routing.get_output(0), (0, top_k, 1))
    weighted = graph.mul(gather.get_output(0), weights)  # [T, top_k, hidden]
    return graph.network.add_reduce(weighted, trt.ReduceOperation.SUM, 1 << 1, False).get_output(0)


def build_network(
    network: trt.INetworkDefinition, config: ModelConfig, weights: ModelWeights, max_length: int
) -> _Graph:
    """Populate ``network``; the returned builder owns the constant arrays until serialization."""
    graph = _Graph(network)
    input_ids = network.add_input(INPUT_NAME, trt.int32, (-1,))
    length = network.add_shape(input_ids).get_output(0)  # int64 [1]
    hidden = network.add_gather(graph.constant(weights.embedding), input_ids, 0).get_output(
        0
    )  # [T, hidden]

    half = config.head_dim // 2
    cos_table, sin_table = config.rope.tables(config.head_dim, max_length)
    table_shape = graph.shape_of(length, half)
    cos = graph.reshape(
        graph.slice(graph.constant(cos_table), (0, 0), table_shape, (1, 1)), (1, -1, half)
    )
    sin = graph.reshape(
        graph.slice(graph.constant(sin_table), (0, 0), table_shape, (1, 1)), (1, -1, half)
    )

    positions = graph.slice(
        graph.constant(np.arange(max_length, dtype=np.int32)), (0,), length, (1,)
    )
    rows = graph.reshape(positions, (1, -1, 1))
    columns = graph.reshape(positions, (1, 1, -1))
    window = graph.constant(np.array([[[config.sliding_window + 1]]], dtype=np.int32))
    inside = graph.elementwise(
        graph.elementwise(graph.sub(rows, columns), window, trt.ElementWiseOperation.LESS),
        graph.elementwise(graph.sub(columns, rows), window, trt.ElementWiseOperation.LESS),
        trt.ElementWiseOperation.AND,
    )  # [1, T, T]: |i - j| <= window
    band = network.add_select(
        inside, graph.scalar(0.0, 3), graph.scalar(float(np.finfo(np.float32).min), 3)
    ).get_output(0)

    for layer in weights.layers:
        normed = graph.rms_norm(hidden, layer.input_norm, config.rms_norm_eps)
        hidden = graph.add(hidden, _attention(graph, config, layer, normed, length, cos, sin, band))
        normed = graph.rms_norm(hidden, layer.post_attention_norm, config.rms_norm_eps)
        hidden = graph.add(hidden, _experts(graph, config, layer, normed, length))
    hidden = graph.rms_norm(hidden, weights.final_norm, config.rms_norm_eps)
    logits = graph.linear(hidden, weights.score_weight, weights.score_bias)
    logits.name = OUTPUT_NAME
    network.mark_output(logits)
    return graph
