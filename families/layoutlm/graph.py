# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT network of LayoutLMForQuestionAnswering.

One window of ``S`` tokens in (``input_ids [S]``, ``bbox [S, 4]``), the start and
end logits of every token out. ``S`` is dynamic up to the bundle's window, so a
window is never padded and needs no attention mask.

The embedding is BERT's plus six layout lookups: the box corners index the x and
y tables (left and right share the x table, top and bottom the y table) and the
box height and width index their own tables. The encoder is a post-norm BERT.
Embedding sums, every LayerNorm and softmax, and the accumulation of every GEMM
run in float32 at either work precision; the token-type row is the first one,
the only type this tokenizer emits.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import PREFIX

_WORK_TYPES = {"fp32": (np.float32, trt.float32), "fp16": (np.float16, trt.float16)}


class _Graph:
    def __init__(self, network, precision: str):
        self.network = network
        self.work_np, self.work_trt = _WORK_TYPES[precision]
        # The network refers to constant buffers until the engine is built.
        self.keep = []

    def constant(self, values, dtype=None):
        array = np.asarray(values, dtype=dtype or self.work_np)
        # np.ascontiguousarray would turn a 0-d index into shape (1,); copy only when needed.
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        self.keep.append(array)
        return self.network.add_constant(array.shape, array).get_output(0)

    def cast(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def binary(self, left, right, operation):
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def matmul(self, left, left_op, right, right_op, keep_float32=False):
        """A GEMM that accumulates in float32 at either work precision.

        In a strongly typed fp16 network TensorRT may pick fp16-accumulating
        GEMM tactics, which tripled the logit error of this model against the
        reference's own fp16 run. fp16 operands are therefore cast to float32
        and multiplied with TF32 enabled (see model.build_engine): TF32 keeps
        the ten mantissa bits of an fp16 value, so the products are exact and
        the sum is float32. The product returns to the work type unless the
        caller keeps it in float32.
        """
        if self.work_trt == trt.float32:
            return self.network.add_matrix_multiply(left, left_op, right, right_op).get_output(0)
        product = self.network.add_matrix_multiply(
            self.cast(left, trt.float32), left_op, self.cast(right, trt.float32),
            right_op).get_output(0)
        return product if keep_float32 else self.cast(product, self.work_trt)

    def linear(self, tensor, weight, bias):
        """``tensor @ weight.T + bias`` for a ``[S, in]`` tensor and an ``[out, in]`` weight."""
        product = self.matmul(tensor, trt.MatrixOperation.NONE,
                              self.constant(np.asarray(weight).T), trt.MatrixOperation.NONE)
        return self.add(product, self.constant(np.asarray(bias).reshape(1, -1)))

    def layer_norm(self, tensor, weight, bias, eps):
        """LayerNorm over the last axis with float32 statistics; keeps the input type."""
        source = self.cast(tensor, trt.float32)
        layer = self.network.add_normalization(
            source, self.constant(np.asarray(weight).reshape(1, -1), np.float32),
            self.constant(np.asarray(bias).reshape(1, -1), np.float32), 1 << 1)
        layer.epsilon = eps
        return self.cast(layer.get_output(0), tensor.dtype)

    def gather(self, table, indices):
        return self.network.add_gather(table, indices, 0).get_output(0)

    def column(self, tensor, index):
        """``tensor[:, index]`` of a ``[S, 4]`` tensor as ``[S]``."""
        return self.network.add_gather(
            tensor, self.constant(np.array(index), np.int32), 1).get_output(0)


def _embeddings(graph: _Graph, weights, config, input_ids, bbox):
    network = graph.network
    name = f"{PREFIX}embeddings."

    def table(key):
        return graph.constant(weights[name + key], np.float32)

    hidden = config["hidden_size"]
    # Rows 0..S-1 of the position table: the reference's arange position ids.
    size = network.add_concatenation(
        [network.add_shape(input_ids).get_output(0),
         graph.constant(np.array([hidden]), np.int64)]).get_output(0)
    positions = network.add_slice(table("position_embeddings.weight"), (0, 0), (1, hidden),
                                  (1, 1))
    positions.set_input(2, size)

    left, top, right, bottom = (graph.column(bbox, index) for index in range(4))
    height = graph.binary(bottom, top, trt.ElementWiseOperation.SUB)
    width = graph.binary(right, left, trt.ElementWiseOperation.SUB)
    x_table = table("x_position_embeddings.weight")
    y_table = table("y_position_embeddings.weight")
    terms = (
        graph.gather(table("word_embeddings.weight"), input_ids),
        positions.get_output(0),
        graph.gather(x_table, left),
        graph.gather(y_table, top),
        graph.gather(x_table, right),
        graph.gather(y_table, bottom),
        graph.gather(table("h_position_embeddings.weight"), height),
        graph.gather(table("w_position_embeddings.weight"), width),
        graph.constant(weights[name + "token_type_embeddings.weight"][:1], np.float32),
    )
    # Summed left to right in the reference's order.
    total = terms[0]
    for term in terms[1:]:
        total = graph.add(total, term)
    normed = graph.layer_norm(total, weights[name + "LayerNorm.weight"],
                              weights[name + "LayerNorm.bias"], config["layer_norm_eps"])
    return graph.cast(normed, graph.work_trt)


def _attention(graph: _Graph, hidden_states, weights, base, config):
    network = graph.network
    heads = config["num_heads"]
    head_dim = config["hidden_size"] // heads

    def split(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (0, heads, head_dim)
        layer.second_transpose = (1, 0, 2)
        return layer.get_output(0)

    def project(name):
        return split(graph.linear(hidden_states, weights[f"{base}attention.self.{name}.weight"],
                                  weights[f"{base}attention.self.{name}.bias"]))

    query, key, value = project("query"), project("key"), project("value")
    scores = graph.matmul(query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE,
                          keep_float32=True)
    # matmul(q, k^T) * head_dim**-0.5, then a float32 softmax over the keys.
    scores = graph.binary(scores,
                          graph.constant(np.full((1, 1, 1), head_dim ** -0.5), np.float32),
                          trt.ElementWiseOperation.PROD)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    probabilities = graph.cast(softmax.get_output(0), graph.work_trt)
    context = graph.matmul(probabilities, trt.MatrixOperation.NONE, value,
                           trt.MatrixOperation.NONE)
    merge = network.add_shuffle(context)
    merge.first_transpose = (1, 0, 2)
    merge.reshape_dims = (0, config["hidden_size"])
    return merge.get_output(0)


def _layer(graph: _Graph, hidden_states, weights, index, config):
    base = f"{PREFIX}encoder.layer.{index}."
    eps = config["layer_norm_eps"]
    context = _attention(graph, hidden_states, weights, base, config)
    attended = graph.layer_norm(
        graph.add(graph.linear(context, weights[base + "attention.output.dense.weight"],
                               weights[base + "attention.output.dense.bias"]), hidden_states),
        weights[base + "attention.output.LayerNorm.weight"],
        weights[base + "attention.output.LayerNorm.bias"], eps)
    inner = graph.network.add_activation(
        graph.linear(attended, weights[base + "intermediate.dense.weight"],
                     weights[base + "intermediate.dense.bias"]),
        trt.ActivationType.GELU_ERF).get_output(0)
    return graph.layer_norm(
        graph.add(graph.linear(inner, weights[base + "output.dense.weight"],
                               weights[base + "output.dense.bias"]), attended),
        weights[base + "output.LayerNorm.weight"], weights[base + "output.LayerNorm.bias"], eps)


def build_network(network, weights, config, precision: str) -> _Graph:
    """Populate ``network`` (strongly typed); returns the graph holding its constants."""
    graph = _Graph(network, precision)
    input_ids = network.add_input("input_ids", trt.int32, (-1,))
    bbox = network.add_input("bbox", trt.int32, (-1, 4))
    hidden_states = _embeddings(graph, weights, config, input_ids, bbox)
    for index in range(config["num_layers"]):
        hidden_states = _layer(graph, hidden_states, weights, index, config)
    logits = graph.cast(graph.linear(hidden_states, weights["qa_outputs.weight"],
                                     weights["qa_outputs.bias"]), trt.float32)
    columns = network.add_shuffle(logits)
    columns.first_transpose = (1, 0)
    for index, name in enumerate(("start_logits", "end_logits")):
        output = network.add_gather(columns.get_output(0),
                                    graph.constant(np.array(index), np.int32), 0).get_output(0)
        output.name = name
        network.mark_output(output)
    return graph
