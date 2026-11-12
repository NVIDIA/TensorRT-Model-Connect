# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MobileBERT pretraining graph in TensorRT, in float32.

Rows are ``[seq, width]``; attention works in ``[heads, seq, head_dim]``. One
layer of the published layout, with ``x`` the 512-wide stream::

    residual = no_norm(dense(x))            # bottleneck.input, 128 wide
    shared   = no_norm(dense(x))            # bottleneck.attention, 128 wide
    a = no_norm(dense(attention(q=shared, k=shared, v=x)) + residual)
    a = no_norm(dense(relu(dense(a))) + a)  # once per extra feed-forward block
    o = no_norm(dense(relu(dense(a))) + a)  # intermediate + output
    x = no_norm(dense(o) + x)               # output.bottleneck, back to 512

``no_norm`` is ``value * weight + bias``: no statistics anywhere in the
encoder. The stream is therefore not renormalised either, and the last layer
of the published checkpoint leaves it at magnitudes near 6e7 - beyond what
float16 can hold, which is why this family builds float32 only.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

_PREFIX = "mobilebert."

# Added to the scores of padded keys before the softmax: the most negative finite
# float32, the value transformers' eager mask adds (its SDPA path masks with a
# boolean), so their weight is exactly zero whatever the scores are. A small floor
# is not enough here: without normalisation the last layer's raw scores reach
# about 4.9e3 on ordinary text.
_MASK_FLOOR = float(np.finfo(np.float32).min)


class _Network:
    """A network plus the host arrays its constants point at.

    TensorRT reads constant weights when the engine is built, not when the
    layer is added, so every array handed to ``add_constant`` is kept alive
    here until the build has finished.
    """

    def __init__(self, network):
        self.network = network
        self._arrays: list[np.ndarray] = []

    def constant(self, values, shape=None) -> trt.ITensor:
        data = np.array(values, dtype=np.float32, copy=True, order="C")
        if shape is not None:
            data = data.reshape(shape)
        self._arrays.append(data)
        return self.network.add_constant(data.shape, data).get_output(0)

    def elementwise(self, left, right, operation) -> trt.ITensor:
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right) -> trt.ITensor:
        return self.elementwise(left, right, trt.ElementWiseOperation.SUM)

    def mul(self, left, right) -> trt.ITensor:
        return self.elementwise(left, right, trt.ElementWiseOperation.PROD)

    def relu(self, tensor) -> trt.ITensor:
        return self.network.add_activation(tensor, trt.ActivationType.RELU).get_output(0)

    def linear(self, tensor, weight, bias=None) -> trt.ITensor:
        """``tensor @ weight.T + bias`` for a 2-D ``tensor``, torch.nn.Linear's layout."""
        weight = np.asarray(weight, dtype=np.float32)
        out = self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE,
            self.constant(weight), trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is None:
            return out
        return self.add(out, self.constant(bias, (1, -1)))

    def no_norm(self, tensor, weight, bias) -> trt.ITensor:
        """MobileBERT's ``NoNorm``: an elementwise affine map, no statistics."""
        return self.add(self.mul(tensor, self.constant(weight, (1, -1))),
                        self.constant(bias, (1, -1)))

    def layer_norm(self, tensor, weight, bias, eps) -> trt.ITensor:
        """A real layer normalisation over the last axis of ``[rows, width]``.

        Written out as mean, centred variance and division. Its input here is
        the transformed last-layer row, whose entries reach about 1e9, so the
        variance is taken from the centred values rather than as
        ``mean(x * x) - mean(x) ** 2``.
        """
        mean = self.network.add_reduce(tensor, trt.ReduceOperation.AVG, 1 << 1,
                                       True).get_output(0)
        centred = self.elementwise(tensor, mean, trt.ElementWiseOperation.SUB)
        variance = self.network.add_reduce(self.mul(centred, centred), trt.ReduceOperation.AVG,
                                           1 << 1, True).get_output(0)
        shifted = self.add(variance, self.constant(np.full((1, 1), eps, np.float32)))
        deviation = self.network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
        normed = self.elementwise(centred, deviation, trt.ElementWiseOperation.DIV)
        return self.no_norm(normed, weight, bias)

    def reshape(self, tensor, shape, *, first=None, second=None) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        if first is not None:
            layer.first_transpose = first
        layer.reshape_dims = shape
        if second is not None:
            layer.second_transpose = second
        return layer.get_output(0)

    def rows(self, tensor, start, count) -> trt.ITensor:
        width = int(tensor.shape[1])
        return self.network.add_slice(tensor, (start, 0), (count, width), (1, 1)).get_output(0)

    def concat(self, tensors, axis) -> trt.ITensor:
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)


def _trigram(net: _Network, tokens: trt.ITensor, seq_len: int, width: int) -> trt.ITensor:
    """``[next, this, previous]`` token embeddings per row, zero beyond either end.

    The reference concatenates, per position, the embedding of the following
    row, of the row itself and of the preceding row, with zeros past the first
    and the last row of the input.
    """
    zero = net.constant(np.zeros((1, width), np.float32))
    if seq_len == 1:
        return net.concat([zero, tokens, zero], axis=1)
    following = net.concat([net.rows(tokens, 1, seq_len - 1), zero], axis=0)
    preceding = net.concat([zero, net.rows(tokens, 0, seq_len - 1)], axis=0)
    return net.concat([following, tokens, preceding], axis=1)


def _embeddings(net, weights, cfg, input_ids, token_type_ids, keep_column, seq_len):
    prefix = f"{_PREFIX}embeddings."
    table = net.constant(weights[f"{prefix}word_embeddings.weight"])
    tokens = net.network.add_gather(table, input_ids, 0).get_output(0)
    # Padded rows are zeroed before the trigram reads them, so the last real row
    # sees zeros on its right exactly as in an unpadded reference run, whatever
    # the checkpoint stores for the padding id.
    tokens = net.mul(tokens, keep_column)
    hidden = net.linear(_trigram(net, tokens, seq_len, cfg["embedding_size"]),
                        weights[f"{prefix}embedding_transformation.weight"],
                        weights[f"{prefix}embedding_transformation.bias"])
    positions = np.asarray(weights[f"{prefix}position_embeddings.weight"],
                           dtype=np.float32)[:seq_len]
    hidden = net.add(hidden, net.constant(positions))
    types = net.network.add_gather(
        net.constant(weights[f"{prefix}token_type_embeddings.weight"]), token_type_ids,
        0).get_output(0)
    hidden = net.add(hidden, types)
    return net.no_norm(hidden, weights[f"{prefix}LayerNorm.weight"],
                       weights[f"{prefix}LayerNorm.bias"])


def _dense(net, weights, module, tensor):
    return net.linear(tensor, weights[f"{module}.dense.weight"], weights[f"{module}.dense.bias"])


def _dense_no_norm(net, weights, module, tensor, residual=None):
    """``no_norm(dense(tensor) [+ residual])`` with the module's dense and LayerNorm tensors."""
    out = _dense(net, weights, module, tensor)
    if residual is not None:
        out = net.add(out, residual)
    return net.no_norm(out, weights[f"{module}.LayerNorm.weight"],
                       weights[f"{module}.LayerNorm.bias"])


def _attention(net, weights, layer, shared, stream, additive_mask, cfg, seq_len):
    heads, head_dim = cfg["num_heads"], cfg["head_dim"]

    def split(tensor):
        return net.reshape(tensor, (seq_len, heads, head_dim), second=(1, 0, 2))

    module = f"{layer}attention.self"
    query = split(net.linear(shared, weights[f"{module}.query.weight"],
                             weights[f"{module}.query.bias"]))
    key = split(net.linear(shared, weights[f"{module}.key.weight"],
                           weights[f"{module}.key.bias"]))
    value = split(net.linear(stream, weights[f"{module}.value.weight"],
                             weights[f"{module}.value.bias"]))
    scores = net.network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = net.mul(scores, net.constant(np.full((1, 1, 1), head_dim ** -0.5, np.float32)))
    scores = net.add(scores, additive_mask)
    softmax = net.network.add_softmax(scores)
    softmax.axes = 1 << 2
    context = net.network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        value, trt.MatrixOperation.NONE).get_output(0)
    return net.reshape(context, (seq_len, heads * head_dim), first=(1, 0, 2))


def _layer(net, weights, cfg, index, stream, additive_mask, seq_len):
    layer = f"{_PREFIX}encoder.layer.{index}."
    residual = _dense_no_norm(net, weights, f"{layer}bottleneck.input", stream)
    shared = _dense_no_norm(net, weights, f"{layer}bottleneck.attention", stream)
    context = _attention(net, weights, layer, shared, stream, additive_mask, cfg, seq_len)
    hidden = _dense_no_norm(net, weights, f"{layer}attention.output", context, residual)
    for ffn in range(cfg["extra_ffns"]):
        module = f"{layer}ffn.{ffn}"
        inner = net.relu(_dense(net, weights, f"{module}.intermediate", hidden))
        hidden = _dense_no_norm(net, weights, f"{module}.output", inner, hidden)
    inner = net.relu(_dense(net, weights, f"{layer}intermediate", hidden))
    hidden = _dense_no_norm(net, weights, f"{layer}output", inner, hidden)
    return _dense_no_norm(net, weights, f"{layer}output.bottleneck", hidden, stream)


def head_weight(weights: dict, cfg: dict) -> np.ndarray:
    """The masked-LM output projection as one ``[vocab, hidden]`` Linear weight.

    The reference multiplies the transformed row by
    ``cat([decoder.weight.T, dense.weight], 0)``: the first ``embedding_size``
    inputs go through the decoder (the word embedding table when tied), the
    remaining ``hidden - embedding_size`` through the extra dense matrix that
    is stored as ``[hidden - embedding_size, vocab]``.
    """
    if cfg["tie_word_embeddings"]:
        decoder = weights[f"{_PREFIX}embeddings.word_embeddings.weight"]
    else:
        decoder = weights["cls.predictions.decoder.weight"]
    decoder = np.asarray(decoder, dtype=np.float32)
    extra = np.asarray(weights["cls.predictions.dense.weight"], dtype=np.float32)
    return np.concatenate((decoder, extra.T), axis=1)


def build_network(network, weights: dict, cfg: dict, seq_len: int) -> _Network:
    """Populate ``network`` with the pretraining graph; return the constant owner."""
    net = _Network(network)
    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    token_type_ids = network.add_input("token_type_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    keep = network.add_cast(attention_mask, trt.float32).get_output(0)
    keep_column = net.reshape(keep, (seq_len, 1))
    # Only the keys are masked, as in the reference: a padded query row still
    # attends to the real keys, and nothing reads that row's result.
    dropped = net.elementwise(net.constant(np.ones((1, 1, 1), np.float32)),
                              net.reshape(keep, (1, 1, seq_len)), trt.ElementWiseOperation.SUB)
    additive_mask = net.mul(dropped, net.constant(np.full((1, 1, 1), _MASK_FLOOR, np.float32)))

    stream = _embeddings(net, weights, cfg, input_ids, token_type_ids, keep_column, seq_len)
    for index in range(cfg["num_layers"]):
        stream = _layer(net, weights, cfg, index, stream, additive_mask, seq_len)

    transform = "cls.predictions.transform"
    head = net.relu(_dense(net, weights, transform, stream))
    head = net.layer_norm(head, weights[f"{transform}.LayerNorm.weight"],
                          weights[f"{transform}.LayerNorm.bias"], cfg["head_eps"])
    logits = net.linear(head, head_weight(weights, cfg), weights["cls.predictions.bias"])
    logits.name = "prediction_logits"
    network.mark_output(logits)

    # The pooled row is the first row as it is (classifier_activation false).
    pooled = net.rows(stream, 0, 1)
    relation = net.linear(pooled, weights["cls.seq_relationship.weight"],
                          weights["cls.seq_relationship.bias"])
    relation = net.reshape(relation, (2,))
    relation.name = "seq_relationship_logits"
    network.mark_output(relation)
    return net
