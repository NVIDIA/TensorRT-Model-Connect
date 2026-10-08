# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Laya's ModernBERT encoder and typed decision head as native TensorRT math."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt


class Graph:
    def __init__(self, network, weights):
        self.n, self.w = network, weights
        self.keep = []

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.n.add_cast(x, dtype).get_output(0)

    def const(self, value, dtype=trt.float32):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.array(value, dtype=np.int32 if dtype == trt.int32 else np.float32, copy=True)
        if dtype == trt.bfloat16:
            import ml_dtypes

            array = array.astype(ml_dtypes.bfloat16).view(np.uint16)
            weight = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        elif dtype == trt.float16:
            array = array.astype(np.float16)
            weight = trt.Weights(array)
        else:
            weight = trt.Weights(array)
        self.keep.extend((array, weight))
        return self.n.add_constant(array.shape, weight).get_output(0)

    def scalar(self, value, x):
        return self.const(np.full((1,) * len(x.shape), value), x.dtype)

    def op(self, a, b, operation):
        if not isinstance(b, trt.ITensor):
            b = self.scalar(b, a)
        return self.n.add_elementwise(a, b, operation).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(self, x, shape):
        layer = self.n.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(self, x, order):
        layer = self.n.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    def concat(self, values, axis=-1):
        layer = self.n.add_concatenation(values)
        layer.axis = axis % len(values[0].shape)
        return layer.get_output(0)

    def slice(self, x, axis, start, size):
        rank = len(x.shape)
        axis %= rank
        starts, sizes = [0] * rank, list(x.shape)
        starts[axis], sizes[axis] = start, size
        layer = self.n.add_slice(x, starts, [max(1, v) for v in sizes], [1] * rank)
        if -1 in sizes:
            shape = self.cast(self.n.add_shape(x).get_output(0), trt.int32)
            parts = [
                self.const([v], trt.int32)
                if v != -1
                else self.n.add_gather(shape, self.const([i], trt.int32), 0).get_output(0)
                for i, v in enumerate(sizes)
            ]
            layer.set_input(2, self.concat(parts, 0))
        return layer.get_output(0)

    def linear(self, x, name, weight=None, bias=None):
        # The original keeps parameters/residuals in FP32 and autocasts Linear
        # to BF16. Checkpoint storage is FP16, so do not round norm/embedding
        # parameters to BF16 together with matrix weights.
        x = self.cast(x, trt.bfloat16)
        weight = self.w[name + ".weight"] if weight is None else weight
        bias = self.w.get(name + ".bias") if bias is None else bias
        rank = len(x.shape)
        output_shape = None
        if rank > 2:
            shape = self.cast(self.n.add_shape(x).get_output(0), trt.int32)
            leading = self.n.add_slice(shape, (0,), (rank - 1,), (1,)).get_output(0)
            output_shape = self.concat([leading, self.const([weight.shape[0]], trt.int32)], 0)
            x = self.reshape(x, (-1, x.shape[-1]))
        w = self.const(weight, trt.bfloat16)
        y = self.n.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if bias is not None:
            y = self.add(y, self.reshape(self.const(bias, trt.bfloat16), (1, len(bias))))
        if output_shape is not None:
            layer = self.n.add_shuffle(y)
            layer.set_input(1, output_shape)
            y = layer.get_output(0)
        return y

    def norm(self, x, name, eps):
        x = self.cast(x, trt.float32)
        shape = (1,) * (len(x.shape) - 1) + (x.shape[-1],)
        w = self.const(self.w[name + ".weight"].reshape(shape))
        bias = self.w.get(name + ".bias")
        b = self.const(np.zeros(shape, np.float32) if bias is None else bias.reshape(shape))
        layer = self.n.add_normalization(x, w, b, 1 << (len(x.shape) - 1))
        layer.epsilon = eps
        return layer.get_output(0)

    def activation(self, x, kind):
        return self.cast(
            self.n.add_activation(self.cast(x, trt.float32), kind).get_output(0), x.dtype
        )

    def reduce(self, x, kind, axis=-1):
        return self.n.add_reduce(x, kind, 1 << (axis % len(x.shape)), True).get_output(0)

    def attention(self, q, k, v, heads, mask, valid_queries):
        width, dim = q.shape[-1], q.shape[-1] // heads

        def layout(x):
            return self.transpose(self.reshape(x, (0, 0, heads, dim)), (0, 2, 1, 3))

        if not self.precise_attention:
            layer = self.n.add_attention(
                self.mul(layout(q), dim**-0.5),
                layout(k),
                layout(v),
                trt.AttentionNormalizationOp.SOFTMAX,
                False,
            )
            # TensorRT requires this bias to share the Q/K/V type. Zero and
            # infinity cast exactly, unlike a finite BF16 minimum sentinel.
            layer.mask = self.cast(mask, q.dtype)
            layer.decomposable = True
            result = self.reshape(self.transpose(layer.get_output(0), (0, 2, 1, 3)), (0, 0, width))
            return self.n.add_select(valid_queries, result, self.scalar(0, result)).get_output(0)

        # Preserve FP32 scores, normalization and value accumulation. Rounding
        # inside BF16 IAttention caused mmBERT probability drift on long conversations.
        q, k, v = [self.cast(tensor, trt.float32) for tensor in (q, k, v)]
        scores = self.n.add_matrix_multiply(
            layout(q), trt.MatrixOperation.NONE, layout(k), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.add(self.mul(scores, dim**-0.5), self.cast(mask, trt.float32))
        probabilities = self.n.add_softmax(scores)
        probabilities.axes = 8
        attended = self.n.add_matrix_multiply(
            probabilities.get_output(0),
            trt.MatrixOperation.NONE,
            layout(v),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        result = self.cast(
            self.reshape(self.transpose(attended, (0, 2, 1, 3)), (0, 0, width)), trt.bfloat16
        )
        # Only valid token rows participate in later attention, marker scoring,
        # or CLS pooling. Keep ignored padding queries finite, including rows
        # whose entire sliding window falls outside the valid key prefix.
        return self.n.add_select(valid_queries, result, self.scalar(0, result)).get_output(0)


def decision_graph(network, weights, config, max_length, debug=False, precise_attention=False):
    import torch

    g = Graph(network, weights)
    g.precise_attention = precise_attention

    def trace(name, tensor):
        if debug:
            output = network.add_identity(tensor).get_output(0)
            output.name = name
            network.mark_output(output)

    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    dim, eps = hidden // heads, config["layer_norm_eps"]
    ids = network.add_input("input_ids", trt.int32, (-1, -1))
    valid = network.add_input("attention_mask", trt.int32, (-1, -1))
    positions = network.add_input("position_ids", trt.int32, (-1,))
    marker_pos = network.add_input("marker_pos", trt.int32, (-1, -1))
    marker_mask = network.add_input("marker_mask", trt.int32, (-1, -1))
    types = network.add_input("qtype", trt.int32, (-1,))
    is_valid = g.op(valid, 0, trt.ElementWiseOperation.GREATER)
    valid_keys = g.reshape(is_valid, (0, 1, 1, -1))
    valid_queries = g.reshape(is_valid, (0, -1, 1))
    distances = g.op(
        g.reshape(positions, (1, 1, -1, 1)),
        g.reshape(positions, (1, 1, 1, -1)),
        trt.ElementWiseOperation.SUB,
    )
    distances = network.add_unary(distances, trt.UnaryOperation.ABS).get_output(0)
    outside = g.op(distances, config["local_attention"] // 2, trt.ElementWiseOperation.GREATER)
    inside = network.add_unary(outside, trt.UnaryOperation.NOT).get_output(0)
    local_keys = g.op(valid_keys, inside, trt.ElementWiseOperation.AND)

    def mask(allowed):
        # SDPA's boolean mask is equivalent to an FP32 -infinity bias, and
        # the original TransformerEncoder head supplies that FP32 bias directly.
        return network.add_select(
            allowed,
            g.const([[[[0.0]]]]),
            g.const([[[[-np.inf]]]]),
        ).get_output(0)

    full_keys = g.op(
        valid_keys,
        g.op(distances, -1, trt.ElementWiseOperation.GREATER),
        trt.ElementWiseOperation.AND,
    )
    masks = {"full_attention": mask(full_keys), "sliding_attention": mask(local_keys)}
    head_mask = network.add_select(
        valid_keys,
        g.const(np.zeros((1, hidden // 64, 1, 1))),
        g.const(np.full((1, hidden // 64, 1, 1), -np.inf)),
    ).get_output(0)
    ropes = {}
    for kind in masks:
        theta = config["rope_parameters"][kind]["rope_theta"]
        inverse = 1.0 / (float(theta) ** (torch.arange(0, dim, 2).float() / dim))
        # Generate constants with the original FP32 CUDA rotary operations.
        angles = torch.arange(max_length, device="cuda").float()[:, None] * inverse.cuda()[None, :]
        angles = torch.cat([angles, angles], -1)
        ropes[kind] = tuple(
            g.reshape(
                network.add_gather(g.const(value.cpu()), positions, 0).get_output(0),
                (1, 1, -1, dim),
            )
            for value in (angles.cos(), angles.sin())
        )

    emb = g.const(weights["encoder.embeddings.tok_embeddings.weight"], trt.float16)
    h = network.add_gather(emb, ids, 0).get_output(0)
    h = g.norm(h, "encoder.embeddings.norm", eps)
    trace("embedding", h)

    def rope(x, cos, sin):
        x = g.transpose(g.reshape(x, (0, 0, heads, dim)), (0, 2, 1, 3))
        f = g.cast(x, trt.float32)
        rotated = g.concat(
            [g.mul(g.slice(f, -1, dim // 2, dim // 2), -1), g.slice(f, -1, 0, dim // 2)]
        )
        y = g.add(g.mul(f, g.cast(cos, trt.float32)), g.mul(rotated, g.cast(sin, trt.float32)))
        return g.reshape(g.transpose(g.cast(y, trt.bfloat16), (0, 2, 1, 3)), (0, 0, hidden))

    for index, kind in enumerate(config["layer_types"]):
        p = f"encoder.layers.{index}"
        n = h if index == 0 else g.norm(h, p + ".attn_norm", eps)
        qkv = g.linear(n, p + ".attn.Wqkv")
        if index == 0:
            trace("qkv", g.cast(qkv, trt.float32))
        q, k, v = [g.slice(qkv, -1, j * hidden, hidden) for j in range(3)]
        q, k = rope(q, *ropes[kind]), rope(k, *ropes[kind])
        attention = g.attention(q, k, v, heads, masks[kind], valid_queries)
        if index == 0:
            trace("q_rotated", g.cast(q, trt.float32))
            trace("k_rotated", g.cast(k, trt.float32))
            trace("attention", g.cast(attention, trt.float32))
        branch = g.linear(attention, p + ".attn.Wo")
        if index == 0:
            trace("attn_output", g.cast(branch, trt.float32))
        h = g.add(h, g.cast(branch, trt.float32))
        n = g.norm(h, p + ".mlp_norm", eps)
        if index == 0:
            trace("mlp_norm", n)
        up = g.linear(n, p + ".mlp.Wi")
        if index == 0:
            trace("mlp_up", g.cast(up, trt.float32))
        inner = config["intermediate_size"]
        gated = g.mul(
            g.activation(g.slice(up, -1, 0, inner), trt.ActivationType.GELU_ERF),
            g.slice(up, -1, inner, inner),
        )
        if index == 0:
            trace("mlp_gated", g.cast(gated, trt.float32))
        h = g.add(h, g.cast(g.linear(gated, p + ".mlp.Wo"), trt.float32))
        trace(f"encoder_{index}", h)
    h = g.norm(h, "encoder.final_norm", eps)
    trace("encoder_norm", h)
    type_emb = network.add_gather(g.const(weights["type_emb.weight"]), types, 0).get_output(0)
    h = g.add(h, g.reshape(type_emb, (0, 1, hidden)))

    index = 0
    while f"head.layers.{index}.norm1.weight" in weights:
        p = f"head.layers.{index}"
        n = g.norm(h, p + ".norm1", 1e-5)
        w, b = weights[p + ".self_attn.in_proj_weight"], weights[p + ".self_attn.in_proj_bias"]
        qkv = g.linear(n, "", w, b)
        q, k, v = [g.slice(qkv, -1, index * hidden, hidden) for index in range(3)]
        attention = g.attention(q, k, v, hidden // 64, head_mask, valid_queries)
        h = g.add(h, g.cast(g.linear(attention, p + ".self_attn.out_proj"), trt.float32))
        n = g.norm(h, p + ".norm2", 1e-5)
        branch = g.linear(
            g.activation(g.linear(n, p + ".linear1"), trt.ActivationType.RELU), p + ".linear2"
        )
        h = g.add(h, g.cast(branch, trt.float32))
        trace(f"head_{index}", h)
        index += 1

    indices = g.op(marker_pos, 0, trt.ElementWiseOperation.MAX)
    indices = g.add(g.reshape(indices, (0, 0, 1)), g.const(np.zeros((1, 1, hidden)), trt.int32))
    gather = network.add_gather_v2(h, indices, trt.GatherMode.ELEMENT)
    gather.axis = 1
    markers = gather.get_output(0)
    features = g.activation(
        g.linear(g.norm(markers, "scorer.0", 1e-5), "scorer.1"), trt.ActivationType.GELU_ERF
    )
    # Keep the final scalar accumulation in FP32 while preserving the original
    # BF16 operands. Calibrated probabilities are sensitive to this projection.
    score_weight = g.cast(g.const(weights["scorer.3.weight"], trt.bfloat16), trt.float32)
    score_bias = g.reshape(
        g.cast(g.const(weights["scorer.3.bias"], trt.bfloat16), trt.float32), (1, 1)
    )
    logits = network.add_matrix_multiply(
        g.reshape(g.cast(features, trt.float32), (-1, hidden)),
        trt.MatrixOperation.NONE,
        score_weight,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    logits = g.add(logits, score_bias)
    shape = network.add_shuffle(logits)
    shape.set_input(1, g.cast(network.add_shape(marker_pos).get_output(0), trt.int32))
    logits = shape.get_output(0)
    mark_valid = g.op(marker_mask, 0, trt.ElementWiseOperation.GREATER)
    logits = network.add_select(mark_valid, logits, g.scalar(-1e4, logits)).get_output(0)
    sm = network.add_softmax(logits)
    sm.axes = 2
    probabilities = sm.get_output(0)
    k = g.reduce(g.cast(marker_mask, trt.float32), trt.ReduceOperation.SUM)
    k = g.op(k, 2.0, trt.ElementWiseOperation.MAX)
    logp = network.add_unary(
        g.op(probabilities, 1e-9, trt.ElementWiseOperation.MAX), trt.UnaryOperation.LOG
    ).get_output(0)
    entropy = g.mul(g.reduce(g.mul(probabilities, logp), trt.ReduceOperation.SUM), -1)
    entropy = g.op(
        entropy,
        network.add_unary(k, trt.UnaryOperation.LOG).get_output(0),
        trt.ElementWiseOperation.DIV,
    )
    padded = g.concat([probabilities, g.mul(k, 0)])
    top = network.add_topk(padded, trt.TopKOperation.MAX, 2, 2).get_output(0)
    first, second = g.slice(top, 1, 0, 1), g.slice(top, 1, 1, 1)
    features = g.concat(
        [
            first,
            g.op(first, second, trt.ElementWiseOperation.SUB),
            entropy,
            g.op(k, 255.0, trt.ElementWiseOperation.DIV),
        ]
    )
    pooled = g.reshape(g.slice(h, 1, 0, 1), (0, hidden))
    act = g.linear(
        g.activation(
            g.linear(g.concat([pooled, features]), "act_head.0"), trt.ActivationType.GELU_ERF
        ),
        "act_head.2",
    )
    for name, tensor in (("logits", logits), ("act_logits", g.cast(act, trt.float32))):
        tensor.name = name
        network.mark_output(tensor)
    return g


def build_plan(root, *, max_sequence_length, max_batch_size=8, max_options=256, debug=False):
    from safetensors.torch import load_file

    root = Path(root)
    config = json.loads((root / "encoder/config.json").read_text())
    agent = json.loads((root / "rl_agent_config.json").read_text())
    if config["model_type"] != "modernbert":
        raise ValueError("Laya requires a ModernBERT encoder")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    _graph = decision_graph(
        network,
        load_file(root / "model.safetensors"),
        config,
        max_sequence_length,
        debug,
        precise_attention=agent["encoder"] == "jhu-clsp/mmBERT-base",
    )
    profile = builder.create_optimization_profile()
    opt_batch, opt_length, opt_options = (
        min(4, max_batch_size),
        min(128, max_sequence_length),
        min(4, max_options),
    )
    for name in ("input_ids", "attention_mask"):
        profile.set_shape(
            name, (1, 1), (opt_batch, opt_length), (max_batch_size, max_sequence_length)
        )
    for name in ("marker_pos", "marker_mask"):
        profile.set_shape(name, (1, 1), (opt_batch, opt_options), (max_batch_size, max_options))
    profile.set_shape("qtype", (1,), (opt_batch,), (max_batch_size,))
    profile.set_shape("position_ids", (1,), (opt_length,), (max_sequence_length,))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.clear_flag(trt.BuilderFlag.TF32)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    settings.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Laya TensorRT build failed")
    return bytes(plan)


def build_variant(request, writer, prefix=""):
    """Write one selected checkpoint and its model-owned runtime data."""
    from .cli import BuildRequest, VARIANTS
    from laya.agent import _load_tokenizer
    from .tokenizer import tokenizer_data

    if not isinstance(request, BuildRequest):
        raise TypeError("Laya requires its family-owned BuildRequest")
    root = Path(request.model_dir) / VARIANTS[request.variant]
    config = json.loads((root / "rl_agent_config.json").read_text())
    encoder = json.loads((root / "encoder/config.json").read_text())
    maximum = request.max_sequence_length or config["max_len"]
    if maximum > encoder["max_position_embeddings"]:
        raise ValueError("sequence capacity exceeds the checkpoint's position capacity")
    tokenizer = _load_tokenizer(str(root / "tokenizer"), config)
    config.update(
        precision="bf16",
        max_len=maximum,
        max_batch_size=request.max_batch_size,
        max_options=request.max_options,
        variant=request.variant,
        num_actions=len(config.get("act_costs", {})) + 1,
    )
    for token in ("cls", "sep", "pad", "mask"):
        value = getattr(tokenizer, token + "_token_id")
        if value is None:
            raise ValueError(f"Laya tokenizer is missing {token}")
        config[token + "_token_id"] = value
    writer.add_json(prefix + "runtime.json", config)
    writer.add_bytes(prefix + "tokenizer.json", tokenizer_data(root / "tokenizer/tokenizer.json"))
    writer.add_bytes(
        prefix + "model.plan",
        build_plan(
            root,
            max_sequence_length=maximum,
            max_batch_size=request.max_batch_size,
            max_options=request.max_options,
        ),
    )


def build(request, writer):
    from dataclasses import replace
    from .cli import BuildRequest, VARIANTS

    if not isinstance(request, BuildRequest):
        raise TypeError("Laya requires its family-owned BuildRequest")
    writer.set_header(family="laya", task=request.task, backend=request.backend)
    if request.variant == "router":
        from .routing import routing_data

        writer.add_json("router.json", routing_data())
        for variant in VARIANTS:
            build_variant(replace(request, variant=variant), writer, variant + "/")
    else:
        build_variant(request, writer)
