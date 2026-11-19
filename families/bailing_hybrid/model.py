# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental Ling3 hybrid KDA/MLA decoder and group-limited routed experts."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorrt as trt


class Weights:
    def __init__(self, directory, stack):
        from safetensors import safe_open

        self.index = json.loads((directory / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.files = {
            name: stack.enter_context(
                safe_open(str(directory / name), framework="pt", device="cpu")
            )
            for name in set(self.index.values())
        }

    def read(self, name):
        return self.files[self.index[name]].get_tensor(name).float().numpy()


class Graph:
    def __init__(self, weights, cfg, logger):
        self.weights, self.cfg, self.logger = weights, cfg, logger
        self.builder = trt.Builder(logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )

    def const(self, value, dtype=np.float32):
        value = np.ascontiguousarray(value, dtype=dtype)
        return self.net.add_constant(value.shape, value).get_output(0)

    def op(self, a, b, kind):
        return self.net.add_elementwise(a, b, kind).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(self, value, shape):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def concat(self, tensors, axis):
        layer = self.net.add_concatenation(tensors)
        layer.axis = axis
        return layer.get_output(0)

    def reduce(self, x, kind, axes, keep=True):
        return self.net.add_reduce(x, kind, axes, keep).get_output(0)

    def unary(self, x, kind):
        return self.net.add_unary(x, kind).get_output(0)

    def sigmoid(self, x):
        return self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)

    def silu(self, x):
        return self.mul(x, self.sigmoid(x))

    def linear(self, x, name):
        return self.net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            self.const(self.weights.read(name + ".weight")),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)

    def norm(self, x, name):
        axis = len(x.shape) - 1
        average = self.reduce(self.mul(x, x), trt.ReduceOperation.AVG, 1 << axis)
        root = self.unary(
            self.add(average, self.const(np.full((1,) * len(x.shape), self.cfg["rms_norm_eps"]))),
            trt.UnaryOperation.SQRT,
        )
        normalized = self.op(x, root, trt.ElementWiseOperation.DIV)
        gamma = self.weights.read(name + ".weight").reshape((1,) * axis + (int(x.shape[-1]),))
        return self.mul(normalized, self.const(gamma))

    def dense(self, x, name):
        return self.linear(
            self.mul(
                self.silu(self.linear(x, name + ".gate_proj")), self.linear(x, name + ".up_proj")
            ),
            name + ".down_proj",
        )

    def output(self, value, name):
        value.name = name
        self.net.mark_output(value)

    def save(self, stream):
        options = self.builder.create_builder_config()
        options.builder_optimization_level = 3
        options.clear_flag(trt.BuilderFlag.TF32)
        # Keep serialization bounded to the current engine. The caller releases
        # this graph before loading the next layer's weights.
        plan = self.builder.build_serialized_network(self.net, options)
        if plan is None:
            raise RuntimeError("Ling3 engine build failed")
        stream.write(memoryview(plan))

    def kda(self, hidden, prefix):
        net, cfg = self.net, self.cfg
        heads, dim, kernel = (
            cfg["num_attention_heads"],
            cfg["head_dim"],
            cfg["short_conv_kernel_size"],
        )
        channels = heads * dim
        vectors = []
        for kind in ("q", "k", "v"):
            projected = self.reshape(self.linear(hidden, prefix + f".{kind}_proj"), (channels, 1))
            history = net.add_input(f"conv_{kind}", trt.float32, (channels, kernel - 1))
            taps = self.concat([history, projected], 1)
            present = net.add_slice(taps, (0, 1), (channels, kernel - 1), (1, 1)).get_output(0)
            self.output(present, f"present_conv_{kind}")
            weight = self.weights.read(prefix + f".{kind}_conv1d.weight").reshape(channels, kernel)
            filtered = self.reduce(
                self.mul(taps, self.const(weight)), trt.ReduceOperation.SUM, 2, False
            )
            vectors.append(self.reshape(self.silu(filtered), (heads, dim)))
        query, key, value = vectors

        def l2norm(x):
            squared = self.reduce(self.mul(x, x), trt.ReduceOperation.SUM, 2)
            root = self.unary(self.add(squared, self.const([[1e-6]])), trt.UnaryOperation.SQRT)
            return self.op(x, root, trt.ElementWiseOperation.DIV)

        query = self.mul(l2norm(query), self.const([[dim**-0.5]]))
        key = l2norm(key)
        gate = self.reshape(self.linear(hidden, prefix + ".f_proj"), (heads, dim))
        gate = self.add(
            gate, self.const(self.weights.read(prefix + ".dt_bias").reshape(heads, dim))
        )
        gate = self.mul(
            gate, self.const(np.exp(self.weights.read(prefix + ".A_log")).reshape(heads, 1))
        )
        decay = self.unary(
            self.mul(self.sigmoid(gate), self.const([[cfg["kda_lower_bound"]]])),
            trt.UnaryOperation.EXP,
        )
        state = net.add_input("state", trt.float32, (heads, dim, dim))
        state = self.mul(state, self.reshape(decay, (heads, dim, 1)))
        predicted = self.reduce(
            self.mul(state, self.reshape(key, (heads, dim, 1))), trt.ReduceOperation.SUM, 2, False
        )
        beta = self.reshape(self.sigmoid(self.linear(hidden, prefix + ".b_proj")), (heads, 1))
        innovation = self.mul(self.op(value, predicted, trt.ElementWiseOperation.SUB), beta)
        state = self.add(
            state,
            self.mul(self.reshape(key, (heads, dim, 1)), self.reshape(innovation, (heads, 1, dim))),
        )
        self.output(state, "present_state")
        attended = self.reduce(
            self.mul(state, self.reshape(query, (heads, dim, 1))), trt.ReduceOperation.SUM, 2, False
        )
        attended = self.norm(attended, prefix + ".o_norm")
        gate = self.reshape(self.sigmoid(self.linear(hidden, prefix + ".g_proj")), (heads, dim))
        return self.linear(
            self.reshape(self.mul(attended, gate), (1, channels)), prefix + ".o_proj"
        )

    def mla(self, hidden, prefix, length):
        import torch

        net, cfg = self.net, self.cfg
        heads, dim, nope, rotary, value_dim = (
            cfg[k]
            for k in (
                "num_attention_heads",
                "qk_head_dim",
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "v_head_dim",
            )
        )
        position = net.add_input("position", trt.int32, (1,))
        query = self.linear(
            self.norm(self.linear(hidden, prefix + ".q_a_proj"), prefix + ".q_a_layernorm"),
            prefix + ".q_b_proj",
        )
        query = self.reshape(query, (1, heads, 1, dim))
        q_pass = net.add_slice(query, (0, 0, 0, 0), (1, heads, 1, nope), (1, 1, 1, 1)).get_output(0)
        q_rot = net.add_slice(
            query, (0, 0, 0, nope), (1, heads, 1, rotary), (1, 1, 1, 1)
        ).get_output(0)
        compressed = self.linear(hidden, prefix + ".kv_a_proj_with_mqa")
        rank = cfg["kv_lora_rank"]
        latent = net.add_slice(compressed, (0, 0), (1, rank), (1, 1)).get_output(0)
        k_rot = self.reshape(
            net.add_slice(compressed, (0, rank), (1, rotary), (1, 1)).get_output(0),
            (1, 1, 1, rotary),
        )
        expanded = self.reshape(
            self.linear(self.norm(latent, prefix + ".kv_a_layernorm"), prefix + ".kv_b_proj"),
            (1, heads, 1, nope + value_dim),
        )
        k_pass = net.add_slice(
            expanded, (0, 0, 0, 0), (1, heads, 1, nope), (1, 1, 1, 1)
        ).get_output(0)
        value = net.add_slice(
            expanded, (0, 0, 0, nope), (1, heads, 1, value_dim), (1, 1, 1, 1)
        ).get_output(0)
        with torch.no_grad():
            inverse = 1 / (cfg["rope_theta"] ** (torch.arange(0, rotary, 2).float() / rotary))
            frequencies = torch.arange(length).float()[:, None] * inverse[None]
            frequencies = torch.cat([frequencies, frequencies], -1)
            cosine, sine = [
                self.reshape(
                    net.add_gather(self.const(t.numpy()), position, 0).get_output(0),
                    (1, 1, 1, rotary),
                )
                for t in (frequencies.cos(), frequencies.sin())
            ]

        def rope(x, count):
            grouped = self.reshape(x, (1, count, 1, rotary // 2, 2))
            shuffled = net.add_shuffle(grouped)
            shuffled.first_transpose = (0, 1, 2, 4, 3)
            shuffled.reshape_dims = (1, count, 1, rotary)
            x = shuffled.get_output(0)
            first = net.add_slice(
                x, (0, 0, 0, 0), (1, count, 1, rotary // 2), (1, 1, 1, 1)
            ).get_output(0)
            second = net.add_slice(
                x, (0, 0, 0, rotary // 2), (1, count, 1, rotary // 2), (1, 1, 1, 1)
            ).get_output(0)
            return self.add(
                self.mul(x, cosine),
                self.mul(self.concat([self.unary(second, trt.UnaryOperation.NEG), first], 3), sine),
            )

        query = self.concat([q_pass, rope(q_rot, heads)], 3)
        key = self.concat(
            [k_pass, self.add(rope(k_rot, 1), self.const(np.zeros((1, heads, 1, rotary))))], 3
        )
        old_k = net.add_input("cache_k", trt.float32, (1, heads, length, dim))
        old_v = net.add_input("cache_v", trt.float32, (1, heads, length, value_dim))
        indices = self.const(np.arange(length), np.int32)
        write = self.reshape(
            self.op(indices, position, trt.ElementWiseOperation.EQUAL), (1, 1, length, 1)
        )
        key, value = [
            net.add_select(write, new, old).get_output(0)
            for new, old in ((key, old_k), (value, old_v))
        ]
        self.output(key, "present_k")
        self.output(value, "present_v")
        scores = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = self.mul(scores, self.const(np.full((1, 1, 1, 1), dim**-0.5)))
        future = self.reshape(
            self.op(indices, position, trt.ElementWiseOperation.GREATER), (1, 1, 1, length)
        )
        scores = net.add_select(future, self.const(np.full((1, 1, 1, 1), -1e9)), scores).get_output(
            0
        )
        probability = net.add_softmax(scores)
        probability.axes = 8
        attended = net.add_matrix_multiply(
            probability.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        gate = self.reshape(self.sigmoid(self.linear(hidden, prefix + ".g_proj")), (1, heads, 1, 1))
        return self.linear(
            self.reshape(self.mul(attended, gate), (1, heads * value_dim)), prefix + ".dense"
        )

    def moe(self, hidden, prefix):
        net, cfg = self.net, self.cfg
        experts, top, groups, selected_groups, middle, width = (
            cfg[k]
            for k in (
                "num_experts",
                "num_experts_per_tok",
                "n_group",
                "topk_group",
                "moe_intermediate_size",
                "hidden_size",
            )
        )
        scores = self.sigmoid(self.linear(hidden, prefix + ".gate"))
        corrected = self.add(
            scores, self.const(self.weights.read(prefix + ".gate.expert_bias").reshape(1, experts))
        )
        grouped = self.reshape(corrected, (1, groups, experts // groups))
        group_top = net.add_topk(grouped, trt.TopKOperation.MAX, 2, 4).get_output(0)
        group_scores = self.reduce(group_top, trt.ReduceOperation.SUM, 4, False)
        group_ids = self.reshape(
            net.add_topk(group_scores, trt.TopKOperation.MAX, selected_groups, 2).get_output(1),
            (selected_groups,),
        )
        selected = net.add_gather(self.const(np.eye(groups)), group_ids, 0).get_output(0)
        selected = self.reshape(
            self.reduce(selected, trt.ReduceOperation.SUM, 1, False), (1, groups, 1)
        )
        keep = self.op(selected, self.const([[[0.5]]]), trt.ElementWiseOperation.GREATER)
        filtered = net.add_select(keep, grouped, self.const(np.full((1, 1, 1), -1e9))).get_output(0)
        ids = self.reshape(
            net.add_topk(
                self.reshape(filtered, (1, experts)), trt.TopKOperation.MAX, top, 2
            ).get_output(1),
            (top,),
        )
        weights = net.add_gather(scores, ids, 1).get_output(0)
        total = self.add(self.reduce(weights, trt.ReduceOperation.SUM, 2), self.const([[1e-20]]))
        weights = self.reshape(
            self.mul(
                self.op(weights, total, trt.ElementWiseOperation.DIV),
                self.const([[cfg["routed_scaling_factor"]]]),
            ),
            (top, 1, 1),
        )
        up = np.empty((experts, 2 * middle, width), np.float32)
        down = np.empty((experts, width, middle), np.float32)
        for expert in range(experts):
            name = prefix + f".experts.{expert}"
            up[expert, :middle] = self.weights.read(name + ".gate_proj.weight")
            up[expert, middle:] = self.weights.read(name + ".up_proj.weight")
            down[expert] = self.weights.read(name + ".down_proj.weight")
        up = net.add_gather(self.const(up), ids, 0).get_output(0)
        down = net.add_gather(self.const(down), ids, 0).get_output(0)
        projected = net.add_matrix_multiply(
            self.reshape(hidden, (1, 1, width)),
            trt.MatrixOperation.NONE,
            up,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        gate = net.add_slice(projected, (0, 0, 0), (top, 1, middle), (1, 1, 1)).get_output(0)
        value = net.add_slice(projected, (0, 0, middle), (top, 1, middle), (1, 1, 1)).get_output(0)
        result = net.add_matrix_multiply(
            self.mul(self.silu(gate), value),
            trt.MatrixOperation.NONE,
            down,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        result = self.reduce(self.mul(result, weights), trt.ReduceOperation.SUM, 1, False)
        return self.add(result, self.dense(hidden, prefix + ".shared_experts"))


def build(request, writer):
    if request.task != "text_continuation" or request.backend != "trt":
        raise ValueError("Ling3 requires text_continuation with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("bailing_hybrid does not support dynamic_kv_cache")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError(
            "Ling3 supports one FP32 sequence with family-owned KDA and MLA state"
        )
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    if (
        cfg["model_type"] != "bailing_hybrid"
        or cfg["hidden_act"] != "silu"
        or cfg["use_qkv_bias"]
        or cfg["use_bias"]
        or not cfg["no_kda_lora"]
        or not cfg["kda_safe_gate"]
        or not cfg["rope_interleave"]
        or cfg["rope_scaling"]
        or cfg["num_shared_experts"] != 1
        or cfg["gated_attention_proj_granularity_type"] != "head_wise"
        or cfg["num_nextn_predict_layers"]
        or cfg["first_k_dense_replace"] != 1
        or not cfg["norm_topk_prob"]
        or cfg["group_norm_size"] != 1
        or cfg["layer_group_size"] < 1
        or cfg["num_hidden_layers"] % cfg["layer_group_size"]
    ):
        raise ValueError("Unsupported Ling3 hybrid attention or routing configuration")
    length = request.max_sequence_length or 128
    if not 2 <= length <= min(512, cfg["max_position_embeddings"]):
        raise ValueError("Ling3 fixed KV profile must be in [2, 512]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.set_header(family="bailing_hybrid", task=request.task, backend=request.backend)
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    kinds = [
        "mla" if (i + 1) % cfg["layer_group_size"] == 0 else "kda"
        for i in range(cfg["num_hidden_layers"])
    ]
    with ExitStack() as stack:
        weights = Weights(directory, stack)
        graph = Graph(weights, cfg, logger)
        token = graph.net.add_input("input_ids", trt.int32, (1,))
        graph.output(
            graph.net.add_gather(
                graph.const(weights.read("model.word_embeddings.weight")), token, 0
            ).get_output(0),
            "result",
        )
        with writer.open_section("embedding.plan") as section:
            graph.save(section)
        del graph
        for layer, kind in enumerate(kinds):
            graph = Graph(weights, cfg, logger)
            prefix = f"model.layers.{layer}"
            hidden = graph.net.add_input("hidden", trt.float32, (1, cfg["hidden_size"]))
            normal = graph.norm(hidden, prefix + ".input_layernorm")
            attention = (
                graph.mla(normal, prefix + ".attention", length)
                if kind == "mla"
                else graph.kda(normal, prefix + ".attention")
            )
            hidden = graph.add(hidden, attention)
            normal = graph.norm(hidden, prefix + ".post_attention_layernorm")
            feed_forward = (
                graph.dense(normal, prefix + ".mlp")
                if layer == 0
                else graph.moe(normal, prefix + ".mlp")
            )
            graph.output(graph.add(hidden, feed_forward), "result")
            with writer.open_section(f"layer.{layer}.plan") as section:
                graph.save(section)
            del graph
        graph = Graph(weights, cfg, logger)
        hidden = graph.net.add_input("hidden", trt.float32, (1, cfg["hidden_size"]))
        graph.output(graph.linear(graph.norm(hidden, "model.norm"), "lm_head"), "logits")
        with writer.open_section("head.plan") as section:
            graph.save(section)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            layers=cfg["num_hidden_layers"],
            layer_types=kinds,
            hidden_size=cfg["hidden_size"],
            heads=cfg["num_attention_heads"],
            head_dim=cfg["head_dim"],
            key_dim=cfg["qk_head_dim"],
            value_dim=cfg["v_head_dim"],
            conv_kernel=cfg["short_conv_kernel_size"],
            vocab_size=cfg["vocab_size"],
            eos_ids=[cfg["eos_token_id"]],
        ),
    )
