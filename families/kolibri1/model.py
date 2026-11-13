# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP16 Kolibri1 from block-scaled FP8 weights, with bounded build memory."""

import gc
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorrt as trt


class Weights:
    """Map source shards and dequantize only the current engine's constants."""

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

    def tensor(self, name):
        return self.files[self.index[name]].get_tensor(name)

    def read(self, name, dtype=np.float16):
        import torch

        value = self.tensor(name).float()
        scale_name = name + "_scale_inv"
        if scale_name in self.index:
            rows, columns = value.shape
            if rows % 128 or columns % 128:
                raise ValueError("Kolibri1 FP8 matrices must have complete 128x128 blocks")
            scales = self.tensor(scale_name).float()
            value = (
                value.reshape(rows // 128, 128, columns // 128, 128) * scales[:, None, :, None]
            ).reshape(rows, columns)
        return value.to(torch.float16 if dtype == np.float16 else torch.float32).numpy()


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)


class Graph:
    def __init__(self, weights, cfg, logger):
        self.weights, self.cfg = weights, cfg
        self.logger = logger
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )

    def const(self, value, dtype=np.float16):
        value = np.ascontiguousarray(value, dtype=dtype)
        return self.net.add_constant(value.shape, value).get_output(0)

    def cast(self, x, dtype):
        return self.net.add_cast(x, dtype).get_output(0)

    def op(self, a, b, kind):
        return self.net.add_elementwise(a, b, kind).get_output(0)

    def add(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def linear(self, x, name, router=False):
        # Router GEMM accumulates into FP32 from FP16-rounded operands.
        weight = self.weights.read(name + ".weight")
        dtype = np.float32 if router else np.float16
        if router:
            x = self.cast(x, trt.float32)
        return self.net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, self.const(weight, dtype), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def norm(self, x, name):
        axis = len(x.shape) - 1
        ones = (1,) * len(x.shape)
        value = self.cast(x, trt.float32)
        average = self.net.add_reduce(
            self.mul(value, value), trt.ReduceOperation.AVG, 1 << axis, True
        ).get_output(0)
        inverse = self.net.add_unary(
            self.add(average, self.const(np.full(ones, self.cfg["rms_norm_eps"]), np.float32)),
            trt.UnaryOperation.SQRT,
        ).get_output(0)
        inverse = self.net.add_unary(inverse, trt.UnaryOperation.RECIP).get_output(0)
        # Match RMSNorm's cast before its FP16 affine multiplication.
        value = self.cast(self.mul(value, inverse), trt.float16)
        gamma = self.weights.read(name + ".weight").reshape((1,) * axis + (int(x.shape[-1]),))
        return self.mul(value, self.const(gamma))

    def silu(self, x):
        value = self.cast(x, trt.float32)
        sigmoid = self.net.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(value, sigmoid), x.dtype)

    def output(self, value, name):
        value.name = name
        self.net.mark_output(value)

    def save(self, stream):
        options = self.builder.create_builder_config()
        options.builder_optimization_level = 3
        options.clear_flag(trt.BuilderFlag.TF32)
        if not self.builder.build_serialized_network_to_stream(
            self.net, options, PlanWriter(stream)
        ):
            raise RuntimeError("Kolibri1 engine build failed")

    def layer(self, index, length):
        net, cfg = self.net, self.cfg
        width, heads, kv, dim = (
            cfg[k]
            for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
        )
        prefix = f"model.layers.{index}"
        hidden = net.add_input("hidden", trt.float16, (1, width))
        position = net.add_input("position", trt.int32, (1,))
        indices = self.const(np.arange(length), np.int32)
        write = self.reshape(
            self.op(indices, position, trt.ElementWiseOperation.EQUAL), (1, 1, length, 1)
        )
        future = self.reshape(
            self.op(indices, position, trt.ElementWiseOperation.GREATER), (1, 1, 1, 1, length)
        )
        normal = self.norm(hidden, prefix + ".input_layernorm")
        query = self.reshape(
            self.norm(
                self.reshape(self.linear(normal, prefix + ".self_attn.q_proj"), (heads, dim)),
                prefix + ".self_attn.q_norm",
            ),
            (1, heads, 1, dim),
        )
        key = self.reshape(
            self.norm(
                self.reshape(self.linear(normal, prefix + ".self_attn.k_proj"), (kv, dim)),
                prefix + ".self_attn.k_norm",
            ),
            (1, kv, 1, dim),
        )
        value = self.reshape(self.linear(normal, prefix + ".self_attn.v_proj"), (1, kv, 1, dim))
        if cfg["layer_types"][index] == "sliding_attention":
            import torch
            from transformers import Kolibri1Config
            from transformers.models.kolibri1.modeling_kolibri1 import Kolibri1RotaryEmbedding

            with torch.no_grad():
                tables = Kolibri1RotaryEmbedding(Kolibri1Config(**cfg))(
                    torch.empty(1, length, width, dtype=torch.float16), torch.arange(length)[None]
                )
            cosine, sine = [
                self.reshape(
                    net.add_gather(self.const(t.numpy()[0]), position, 0).get_output(0),
                    (1, 1, 1, dim),
                )
                for t in tables
            ]

            def rope(x, count):
                first = net.add_slice(
                    x, (0, 0, 0, 0), (1, count, 1, dim // 2), (1, 1, 1, 1)
                ).get_output(0)
                second = net.add_slice(
                    x, (0, 0, 0, dim // 2), (1, count, 1, dim // 2), (1, 1, 1, 1)
                ).get_output(0)
                rotated = net.add_concatenation(
                    [net.add_unary(second, trt.UnaryOperation.NEG).get_output(0), first]
                )
                rotated.axis = 3
                return self.add(self.mul(x, cosine), self.mul(rotated.get_output(0), sine))

            query, key = rope(query, heads), rope(key, kv)
        # Global layers intentionally have no rotary position embedding.
        old_k = net.add_input("cache_k", trt.float16, (1, kv, length, dim))
        old_v = net.add_input("cache_v", trt.float16, (1, kv, length, dim))
        key, value = [
            net.add_select(write, new, old).get_output(0)
            for new, old in ((key, old_k), (value, old_v))
        ]
        self.output(key, "present_k")
        self.output(value, "present_v")
        query = self.reshape(query, (1, kv, heads // kv, 1, dim))
        scores = net.add_matrix_multiply(
            query,
            trt.MatrixOperation.NONE,
            self.reshape(key, (1, kv, 1, length, dim)),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        # PyTorch's scalar operation evaluates at FP32 opmath precision before
        # rounding back to FP16. A half-stored scale changes every head's logits.
        scores = self.cast(
            self.mul(
                self.cast(scores, trt.float32),
                self.const(np.full((1, 1, 1, 1, 1), dim**-0.5), np.float32),
            ),
            trt.float16,
        )
        scores = net.add_select(
            future, self.const(np.full((1, 1, 1, 1, 1), -65504)), scores
        ).get_output(0)
        probability = net.add_softmax(self.cast(scores, trt.float32))
        probability.axes = 16
        attended = net.add_matrix_multiply(
            self.cast(probability.get_output(0), trt.float16),
            trt.MatrixOperation.NONE,
            self.reshape(value, (1, kv, 1, length, dim)),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        attention = self.linear(
            self.reshape(attended, (1, heads * dim)), prefix + ".self_attn.o_proj"
        )
        hidden = self.add(hidden, self.norm(attention, prefix + ".post_attn_norm"))
        normal = self.norm(hidden, prefix + ".post_attention_layernorm")
        routed = self.moe(normal, prefix)
        self.output(self.add(hidden, self.norm(routed, prefix + ".post_ffn_norm")), "result")

    def moe(self, normal, prefix):
        net, cfg = self.net, self.cfg
        top, middle, width, experts = (
            cfg[k]
            for k in ("num_experts_per_tok", "moe_intermediate_size", "hidden_size", "num_experts")
        )
        logits = self.linear(normal, prefix + ".mlp.gate", router=True)
        bias = self.weights.read(prefix + ".moe.router.expert_bias", np.float32).reshape(1, experts)
        selected = net.add_topk(
            self.add(logits, self.const(bias, np.float32)), trt.TopKOperation.MAX, top, 2
        )
        # The official expert loop accumulates in ascending expert-ID order.
        ordered = net.add_topk(
            self.cast(selected.get_output(1), trt.float32), trt.TopKOperation.MIN, top, 2
        )
        ids = self.reshape(self.cast(ordered.get_output(0), trt.int32), (top,))
        scores = net.add_activation(
            net.add_gather(logits, ids, 1).get_output(0), trt.ActivationType.SIGMOID
        ).get_output(0)
        scores = self.reshape(scores, (top, 1, 1))
        up = np.empty((experts, 2 * middle, width), dtype=np.float16)
        down = np.empty((experts, width, middle), dtype=np.float16)
        for expert in range(experts):
            path = prefix + f".mlp.experts.{expert}"
            up[expert, :middle] = self.weights.read(path + ".gate_proj.weight")
            up[expert, middle:] = self.weights.read(path + ".up_proj.weight")
            down[expert] = self.weights.read(path + ".down_proj.weight")
        selected_up = net.add_gather(self.const(up), ids, 0).get_output(0)
        selected_down = net.add_gather(self.const(down), ids, 0).get_output(0)
        projected = net.add_matrix_multiply(
            self.reshape(normal, (1, 1, width)),
            trt.MatrixOperation.NONE,
            selected_up,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        gate = net.add_slice(projected, (0, 0, 0), (top, 1, middle), (1, 1, 1)).get_output(0)
        value = net.add_slice(projected, (0, 0, middle), (top, 1, middle), (1, 1, 1)).get_output(0)
        result = net.add_matrix_multiply(
            self.mul(self.silu(gate), value),
            trt.MatrixOperation.NONE,
            selected_down,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        weighted = self.cast(self.mul(self.cast(result, trt.float32), scores), trt.float16)
        merged = self.const(np.zeros((1, width)))
        for expert in range(top):
            part = net.add_slice(weighted, (expert, 0, 0), (1, 1, width), (1, 1, 1)).get_output(0)
            merged = self.add(merged, self.reshape(part, (1, width)))
        shared = prefix + ".mlp.shared_experts"
        shared_result = self.linear(
            self.mul(
                self.silu(self.linear(normal, shared + ".gate_proj")),
                self.linear(normal, shared + ".up_proj"),
            ),
            shared + ".down_proj",
        )
        return self.add(merged, shared_result)


def build(request, writer):
    if request.task != "text_continuation" or request.backend != "trt":
        raise ValueError("Kolibri1 requires text_continuation with backend=trt")
    if (
        request.precision != "fp16"
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
            "Kolibri1 supports one FP16 sequence dequantized from its FP8 source"
        )
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    quant = cfg.get("quantization_config", {})
    length = request.max_sequence_length or 128
    if (
        cfg["model_type"] != "kolibri1"
        or cfg["attention_bias"]
        or cfg["hidden_act"] != "silu"
        or cfg["norm_topk_prob"]
        or cfg["tie_word_embeddings"]
        or cfg.get("rope_scaling")
        or quant.get("quant_method") != "fp8"
        or quant.get("weight_block_size") != [128, 128]
        or len(cfg["layer_types"]) != cfg["num_hidden_layers"]
        or set(cfg["layer_types"]) - {"full_attention", "sliding_attention"}
        or cfg["num_attention_heads"] % cfg["num_key_value_heads"]
    ):
        raise ValueError(
            "Unsupported Kolibri1 routing, position, or source quantization configuration"
        )
    if not 2 <= length <= min(cfg["sliding_window"], cfg["max_position_embeddings"]):
        raise ValueError("Kolibri1 fixed KV profile must fit within the sliding window")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    generation = json.loads((directory / "generation_config.json").read_text())
    writer.set_header(family="kolibri1", task=request.task, backend=request.backend)
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    with ExitStack() as stack:
        weights = Weights(directory, stack)
        graph = Graph(weights, cfg, logger)
        token = graph.net.add_input("input_ids", trt.int32, (1,))
        graph.output(
            graph.net.add_gather(
                graph.const(weights.read("model.embed_tokens.weight")), token, 0
            ).get_output(0),
            "result",
        )
        with writer.open_section("embedding.plan") as section:
            graph.save(section)
        del graph
        gc.collect()
        for layer in range(cfg["num_hidden_layers"]):
            graph = Graph(weights, cfg, logger)
            graph.layer(layer, length)
            with writer.open_section(f"layer.{layer}.plan") as section:
                graph.save(section)
            del graph
            gc.collect()
        graph = Graph(weights, cfg, logger)
        hidden = graph.net.add_input("hidden", trt.float16, (1, cfg["hidden_size"]))
        graph.output(
            graph.cast(graph.linear(graph.norm(hidden, "model.norm"), "lm_head"), trt.float32),
            "logits",
        )
        with writer.open_section("head.plan") as section:
            graph.save(section)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            layers=cfg["num_hidden_layers"],
            hidden_size=cfg["hidden_size"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
            eos_ids=generation["eos_token_id"],
        ),
    )
