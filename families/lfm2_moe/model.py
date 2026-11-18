# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native incremental LFM2 MoE decoder with short convolution and attention state."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorrt as trt


def load_weights(directory, cfg):
    from safetensors import safe_open

    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    # Pack original expert tensors directly into their graph layout. Avoid a
    # second complete fp32 checkpoint while retaining every original tensor.
    with ExitStack() as stack:
        files = {
            name: stack.enter_context(
                safe_open(str(directory / name), framework="pt", device="cpu")
            )
            for name in set(index.values())
        }

        def read(key):
            return files[index[key]].get_tensor(key).float().numpy()

        weights = {key: read(key) for key in index if ".feed_forward.experts." not in key}
        experts, middle, width = (
            cfg["num_experts"],
            cfg["moe_intermediate_size"],
            cfg["hidden_size"],
        )
        for layer in range(cfg["num_hidden_layers"]):
            if layer < cfg["num_dense_layers"]:
                continue
            prefix = f"model.layers.{layer}.feed_forward.experts"
            up = np.empty((experts, 2 * middle, width), dtype=np.float32)
            down = np.empty((experts, width, middle), dtype=np.float32)
            for expert in range(experts):
                up[expert, :middle] = read(f"{prefix}.{expert}.w1.weight")
                up[expert, middle:] = read(f"{prefix}.{expert}.w3.weight")
                down[expert] = read(f"{prefix}.{expert}.w2.weight")
            weights[prefix + ".gate_up"] = up
            weights[prefix + ".down"] = down
    if "lm_head.weight" not in weights and cfg["tie_word_embeddings"]:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights


def build_engine(weights, cfg, length, stream, verbose=False):
    import torch
    from transformers import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeRotaryEmbedding

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width, heads, kv, dim = (
        cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    top = cfg["num_experts_per_tok"]
    middle = cfg["moe_intermediate_size"]

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def op(a, b, kind):
        return net.add_elementwise(a, b, kind).get_output(0)

    def add(a, b):
        return op(a, b, trt.ElementWiseOperation.SUM)

    def mul(a, b):
        return op(a, b, trt.ElementWiseOperation.PROD)

    def reshape(x, shape):
        layer = net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def linear(x, key):
        return net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            const(weights[key + ".weight"]),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)

    def norm(x, key):
        axis = len(x.shape) - 1
        average = net.add_reduce(mul(x, x), trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
        ones = (1,) * len(x.shape)
        root = net.add_unary(
            add(average, const(np.full(ones, cfg["norm_eps"]))), trt.UnaryOperation.SQRT
        ).get_output(0)
        gamma_shape = (1,) * axis + (int(x.shape[-1]),)
        return mul(
            op(x, root, trt.ElementWiseOperation.DIV),
            const(weights[key + ".weight"].reshape(gamma_shape)),
        )

    def silu(x):
        return mul(x, net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0))

    token = net.add_input("input_ids", trt.int32, (1,))
    position = net.add_input("position", trt.int32, (1,))
    indices = net.add_constant((length,), np.arange(length, dtype=np.int32)).get_output(0)
    current = op(indices, position, trt.ElementWiseOperation.EQUAL)
    write = reshape(current, (1, 1, length, 1))
    future = op(indices, position, trt.ElementWiseOperation.GREATER)
    mask = reshape(
        mul(net.add_cast(future, trt.float32).get_output(0), const(np.array([-1e9]))),
        (1, 1, 1, 1, length),
    )
    hidden = net.add_gather(const(weights["model.embed_tokens.weight"]), token, 0).get_output(0)
    rotary = Lfm2MoeRotaryEmbedding(Lfm2MoeConfig(**cfg))
    with torch.no_grad():
        cosine, sine = rotary(torch.empty(1, length, width), torch.arange(length)[None, :])
    cosine, sine = tuple(
        reshape(net.add_gather(const(x.numpy()[0]), position, 0).get_output(0), (1, 1, 1, dim))
        for x in (cosine, sine)
    )

    def dense(x, key):
        return linear(mul(silu(linear(x, key + ".w1")), linear(x, key + ".w3")), key + ".w2")

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normal = norm(hidden, prefix + ".operator_norm")
        if cfg["layer_types"][layer] == "full_attention":
            query = norm(
                reshape(linear(normal, prefix + ".self_attn.q_proj"), (heads, dim)),
                prefix + ".self_attn.q_layernorm",
            )
            key = norm(
                reshape(linear(normal, prefix + ".self_attn.k_proj"), (kv, dim)),
                prefix + ".self_attn.k_layernorm",
            )
            query = reshape(query, (1, heads, 1, dim))
            key = reshape(key, (1, kv, 1, dim))
            value = reshape(linear(normal, prefix + ".self_attn.v_proj"), (1, kv, 1, dim))

            def rope(x, nheads):
                first = net.add_slice(
                    x, (0, 0, 0, 0), (1, nheads, 1, dim // 2), (1, 1, 1, 1)
                ).get_output(0)
                second = net.add_slice(
                    x, (0, 0, 0, dim // 2), (1, nheads, 1, dim // 2), (1, 1, 1, 1)
                ).get_output(0)
                second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
                concat = net.add_concatenation([second, first])
                concat.axis = 3
                return add(mul(x, cosine), mul(concat.get_output(0), sine))

            query = rope(query, heads)
            key = rope(key, kv)
            old_k = net.add_input(f"cache_k_{layer}", trt.float32, (1, kv, length, dim))
            old_v = net.add_input(f"cache_v_{layer}", trt.float32, (1, kv, length, dim))
            key = net.add_select(write, key, old_k).get_output(0)
            value = net.add_select(write, value, old_v).get_output(0)
            for tensor, name in ((key, f"present_k_{layer}"), (value, f"present_v_{layer}")):
                tensor.name = name
                net.mark_output(tensor)
            query = reshape(query, (1, kv, heads // kv, 1, dim))
            grouped_key = reshape(key, (1, kv, 1, length, dim))
            grouped_value = reshape(value, (1, kv, 1, length, dim))
            scores = net.add_matrix_multiply(
                query, trt.MatrixOperation.NONE, grouped_key, trt.MatrixOperation.TRANSPOSE
            ).get_output(0)
            scores = add(mul(scores, const(np.full((1, 1, 1, 1, 1), dim**-0.5))), mask)
            probability = net.add_softmax(scores)
            probability.axes = 16
            attended = net.add_matrix_multiply(
                probability.get_output(0),
                trt.MatrixOperation.NONE,
                grouped_value,
                trt.MatrixOperation.NONE,
            ).get_output(0)
            hidden = add(
                hidden, linear(reshape(attended, (1, heads * dim)), prefix + ".self_attn.out_proj")
            )
        else:
            kernel = cfg["conv_L_cache"]
            projected = linear(normal, prefix + ".conv.in_proj")
            b, c, x = [
                net.add_slice(projected, (0, i * width), (1, width), (1, 1)).get_output(0)
                for i in range(3)
            ]
            bx = reshape(mul(b, x), (1, width, 1))
            history = net.add_input(f"cache_conv_{layer}", trt.float32, (1, width, kernel - 1))
            joined = net.add_concatenation([history, bx])
            joined.axis = 2
            taps = joined.get_output(0)
            present = net.add_slice(taps, (0, 0, 1), (1, width, kernel - 1), (1, 1, 1)).get_output(
                0
            )
            present.name = f"present_conv_{layer}"
            net.mark_output(present)
            filtered = net.add_reduce(
                mul(taps, const(weights[prefix + ".conv.conv.weight"].reshape(1, width, kernel))),
                trt.ReduceOperation.SUM,
                4,
                False,
            ).get_output(0)
            hidden = add(hidden, linear(mul(c, filtered), prefix + ".conv.out_proj"))
        normal = norm(hidden, prefix + ".ffn_norm")
        if layer < cfg["num_dense_layers"]:
            hidden = add(hidden, dense(normal, prefix + ".feed_forward"))
            continue
        router = linear(normal, prefix + ".feed_forward.gate")
        probability = net.add_activation(router, trt.ActivationType.SIGMOID).get_output(0)
        corrected = add(
            probability, const(weights[prefix + ".feed_forward.expert_bias"].reshape(1, -1))
        )
        selection = net.add_topk(corrected, trt.TopKOperation.MAX, top, 2)
        expert_ids = reshape(selection.get_output(1), (top,))
        scores = net.add_gather(probability, expert_ids, 1).get_output(0)
        total = net.add_reduce(scores, trt.ReduceOperation.SUM, 2, True).get_output(0)
        total = add(total, const(np.array([[1e-6]])))
        scores = reshape(
            mul(
                op(scores, total, trt.ElementWiseOperation.DIV),
                const(np.array([[cfg["routed_scaling_factor"]]])),
            ),
            (top, 1, 1),
        )
        up = net.add_gather(
            const(weights[prefix + ".feed_forward.experts.gate_up"]), expert_ids, 0
        ).get_output(0)
        down = net.add_gather(
            const(weights[prefix + ".feed_forward.experts.down"]), expert_ids, 0
        ).get_output(0)
        projected = net.add_matrix_multiply(
            reshape(normal, (1, 1, width)),
            trt.MatrixOperation.NONE,
            up,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        gate = net.add_slice(projected, (0, 0, 0), (top, 1, middle), (1, 1, 1)).get_output(0)
        value = net.add_slice(projected, (0, 0, middle), (top, 1, middle), (1, 1, 1)).get_output(0)
        result = net.add_matrix_multiply(
            mul(silu(gate), value), trt.MatrixOperation.NONE, down, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        merged = net.add_reduce(mul(result, scores), trt.ReduceOperation.SUM, 1, False).get_output(
            0
        )
        hidden = add(hidden, merged)
    logits = linear(norm(hidden, "model.embedding_norm"), "lm_head")
    logits.name = "logits"
    net.mark_output(logits)
    if not builder.build_serialized_network_to_stream(net, options, PlanWriter(stream)):
        raise RuntimeError("LFM2 MoE engine build failed")


class PlanWriter(trt.IStreamWriter):
    def __init__(self, stream):
        trt.IStreamWriter.__init__(self)
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)


def build(request, writer):
    if request.dynamic_kv_cache:
        raise NotImplementedError("lfm2_moe does not support dynamic_kv_cache")
    if request.task != "text_continuation" or request.backend != "trt":
        raise ValueError("Lfm2Moe requires text_continuation with backend=trt")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.image_height is not None
        or request.image_width is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("Lfm2Moe supports one fp32 sequence with a fixed KV profile")
    directory = Path(request.model_dir)
    from transformers import Lfm2MoeConfig

    reference_config = Lfm2MoeConfig.from_pretrained(directory)
    cfg = reference_config.to_dict()
    cfg["head_dim"] = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    length = request.max_sequence_length or 128
    if (
        cfg["model_type"] != "lfm2_moe"
        or cfg["conv_bias"]
        or not cfg["norm_topk_prob"]
        or not cfg["use_expert_bias"]
        or not cfg["tie_word_embeddings"]
        or cfg["conv_L_cache"] < 2
        or len(cfg["layer_types"]) != cfg["num_hidden_layers"]
        or set(cfg["layer_types"]) - {"conv", "full_attention"}
    ):
        raise ValueError("Unsupported LFM2 MoE attention/convolution or routing configuration")
    if not 2 <= length <= min(1024, cfg["max_position_embeddings"]):
        raise ValueError("LFM2 MoE fixed state profile must be in [2, 1024]")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    generation = json.loads((directory / "generation_config.json").read_text())
    writer.set_header(family="lfm2_moe", task=request.task, backend=request.backend)
    with writer.open_section("model.plan") as section:
        build_engine(load_weights(directory, cfg), cfg, length, section, request.verbose)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            layers=cfg["num_hidden_layers"],
            layer_types=cfg["layer_types"],
            conv_kernel=cfg["conv_L_cache"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
            eos_id=cfg["eos_token_id"],
            repetition_penalty=generation.get("repetition_penalty", 1.0),
        ),
    )
