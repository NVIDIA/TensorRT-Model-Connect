# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native incremental Ernie4_5_Moe decoder with routed experts and fixed KV storage."""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorrt as trt

from .streaming import PlanWriter


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

        weights = {key: read(key) for key in index if ".mlp.experts." not in key}
        experts, middle, width = (
            cfg["num_experts"],
            cfg["moe_intermediate_size"],
            cfg["hidden_size"],
        )
        for layer in range(cfg["num_hidden_layers"]):
            if layer not in cfg["_sparse_layers"]:
                continue
            prefix = f"model.layers.{layer}.mlp.experts"
            up = np.empty((experts, 2 * middle, width), dtype=np.float32)
            down = np.empty((experts, width, middle), dtype=np.float32)
            for expert in range(experts):
                up[expert, :middle] = read(f"{prefix}.{expert}.gate_proj.weight")
                up[expert, middle:] = read(f"{prefix}.{expert}.up_proj.weight")
                down[expert] = read(f"{prefix}.{expert}.down_proj.weight")
            weights[prefix + ".gate_up"] = up
            weights[prefix + ".down"] = down
    if "lm_head.weight" not in weights and cfg["tie_word_embeddings"]:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    return weights


def build_engine(weights, cfg, length, stream, verbose=False):
    import torch
    from transformers import Ernie4_5_MoeConfig
    from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeRotaryEmbedding

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
            add(average, const(np.full(ones, cfg["rms_norm_eps"]))), trt.UnaryOperation.SQRT
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
    rotary = Ernie4_5_MoeRotaryEmbedding(Ernie4_5_MoeConfig(**cfg))
    with torch.no_grad():
        cosine, sine = rotary(torch.empty(1, length, width), torch.arange(length)[None, :])
    cosine, sine = tuple(
        reshape(
            net.add_gather(
                const(np.repeat(x.numpy()[0, :, : dim // 2], 2, axis=-1)), position, 0
            ).get_output(0),
            (1, 1, 1, dim),
        )
        for x in (cosine, sine)
    )

    def dense(x, key):
        return linear(
            mul(silu(linear(x, key + ".gate_proj")), linear(x, key + ".up_proj")),
            key + ".down_proj",
        )

    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        normal = norm(hidden, prefix + ".input_layernorm")
        query = reshape(linear(normal, prefix + ".self_attn.q_proj"), (1, heads, 1, dim))
        key = reshape(linear(normal, prefix + ".self_attn.k_proj"), (1, kv, 1, dim))
        value = reshape(linear(normal, prefix + ".self_attn.v_proj"), (1, kv, 1, dim))

        def rope(x, nheads):
            first = net.add_slice(
                x, (0, 0, 0, 0), (1, nheads, 1, dim // 2), (1, 1, 1, 2)
            ).get_output(0)
            second = net.add_slice(
                x, (0, 0, 0, 1), (1, nheads, 1, dim // 2), (1, 1, 1, 2)
            ).get_output(0)
            second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
            concat = net.add_concatenation(
                [
                    reshape(second, (1, nheads, 1, dim // 2, 1)),
                    reshape(first, (1, nheads, 1, dim // 2, 1)),
                ]
            )
            concat.axis = 4
            return add(
                mul(x, cosine), mul(reshape(concat.get_output(0), (1, nheads, 1, dim)), sine)
            )

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
            hidden, linear(reshape(attended, (1, heads * dim)), prefix + ".self_attn.o_proj")
        )
        normal = norm(hidden, prefix + ".post_attention_layernorm")
        if layer not in cfg["_sparse_layers"]:
            hidden = add(hidden, dense(normal, prefix + ".mlp"))
            continue
        router = linear(normal, prefix + ".mlp.gate")
        probability = net.add_softmax(router)
        probability.axes = 2
        corrected = add(
            probability.get_output(0),
            const(weights[prefix + ".mlp.moe_statics.e_score_correction_bias"].reshape(1, -1)),
        )
        selection = net.add_topk(corrected, trt.TopKOperation.MAX, top, 2)
        expert_ids = reshape(selection.get_output(1), (top,))
        scores = net.add_gather(probability.get_output(0), expert_ids, 1).get_output(0)
        total = net.add_reduce(scores, trt.ReduceOperation.SUM, 2, True).get_output(0)
        total = op(total, const(np.array([[cfg["moe_norm_min"]]])), trt.ElementWiseOperation.MAX)
        scores = reshape(op(scores, total, trt.ElementWiseOperation.DIV), (top, 1, 1))
        up = net.add_gather(
            const(weights[prefix + ".mlp.experts.gate_up"]), expert_ids, 0
        ).get_output(0)
        down = net.add_gather(
            const(weights[prefix + ".mlp.experts.down"]), expert_ids, 0
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
        if cfg["moe_num_shared_experts"]:
            merged = add(merged, dense(normal, prefix + ".mlp.shared_experts"))
        hidden = add(hidden, merged)
    logits = linear(norm(hidden, "model.norm"), "lm_head")
    logits.name = "logits"
    net.mark_output(logits)
    if not builder.build_serialized_network_to_stream(net, options, PlanWriter(stream)):
        raise RuntimeError("ERNIE engine build failed")


def build(request, writer):
    if request.task != "text_continuation" or request.backend != "trt":
        raise ValueError("Ernie4_5_Moe requires text_continuation with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("ernie4_5_moe does not support dynamic_kv_cache")
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
        raise NotImplementedError("Ernie4_5_Moe supports one fp32 sequence with a fixed KV profile")
    directory = Path(request.model_dir)
    from transformers import Ernie4_5_MoeConfig

    reference_config = Ernie4_5_MoeConfig.from_pretrained(directory)
    cfg = reference_config.to_dict()
    cfg["num_experts"] = reference_config.num_experts
    cfg["num_experts_per_tok"] = reference_config.num_experts_per_tok
    cfg["head_dim"] = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    cfg["_sparse_layers"] = [
        i
        for i in range(cfg["num_hidden_layers"])
        if (i + 1) % cfg["moe_layer_interval"] == 0
        and cfg["moe_layer_start_index"] <= i <= cfg["moe_layer_end_index"]
    ]
    length = request.max_sequence_length or 128
    if cfg["model_type"] != "ernie4_5_moe" or cfg["use_bias"] or cfg["hidden_act"] != "silu":
        raise ValueError("Expected ERNIE 4.5 MoE decoder without projection biases")
    if not 2 <= length <= min(1024, cfg["max_position_embeddings"]):
        raise ValueError("ERNIE fixed KV profile must be in [2, 1024]")
    eos_id = cfg.get("eos_token_id")
    if type(eos_id) is not int or not 0 <= eos_id < cfg["vocab_size"]:
        raise ValueError("ERNIE requires one scalar EOS token ID within the vocabulary")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    writer.set_header(family="ernie4_5_moe", task=request.task, backend=request.backend)
    with writer.open_section("model.plan") as section:
        build_engine(load_weights(directory, cfg), cfg, length, section, request.verbose)
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            layers=cfg["num_hidden_layers"],
            kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            vocab_size=cfg["vocab_size"],
            eos_id=cfg["eos_token_id"],
        ),
    )
