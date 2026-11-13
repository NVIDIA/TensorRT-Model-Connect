# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional rotary encoder with native query and passage LoRA selection."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt


def build_engine(weights, adapters, cfg, length, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    dim = width // heads

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

    def mm(x, w, transpose=False):
        return net.add_matrix_multiply(
            x,
            trt.MatrixOperation.NONE,
            w,
            trt.MatrixOperation.TRANSPOSE if transpose else trt.MatrixOperation.NONE,
        ).get_output(0)

    adapter_id = net.add_input("adapter_id", trt.int32, (1,))

    def selected(key):
        query, passage = (adapter[key] for adapter in adapters)
        table = const(np.stack([np.zeros_like(query), query, passage]))
        return reshape(net.add_gather(table, adapter_id, 0).get_output(0), query.shape)

    def lora(x, key):
        return mul(
            mm(
                mm(x, selected(key + ".lora_A.weight"), True),
                selected(key + ".lora_B.weight"),
                True,
            ),
            const(np.array([[0.25]])),
        )

    def linear(x, key, adapter_key=None, rows=None):
        weight, bias = weights[key + ".weight"], weights[key + ".bias"]
        if rows is not None:
            weight, bias = weight[rows], bias[rows]
        output = add(mm(x, const(weight), True), const(bias.reshape(1, -1)))
        return add(output, lora(x, adapter_key or key))

    def norm(x, key):
        layer = net.add_normalization(
            x,
            const(weights[key + ".weight"].reshape(1, -1)),
            const(weights[key + ".bias"].reshape(1, -1)),
            2,
        )
        layer.epsilon = cfg["layer_norm_eps"]
        return layer.get_output(0)

    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))

    def embeddings(key, token_ids):
        output = net.add_gather(const(weights[key + ".weight"]), token_ids, 0).get_output(0)
        a = selected(key + ".lora_embedding_A")
        b = selected(key + ".lora_embedding_B")
        transpose = net.add_shuffle(a)
        transpose.first_transpose = (1, 0)
        low = net.add_gather(transpose.get_output(0), token_ids, 0).get_output(0)
        return add(output, mul(mm(low, b, True), const(np.array([[0.25]]))))

    hidden = embeddings("roberta.embeddings.word_embeddings", ids)
    zero = net.add_constant((1,), np.zeros(1, dtype=np.int32)).get_output(0)
    hidden = norm(
        add(hidden, embeddings("roberta.embeddings.token_type_embeddings", zero)), "roberta.emb_ln"
    )
    valid_float = net.add_cast(valid, trt.float32).get_output(0)
    mask = reshape(
        mul(
            op(const(np.ones(1)), valid_float, trt.ElementWiseOperation.SUB),
            const(np.array([-1e9])),
        ),
        (1, 1, 1, length),
    )
    frequency = 1 / (cfg["rotary_emb_base"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = np.arange(length, dtype=np.float32)[:, None] * frequency
    angles = np.concatenate([angles, angles], axis=-1)
    cosine = const(np.cos(angles).reshape(1, 1, length, dim))
    sine = const(np.sin(angles).reshape(1, 1, length, dim))

    def split(x):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, length, heads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def rope(x):
        first = net.add_slice(
            x, (0, 0, 0, 0), (1, heads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_slice(
            x, (0, 0, 0, dim // 2), (1, heads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        negative = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        layer = net.add_concatenation([negative, first])
        layer.axis = 3
        return add(mul(x, cosine), mul(layer.get_output(0), sine))

    for index in range(cfg["num_hidden_layers"]):
        prefix = f"roberta.encoder.layers.{index}"
        q, k, v = [
            split(
                linear(
                    hidden,
                    prefix + ".mixer.Wqkv",
                    prefix + ".self_attn." + name + "_proj",
                    slice(i * width, (i + 1) * width),
                )
            )
            for i, name in enumerate(("q", "k", "v"))
        ]
        scores = mm(rope(q), rope(k), True)
        scores = add(mul(scores, const(np.array([[[[dim**-0.5]]]]))), mask)
        softmax = net.add_softmax(scores)
        softmax.axes = 8
        context = mm(softmax.get_output(0), v)
        layer = net.add_shuffle(context)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (length, width)
        hidden = norm(
            add(hidden, linear(layer.get_output(0), prefix + ".mixer.out_proj")), prefix + ".norm1"
        )
        mlp = linear(hidden, prefix + ".mlp.fc1")
        mlp = net.add_activation(mlp, trt.ActivationType.GELU_ERF).get_output(0)
        hidden = norm(add(hidden, linear(mlp, prefix + ".mlp.fc2")), prefix + ".norm2")
    total = net.add_reduce(
        mul(hidden, reshape(valid_float, (length, 1))), trt.ReduceOperation.SUM, 1, True
    ).get_output(0)
    count = reshape(
        net.add_reduce(valid_float, trt.ReduceOperation.SUM, 1, True).get_output(0), (1, 1)
    )
    mean = op(total, count, trt.ElementWiseOperation.DIV)
    norm2 = net.add_reduce(mul(mean, mean), trt.ReduceOperation.SUM, 2, True).get_output(0)
    output = op(
        mean,
        net.add_unary(norm2, trt.UnaryOperation.SQRT).get_output(0),
        trt.ElementWiseOperation.DIV,
    )
    output.name = "embedding"
    net.mark_output(output)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("Jina embedding engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "text_to_embedding" or request.backend != "trt":
        raise ValueError("Jina embeddings requires text_to_embedding and backend=trt")
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
        raise NotImplementedError("Jina embedding supports a single fp32 text input")
    directory = Path(request.model_dir)
    cfg = json.loads((directory / "config.json").read_text())
    length = request.max_sequence_length or 256
    if cfg["model_type"] != "jina_embeddings_v3" or cfg["hidden_act"] != "gelu":
        raise ValueError("Expected Jina embeddings v3 GELU encoder")
    if not 2 <= length <= min(8192, cfg["max_position_embeddings"]):
        raise ValueError("Jina embedding requires a sequence profile in [2, 8192]")
    from safetensors import safe_open

    def load(filename):
        with safe_open(str(directory / filename), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key).float().numpy() for key in handle.keys()}

    weights = load("model.safetensors")
    adapters = []
    for name in ("retrieval_query", "retrieval_passage"):
        config = json.loads((directory / name / "adapter_config.json").read_text())
        if config["r"] != 4 or config["lora_alpha"] != 1:
            raise ValueError("Jina retrieval adapter scaling must be alpha=1, rank=4")
        adapters.append(load(name + "/adapter_model.safetensors"))
    writer.set_header(family="jina_embeddings_v3", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, adapters, cfg, length, request.verbose))
    writer.add_bytes("tokenizer.json", (directory / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            pad_token_id=cfg["pad_token_id"],
            vocab_size=cfg["vocab_size"],
        ),
    )
