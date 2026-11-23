# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native bidirectional Ministral text encoder for Nemotron 3.5 embeddings."""

import json
from pathlib import Path
import numpy as np
import tensorrt as trt


def _validate_tokenizer(tokenizer):
    pretokenizer = tokenizer.get("pre_tokenizer") or {}
    parts = pretokenizer.get("pretokenizers", [])
    if (
        pretokenizer.get("type") != "Sequence"
        or len(parts) != 2
        or parts[0].get("type") != "Split"
        or not isinstance(parts[0].get("pattern", {}).get("Regex"), str)
        or parts[1].get("type") != "ByteLevel"
        or parts[1].get("use_regex", True)
        or parts[1].get("add_prefix_space", True)
    ):
        raise ValueError(
            "Nemotron embedding tokenizer requires Sequence[Split(Regex), ByteLevel] "
            "with use_regex=false and add_prefix_space=false; "
            f"got {pretokenizer!r}"
        )


def build_engine(weights, cfg, length, verbose=False):
    from transformers import Ministral3Config
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding
    import torch

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    dim = cfg["head_dim"]

    def const(value, shape=None):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(tuple(value.shape) if shape is None else shape, value).get_output(0)

    def op(left, right, kind):
        return net.add_elementwise(left, right, kind).get_output(0)

    def add(left, right):
        return op(left, right, trt.ElementWiseOperation.SUM)

    def mul(left, right):
        return op(left, right, trt.ElementWiseOperation.PROD)

    def linear(x, key):
        value = weights[key + ".weight"]
        return net.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, const(value), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def norm(x, key):
        square = mul(x, x)
        mean = net.add_reduce(square, trt.ReduceOperation.AVG, 2, True).get_output(0)
        rsqrt = net.add_unary(
            add(mean, const(np.array([[cfg["rms_norm_eps"]]]))), trt.UnaryOperation.SQRT
        ).get_output(0)
        return mul(
            op(x, rsqrt, trt.ElementWiseOperation.DIV), const(weights[key + ".weight"], (1, width))
        )

    rotary = Ministral3RotaryEmbedding(Ministral3Config(**cfg))
    with torch.no_grad():
        cosine, sine = rotary(torch.empty(1, length, width), torch.arange(length).reshape(1, -1))
    cosine = const(cosine.numpy(), (1, 1, length, dim))
    sine = const(sine.numpy(), (1, 1, length, dim))

    def split(x, nheads):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, length, nheads, dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    def rope(x, nheads):
        first = net.add_slice(
            x, (0, 0, 0, 0), (1, nheads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_slice(
            x, (0, 0, 0, dim // 2), (1, nheads, length, dim // 2), (1, 1, 1, 1)
        ).get_output(0)
        second = net.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        concat = net.add_concatenation([second, first])
        concat.axis = 3
        return add(mul(x, cosine), mul(concat.get_output(0), sine))

    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    hidden = net.add_gather(
        const(weights["language_model.embed_tokens.weight"]), ids, 0
    ).get_output(0)
    valid_float = net.add_cast(valid, trt.float32).get_output(0)
    mask = mul(
        op(const(np.ones(1)), valid_float, trt.ElementWiseOperation.SUB), const(np.array([-1e9]))
    )
    reshape = net.add_shuffle(mask)
    reshape.reshape_dims = (1, 1, 1, 1, length)
    mask = reshape.get_output(0)

    def grouped(x, groups):
        layer = net.add_shuffle(x)
        layer.reshape_dims = (1, kv_heads, groups, length, dim)
        return layer.get_output(0)

    for index in range(cfg["num_hidden_layers"]):
        prefix = f"language_model.layers.{index}"
        normalized = norm(hidden, prefix + ".input_layernorm")
        query = rope(split(linear(normalized, prefix + ".self_attn.q_proj"), heads), heads)
        key = rope(split(linear(normalized, prefix + ".self_attn.k_proj"), kv_heads), kv_heads)
        value = split(linear(normalized, prefix + ".self_attn.v_proj"), kv_heads)
        query = grouped(query, heads // kv_heads)
        key = grouped(key, 1)
        value = grouped(value, 1)
        scores = net.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = add(mul(scores, const(np.array([1 / np.sqrt(dim)]), (1, 1, 1, 1, 1))), mask)
        probability = net.add_softmax(scores)
        probability.axes = 16
        attended = net.add_matrix_multiply(
            probability.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        reshape = net.add_shuffle(attended)
        reshape.first_transpose = (0, 3, 1, 2, 4)
        reshape.reshape_dims = (length, heads * dim)
        hidden = add(hidden, linear(reshape.get_output(0), prefix + ".self_attn.o_proj"))
        normalized = norm(hidden, prefix + ".post_attention_layernorm")
        gate = linear(normalized, prefix + ".mlp.gate_proj")
        gate = mul(gate, net.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0))
        hidden = add(
            hidden,
            linear(
                mul(gate, linear(normalized, prefix + ".mlp.up_proj")), prefix + ".mlp.down_proj"
            ),
        )
    hidden = norm(hidden, "language_model.norm")
    reshape = net.add_shuffle(valid_float)
    reshape.reshape_dims = (length, 1)
    total = net.add_reduce(
        mul(hidden, reshape.get_output(0)), trt.ReduceOperation.SUM, 1, True
    ).get_output(0)
    count = net.add_reduce(valid_float, trt.ReduceOperation.SUM, 1, True).get_output(0)
    reshape = net.add_shuffle(count)
    reshape.reshape_dims = (1, 1)
    mean = op(total, reshape.get_output(0), trt.ElementWiseOperation.DIV)
    norm2 = net.add_reduce(mul(mean, mean), trt.ReduceOperation.SUM, 2, True).get_output(0)
    denominator = net.add_unary(norm2, trt.UnaryOperation.SQRT).get_output(0)
    output = op(mean, denominator, trt.ElementWiseOperation.DIV)
    output.name = "embedding"
    net.mark_output(output)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("Nemotron embedding engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "text_to_embedding" or request.backend != "trt":
        raise ValueError("Nemotron embedding requires text_to_embedding and backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("nemotron_embed does not support dynamic_kv_cache")
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
        raise NotImplementedError("Nemotron text embedding supports a single fp32 sequence")
    directory = Path(request.model_dir)
    raw = json.loads((directory / "config.json").read_text())
    cfg = raw["text_config"]
    if cfg.get("hidden_act", "silu") != "silu":
        raise ValueError("Nemotron embedding requires hidden_act=silu")
    _validate_tokenizer(json.loads((directory / "tokenizer.json").read_text()))
    if (
        raw.get("model_type") != "mistral3"
        or cfg.get("model_type") != "ministral3"
        or cfg.get("is_causal", True)
    ):
        raise ValueError("Expected the bidirectional Nemotron embedding Ministral3 checkpoint")
    length = request.max_sequence_length or 256
    if length <= 0 or length > cfg["rope_parameters"]["original_max_position_embeddings"]:
        raise ValueError("Nemotron embedding profile must remain inside the original RoPE context")
    pooling = json.loads((directory / "1_Pooling/config.json").read_text())
    if not pooling.get("pooling_mode_mean_tokens") or not pooling.get("include_prompt"):
        raise ValueError("Nemotron embedding requires mean pooling including prompt tokens")
    from safetensors import safe_open

    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for filename in sorted(set(index.values())):
        with safe_open(str(directory / filename), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("language_model."):
                    weights[key] = handle.get_tensor(key).float().numpy()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    prompts = json.loads((directory / "config_sentence_transformers.json").read_text())["prompts"]
    writer.set_header(family="nemotron_embed", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, cfg, length, request.verbose))
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            hidden_size=cfg["hidden_size"],
            pad_token_id=cfg["pad_token_id"],
            vocab_size=cfg["vocab_size"],
            prompts=prompts,
        ),
    )
