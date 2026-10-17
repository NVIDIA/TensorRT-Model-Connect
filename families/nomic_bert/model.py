# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
from safetensors import safe_open

from .cli import coerce_request
from .config import ModelConfig


def load_weights(path, config):
    weights = {}
    shapes = {
        "embeddings.word_embeddings.weight": (config.vocab_size, 768),
        "embeddings.token_type_embeddings.weight": (2, 768),
        "emb_ln.weight": (768,),
        "emb_ln.bias": (768,),
    }
    for index in range(12):
        prefix = f"encoder.layers.{index}."
        shapes.update(
            {
                prefix + key: shape
                for key, shape in {
                    "attn.Wqkv.weight": (2304, 768),
                    "attn.out_proj.weight": (768, 768),
                    "mlp.fc11.weight": (3072, 768),
                    "mlp.fc12.weight": (3072, 768),
                    "mlp.fc2.weight": (768, 3072),
                    "norm1.weight": (768,),
                    "norm1.bias": (768,),
                    "norm2.weight": (768,),
                    "norm2.bias": (768,),
                }.items()
            }
        )
    with safe_open(str(path / "model.safetensors"), framework="numpy") as reader:
        for key, shape in shapes.items():
            value = reader.get_tensor(key)
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"invalid Nomic checkpoint tensor: {key}")
            weights[key] = np.ascontiguousarray(value)
    return weights


def build_engine(weights, config, length, verbose=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 1
    options.clear_flag(trt.BuilderFlag.TF32)

    def constant(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return network.add_constant(value.shape, trt.Weights(value)).get_output(0)

    def element(left, right, op):
        return network.add_elementwise(left, right, op).get_output(0)

    def linear(x, weight):
        return network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, constant(weight.T), trt.MatrixOperation.NONE
        ).get_output(0)

    def norm(x, prefix):
        layer = network.add_normalization_v2(
            x,
            constant(weights[prefix + ".weight"][None]),
            constant(weights[prefix + ".bias"][None]),
            2,
        )
        layer.epsilon = config.epsilon
        return layer.get_output(0)

    def heads(x):
        layer = network.add_shuffle(x)
        layer.reshape_dims = (length, 12, 64)
        layer.second_transpose = trt.Permutation([1, 0, 2])
        return layer.get_output(0)

    def rows(x):
        layer = network.add_shuffle(x)
        layer.first_transpose = trt.Permutation([1, 0, 2])
        layer.reshape_dims = (length, 768)
        return layer.get_output(0)

    ids = network.add_input("input_ids", trt.int32, (length,))
    mask = network.add_input("attention_mask", trt.float32, (length,))
    x = network.add_gather(
        constant(weights["embeddings.word_embeddings.weight"]), ids, 0
    ).get_output(0)
    x = element(
        x,
        constant(weights["embeddings.token_type_embeddings.weight"][0:1]),
        trt.ElementWiseOperation.SUM,
    )
    x = norm(x, "emb_ln")
    mask_row = network.add_shuffle(mask)
    mask_row.reshape_dims = (1, 1, length)
    attention_bias = element(
        element(constant(np.ones((1, 1, 1))), mask_row.get_output(0), trt.ElementWiseOperation.SUB),
        constant(np.full((1, 1, 1), -1e9)),
        trt.ElementWiseOperation.PROD,
    )
    frequencies = np.outer(
        np.arange(length, dtype=np.float32),
        1.0 / config.rotary_base ** (np.arange(0, 64, 2, dtype=np.float32) / 64),
    )
    cos = constant(np.cos(np.concatenate([frequencies, frequencies], axis=1))[None])
    sin = constant(np.sin(np.concatenate([frequencies, frequencies], axis=1))[None])

    def rotary(x):
        first = network.add_slice(x, (0, 0, 0), (12, length, 32), (1, 1, 1)).get_output(0)
        second = network.add_slice(x, (0, 0, 32), (12, length, 32), (1, 1, 1)).get_output(0)
        second = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
        rotated = network.add_concatenation([second, first])
        rotated.axis = 2
        return element(
            element(x, cos, trt.ElementWiseOperation.PROD),
            element(rotated.get_output(0), sin, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM,
        )

    for index in range(12):
        prefix = f"encoder.layers.{index}."
        qkv = linear(x, weights[prefix + "attn.Wqkv.weight"])
        query, key, value = [
            heads(network.add_slice(qkv, (0, offset), (length, 768), (1, 1)).get_output(0))
            for offset in (0, 768, 1536)
        ]
        query, key = rotary(query), rotary(key)
        scores = network.add_matrix_multiply(
            query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = element(
            scores, constant(np.full((1, 1, 1), 1.0 / 8)), trt.ElementWiseOperation.PROD
        )
        scores = element(scores, attention_bias, trt.ElementWiseOperation.SUM)
        softmax = network.add_softmax(scores)
        softmax.axes = 4
        context = network.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        attention = linear(rows(context), weights[prefix + "attn.out_proj.weight"])
        x = norm(element(x, attention, trt.ElementWiseOperation.SUM), prefix + "norm1")
        gate = linear(x, weights[prefix + "mlp.fc12.weight"])
        gate = element(
            gate,
            network.add_activation(gate, trt.ActivationType.SIGMOID).get_output(0),
            trt.ElementWiseOperation.PROD,
        )
        mlp = element(
            linear(x, weights[prefix + "mlp.fc11.weight"]), gate, trt.ElementWiseOperation.PROD
        )
        mlp = linear(mlp, weights[prefix + "mlp.fc2.weight"])
        x = norm(element(x, mlp, trt.ElementWiseOperation.SUM), prefix + "norm2")

    mask_column = network.add_shuffle(mask)
    mask_column.reshape_dims = (length, 1)
    masked = element(x, mask_column.get_output(0), trt.ElementWiseOperation.PROD)
    total = network.add_reduce(masked, trt.ReduceOperation.SUM, 1, True).get_output(0)
    count = network.add_reduce(
        mask_column.get_output(0), trt.ReduceOperation.SUM, 1, True
    ).get_output(0)
    pooled = element(total, count, trt.ElementWiseOperation.DIV)
    squared = element(pooled, pooled, trt.ElementWiseOperation.PROD)
    magnitude = network.add_reduce(squared, trt.ReduceOperation.SUM, 2, True).get_output(0)
    magnitude = network.add_unary(magnitude, trt.UnaryOperation.SQRT).get_output(0)
    magnitude = element(magnitude, constant(np.full((1, 1), 1e-12)), trt.ElementWiseOperation.MAX)
    output = element(pooled, magnitude, trt.ElementWiseOperation.DIV)
    output.name = "embedding"
    network.mark_output(output)
    plan = builder.build_serialized_network(network, options)
    if plan is None:
        raise RuntimeError("Nomic TensorRT engine build failed")
    return bytes(plan)


def build(request, writer):
    request = coerce_request(request)
    path = request.model_dir
    config = ModelConfig.from_dir(path)
    tokenizer = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if (
        tokenizer.get("model", {}).get("type") != "WordPiece"
        or vocab.get("[PAD]") != 0
        or vocab.get("[UNK]") != 100
        or vocab.get("[CLS]") != 101
        or vocab.get("[SEP]") != 102
        or any(type(index) is not int for index in vocab.values())
        or sorted(vocab.values()) != list(range(len(vocab)))
        or len(vocab) > config.vocab_size
        or tokenizer.get("normalizer", {}).get("type") != "BertNormalizer"
        or tokenizer.get("pre_tokenizer", {}).get("type") != "BertPreTokenizer"
        or tokenizer.get("post_processor", {}).get("type") != "TemplateProcessing"
        or tokenizer.get("post_processor", {}).get("single")
        != [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
        ]
    ):
        raise ValueError("unsupported Nomic WordPiece tokenizer framing")
    weights = load_weights(path, config)
    writer.set_header(family="nomic_bert", task=request.task, backend=request.backend)
    writer.add_bytes(
        "engine.plan", build_engine(weights, config, request.max_sequence_length, request.verbose)
    )
    writer.add_bytes("tokenizer.json", (path / "tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": request.max_sequence_length,
            "vocab_size": config.vocab_size,
            "hidden_size": 768,
            "embedding_space": "",
        },
    )
