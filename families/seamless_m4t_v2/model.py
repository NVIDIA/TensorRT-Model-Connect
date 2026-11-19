# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SeamlessM4T v2's text encoder and multilingual text decoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from .builders import Graph


def build_engine(weights, cfg, length, *, decoder=False, verbose=False):
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
        SeamlessM4Tv2SinusoidalPositionalEmbedding,
    )

    graph = Graph(weights, verbose)
    net = graph.net
    width = cfg["hidden_size"]
    prefix = "text_decoder" if decoder else "text_encoder"
    ids = net.add_input("input_ids", trt.int32, (length,))
    valid = net.add_input("attention_mask", trt.int32, (length,))
    positions = net.add_input("position_ids", trt.int32, (length,))
    hidden = net.add_gather(graph.const(weights["shared.weight"]), ids, 0).get_output(0)
    hidden = graph.mul(
        hidden, graph.const(np.full((1, 1), width**0.5 if cfg["scale_embedding"] else 1))
    )
    table = SeamlessM4Tv2SinusoidalPositionalEmbedding.get_embedding(
        length + 2, width, cfg["pad_token_id"]
    ).numpy()
    hidden = graph.add(hidden, net.add_gather(graph.const(table), positions, 0).get_output(0))

    def mask(tensor):
        valid = graph.reshape(net.add_cast(tensor, trt.float32).get_output(0), (1, 1, 1, length))
        invalid = net.add_elementwise(
            graph.const(np.ones((1, 1, 1, 1))), valid, trt.ElementWiseOperation.SUB
        ).get_output(0)
        return graph.mul(invalid, graph.const(np.full((1, 1, 1, 1), -1e9)))

    attention_mask = mask(valid)
    if decoder:
        memory = net.add_input("encoder_hidden", trt.float32, (length, width))
        source_valid = net.add_input("encoder_mask", trt.int32, (length,))
        source_mask = mask(source_valid)
        last = net.add_input("last_index", trt.int32, (1,))
        attention_mask = graph.add(
            attention_mask,
            graph.const(np.triu(np.full((1, 1, length, length), -1e9, np.float32), k=1)),
        )
    heads = cfg["decoder_attention_heads" if decoder else "encoder_attention_heads"]
    for index in range(cfg["decoder_layers" if decoder else "encoder_layers"]):
        name = f"{prefix}.layers.{index}"
        normalized = graph.norm(hidden, name + ".self_attn_layer_norm", 1e-5)
        q, k, v = [
            graph.linear(normalized, name + ".self_attn." + key)
            for key in ("q_proj", "k_proj", "v_proj")
        ]
        attended = graph.attention(q, k, v, heads, attention_mask)
        hidden = graph.add(hidden, graph.linear(attended, name + ".self_attn.out_proj"))
        if decoder:
            normalized = graph.norm(hidden, name + ".cross_attention_layer_norm", 1e-5)
            q = graph.linear(normalized, name + ".cross_attention.q_proj")
            k = graph.linear(memory, name + ".cross_attention.k_proj")
            v = graph.linear(memory, name + ".cross_attention.v_proj")
            attended = graph.attention(q, k, v, heads, source_mask)
            hidden = graph.add(hidden, graph.linear(attended, name + ".cross_attention.out_proj"))
        inner = graph.linear(graph.norm(hidden, name + ".ffn_layer_norm", 1e-5), name + ".ffn.fc1")
        inner = net.add_activation(inner, trt.ActivationType.RELU).get_output(0)
        hidden = graph.add(hidden, graph.linear(inner, name + ".ffn.fc2"))
    hidden = graph.norm(hidden, prefix + ".layer_norm", 1e-5)
    if decoder:
        hidden = graph.linear(net.add_gather(hidden, last, 0).get_output(0), "lm_head")
        hidden.name = "logits"
    else:
        hidden.name = "encoder_hidden"
    net.mark_output(hidden)
    return graph.finish()


def build(request, writer):
    if request.task != "text_translation" or request.backend != "trt":
        raise ValueError("SeamlessM4T v2 requires text_translation with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("seamless_m4t_v2 does not support dynamic_kv_cache")
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
        raise NotImplementedError("SeamlessM4T v2 supports one fp32 text translation")
    from transformers import AutoTokenizer, SeamlessM4Tv2Config
    from safetensors import safe_open

    directory = Path(request.model_dir)
    cfg = SeamlessM4Tv2Config.from_pretrained(directory).to_dict()
    if cfg["activation_function"] != "relu" or cfg["layer_norm_eps"] != 1e-5:
        raise ValueError("Unsupported SeamlessM4T v2 text feed-forward or normalization")
    length = request.max_sequence_length or 128
    if not 3 <= length <= cfg["max_position_embeddings"]:
        raise ValueError(
            "SeamlessM4T v2 profile must fit source tokens and decoder language prefix"
        )
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {
        key: shard
        for key, shard in index.items()
        if key.startswith(("text_encoder.", "text_decoder."))
        or key in ("shared.weight", "lm_head.weight")
    }
    weights = {}
    for shard in sorted(set(selected.values())):
        with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
            for key in selected:
                if selected[key] == shard:
                    weights[key] = handle.get_tensor(key).float().numpy()
    if "lm_head.weight" not in weights and cfg["tie_word_embeddings"]:
        weights["lm_head.weight"] = weights["shared.weight"]
    tokenizer = AutoTokenizer.from_pretrained(directory)
    generation = json.loads((directory / "generation_config.json").read_text())
    languages = generation["text_decoder_lang_to_code_id"]
    for language, token in languages.items():
        if tokenizer.convert_tokens_to_ids(f"__{language}__") != token:
            raise ValueError("SeamlessM4T v2 tokenizer and generation language map disagree")
    writer.set_header(family="seamless_m4t_v2", task=request.task, backend=request.backend)
    writer.add_bytes("encoder.plan", build_engine(weights, cfg, length, verbose=request.verbose))
    writer.add_bytes(
        "decoder.plan", build_engine(weights, cfg, length, decoder=True, verbose=request.verbose)
    )
    writer.add_bytes("tokenizer.json", tokenizer.backend_tokenizer.to_str().encode())
    writer.add_json(
        "runtime.json",
        dict(
            sequence_length=length,
            source_width=cfg["hidden_size"],
            vocab_size=cfg["vocab_size"],
            bos_id=generation["decoder_start_token_id"],
            eos_id=generation["eos_token_id"],
            pad_id=generation["pad_token_id"],
            source_language="eng",
            languages=languages,
        ),
    )
