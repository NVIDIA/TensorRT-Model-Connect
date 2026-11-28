# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile and stage the ordinary Flash-Next language decoder for native execution."""

import tensorrt as trt

from .decoder import decoder_layer, embed_tokens, token_logits
from .expert_weights import write_expert
from .experts import routed_expert
from .moe import combine_experts
from .ple_weights import write_checkpoint_ngram_table


def _write_plan(writer, section, chunk, define):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    profile = builder.create_optimization_profile()

    def rows(name, dtype, *tail):
        tensor = network.add_input(name, dtype, (-1, *tail))
        tensor.set_dimension_name(0, "token_rows")
        profile.set_shape(name, (1, *tail), (min(4, chunk), *tail), (chunk, *tail))
        return tensor

    for name, tensor in define(network, rows).items():
        tensor.name = name
        network.mark_output(tensor)
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT could not build the native Flash-Next stage {section}")
    with writer.open_section(section) as sink:
        sink.write(plan)


def write_layer_plan(checkpoint, writer, config, layer, chunk, capacity):
    def define(n, rows):
        fp16, fp32 = trt.float16, trt.float32
        state = {}
        if layer + 1 in config["ple_layer_ids"]:
            state["lexical_words"] = rows("lexical_words", trt.int32, 640)
            state["lexical_history"] = n.add_input("lexical_history", fp16, (9, 10240))
        if config["layer_types"][layer] == "linear_attention":
            state["history"] = n.add_input("history", fp16, (3, 10240))
            state["state"] = n.add_input("state", fp32, (48, 128, 128))
        else:
            state["position"] = rows("position", trt.int32)
            state["rotary"] = n.add_input("rotary", fp32, (capacity, 64))
            for name in ("key", "value"):
                state[name] = n.add_input(name, fp16, (capacity, 2, 256))
            state["index_key"] = n.add_input("index_key", fp16, (capacity, 128))
        return decoder_layer(n, rows("hyper", fp16, 10240), state, checkpoint, config, layer, chunk)

    _write_plan(writer, f"layer.{layer}.plan", chunk, define)


def write_shared_decoder_plans(checkpoint, writer, config, chunk):
    def embedding(n, rows):
        return {
            "hyper": embed_tokens(
                n,
                rows("ids", trt.int32),
                rows("media_mask", trt.int32, 1),
                rows("media_vectors", trt.float16, 2560),
                checkpoint,
                config["vocab_size"],
            )
        }

    def head(n, rows):
        return {
            "logits": token_logits(
                n,
                rows("hyper", trt.float16, 10240),
                checkpoint,
                config["vocab_size"],
                config["rms_norm_eps"],
            )
        }

    for name, define in (("embed", embedding), ("head", head)):
        _write_plan(writer, name + ".plan", chunk, define)
    write_moe_plans(writer, chunk)


def write_moe_plans(writer, chunk):
    """Build the parameter-input expert and common residual combine exactly once."""

    def expert(n, rows):
        return {"result": routed_expert(n, rows("tokens", trt.float16, 2560))}

    def combine(n, rows):
        return {
            "hyper_out": combine_experts(
                n,
                rows("skip", trt.float16, 10240),
                rows("inject", trt.float16, 4),
                rows("shared", trt.float16, 2560),
                rows("weights", trt.float16, 10),
                rows("experts", trt.float16, 10, 2560),
            )
        }

    for name, define in (("expert", expert), ("combine", combine)):
        _write_plan(writer, name + ".plan", chunk, define)


def write_language_decoder(checkpoint, writer, config, *, chunk, capacity):
    """Write language stages, lookup metadata and expert banks, not a vision frontend."""
    if chunk < 1 or capacity < chunk or capacity % 4:
        raise ValueError("Flash-Next requires a positive chunk and a four-row-aligned KV capacity")
    expected = {
        "hidden_size": 2560,
        "hc_count": 4,
        "hc_lowrank": 320,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "norm_topk_prob": True,
        "output_gate_type": "sigmoid",
        "linear_conv_kernel_dim": 4,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "ple_layer_ids": [2],
    }
    for field, required in expected.items():
        if config.get(field) != required:
            raise ValueError(
                f"Unsupported Flash-Next decoder geometry: {field}={config.get(field)!r}"
            )
    rope = config["rope_parameters"]
    if not (
        rope.get("rope_type") == "default"
        and rope.get("mrope_interleaved") is True
        and rope.get("mrope_section") == [11, 11, 10]
        and rope.get("partial_rotary_factor") == 0.25
        and rope.get("rope_theta") == 10000000
    ):
        raise ValueError("Unsupported Flash-Next rotary configuration")
    kinds = config["layer_types"]
    if len(kinds) != config["num_hidden_layers"] or set(kinds) != {
        "full_attention",
        "linear_attention",
    }:
        raise ValueError("Invalid Flash-Next layer topology")
    write_shared_decoder_plans(checkpoint, writer, config, chunk)
    write_checkpoint_ngram_table(checkpoint, writer)
    for layer in range(len(kinds)):
        write_layer_plan(checkpoint, writer, config, layer, chunk, capacity)
        for expert in range(config["num_experts"]):
            write_expert(checkpoint, writer, layer, expert)
    writer.add_json(
        "decoder_layout.json",
        {
            "format": 1,
            "rope_theta": rope["rope_theta"],
            "ngram_eos": config["eos_token_id"],
            "max_chunk": chunk,
            "capacity": capacity,
            "layers": kinds,
            "ple_layer_ids": config["ple_layer_ids"],
            "vocab_size": config["vocab_size"],
        },
    )
