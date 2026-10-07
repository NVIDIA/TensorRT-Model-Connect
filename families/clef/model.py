# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef TensorRT topology: full-sequence backbone and joint schema decisions."""

import json
import math
from pathlib import Path

import tensorrt as trt

from .graph import Graph


def joint_head(network, weights, config, *, dtype=trt.bfloat16):
    """Vectorize ragged questions while retaining every head operation.

    Pool masks average exactly the source token spans. option_fields maps each
    option to its owning question. The masks do not encode model predictions.
    """
    g = Graph(network, weights)
    g.precise_norm = True
    hidden = config["hidden_size"]
    width = config["width"]
    heads = config["heads"]
    states = network.add_input("hidden_states", dtype, (-1, hidden))
    lexical = network.add_input("lexical_options", dtype, (-1, hidden))
    q_pool = network.add_input("question_pool", trt.float32, (-1, -1))
    o_pool = network.add_input("option_pool", trt.float32, (-1, -1))
    option_fields = network.add_input("option_fields", trt.int32, (-1,))
    type_ids = network.add_input("type_ids", trt.int32, (-1,))
    # Zero for options belonging to a question, negative infinity otherwise.
    group_mask = network.add_input("group_mask", dtype, (-1, -1))
    last_index = network.add_input("last_index", trt.int32, (1,))
    seq = g.norm(states, "hidden_norm")
    memory = g.linear(seq, "memory_projection")
    global_vector = network.add_gather(seq, last_index, 0).get_output(0)
    questions = g.mean_pool(q_pool, seq)
    contexts = g.mean_pool(o_pool, seq)
    option_questions = network.add_gather(questions, option_fields, 0).get_output(0)
    routed = g.add(
        g.add(
            g.linear(contexts, "option_context_projection"),
            g.linear(lexical, "option_lexical_projection"),
        ),
        g.linear(option_questions, "option_question_projection"),
    )
    for i in range(config["routing_layers"]):
        p = f"evidence_layers.{i}"
        routed = g.add(
            routed,
            g.mha(
                g.norm(routed, p + ".query_norm"),
                g.norm(memory, p + ".memory_norm"),
                p + ".attention",
                heads,
            ),
        )
        routed = g.add(
            routed,
            g.linear(
                g.gelu(g.linear(g.norm(routed, p + ".feedforward_norm"), p + ".feedforward.0")),
                p + ".feedforward.3",
            ),
        )
    base = g.linear(questions, "question_projection")
    routing_logits = g.div(g.mm(base, routed, True), math.sqrt(width))
    routing_weights = g.softmax(g.add(routing_logits, group_mask))
    # Torch performs the weighted product in BF16 before the sum reduction.
    # A BF16 matmul would omit that rounding boundary.
    product = g.mul(g.reshape(routing_weights, (0, -1, 1)), g.reshape(routed, (1, -1, width)))
    summary = g.cast(g.reduce(g.cast(product, trt.float32), axis=1, keep=False), dtype)
    types = network.add_gather(
        g.const(weights["type_embedding.weight"], dtype), type_ids, 0
    ).get_output(0)
    fields = g.add(
        g.add(
            g.add(base, g.norm(summary, "option_summary_norm")),
            g.linear(global_vector, "global_projection"),
        ),
        types,
    )
    for i in range(config["layers"]):
        p = f"layers.{i}"
        n = g.norm(fields, p + ".norm1")
        fields = g.add(fields, g.mha(n, n, p + ".self_attn", heads))
        fields = g.add(
            fields, g.mha(g.norm(fields, p + ".norm2"), memory, p + ".multihead_attn", heads)
        )
        fields = g.add(
            fields,
            g.linear(
                g.gelu(g.linear(g.norm(fields, p + ".norm3"), p + ".linear1")), p + ".linear2"
            ),
        )
    fields = g.norm(fields, "field_norm")
    repeated = network.add_gather(fields, option_fields, 0).get_output(0)
    anchor = g.unit(g.add(option_questions, global_vector), 1e-12)
    lexical_anchor = g.unit(lexical, 1e-12)
    prior = g.reshape(
        g.mm(g.reshape(lexical_anchor, (-1, 1, hidden)), g.reshape(anchor, (-1, hidden, 1))),
        (-1, 1),
    )
    options = g.norm(routed, "option_norm")
    cosine = g.cast(
        g.reduce(g.cast(g.mul(g.unit(repeated, 1e-8), g.unit(options, 1e-8)), trt.float32)), dtype
    )
    features = g.concat(
        [repeated, options, g.mul(repeated, options), g.abs(g.sub(repeated, options))]
    )
    residual = g.linear(g.gelu(g.linear(features, "residual_scorer.0")), "residual_scorer.3")
    # Scales are evaluated in the head's BF16 dtype by the original module.
    import torch

    def scale(name):
        return weights[name].to(torch.bfloat16).clamp(max=math.log(100.0)).exp().float().item()

    gate = weights["residual_gate"].to(torch.bfloat16).sigmoid().float().item()
    logits = g.add(
        g.mul(prior, scale("prior_logit_scale")),
        g.mul(g.add(g.mul(cosine, scale("joint_logit_scale")), residual), gate),
    )
    output = g.reshape(g.cast(logits, trt.float32), (-1,))
    output.name = "logits"
    network.mark_output(output)
    return g


def build_head(weights, config, *, max_sequence_length=16384, max_questions=64, max_options=512):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    _graph = joint_head(network, weights, config)  # Retain constant weight storage through build.
    profile = builder.create_optimization_profile()
    dimensions = {
        "hidden_states": (
            (1, config["hidden_size"]),
            (512, config["hidden_size"]),
            (max_sequence_length, config["hidden_size"]),
        ),
        "lexical_options": (
            (1, config["hidden_size"]),
            (8, config["hidden_size"]),
            (max_options, config["hidden_size"]),
        ),
        "question_pool": ((1, 1), (3, 512), (max_questions, max_sequence_length)),
        "option_pool": ((1, 1), (8, 512), (max_options, max_sequence_length)),
        "option_fields": ((1,), (8,), (max_options,)),
        "type_ids": ((1,), (3,), (max_questions,)),
        "group_mask": ((1, 1), (3, 8), (max_questions, max_options)),
        "last_index": ((1,), (1,), (1,)),
    }
    for name, shapes in dimensions.items():
        profile.set_shape(name, *shapes)
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.add_optimization_profile(profile)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Clef joint head build failed")
    return bytes(plan)


def build_decoder_layer(
    weights, config, index, *, max_sequence_length=16384, debug=False, timing_cache=None
):
    from .backbone import decoder_layer

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    _graph = decoder_layer(network, weights, config, index, debug=debug)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        tail = tuple(tensor.shape)[1:]
        profile.set_shape(tensor.name, (64, *tail), (512, *tail), (max_sequence_length, *tail))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.add_optimization_profile(profile)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    settings.clear_flag(trt.BuilderFlag.TF32)
    if timing_cache is not None:
        settings.set_timing_cache(settings.create_timing_cache(bytes(timing_cache)), False)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError(f"Clef decoder layer {index} build failed")
    if timing_cache is not None:
        timing_cache[:] = bytes(settings.get_timing_cache().serialize())
    return bytes(plan)


def build_final_norm(weight, config, max_sequence_length):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    g = Graph(network, {"norm.weight": weight})
    x = network.add_input("hidden_states", trt.bfloat16, (-1, config["hidden_size"]))
    y = g.rms(x, "norm", config["rms_norm_eps"])
    y.name = "output"
    network.mark_output(y)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "hidden_states",
        (1, config["hidden_size"]),
        (512, config["hidden_size"]),
        (max_sequence_length, config["hidden_size"]),
    )
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.add_optimization_profile(profile)
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Clef final normalization build failed")
    return bytes(plan)


def build_backbone(checkpoint, config, max_sequence_length):
    """One execution context shares workspace across all decoder blocks."""
    from .backbone import decoder_block

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, {})
    hidden = config["hidden_size"]
    rotary = int(config["head_dim"] * config["rope_parameters"]["partial_rotary_factor"])
    x = network.add_input("hidden_states", trt.bfloat16, (-1, hidden))
    cos = network.add_input("rope_cos", trt.bfloat16, (-1, 1, rotary))
    sin = network.add_input("rope_sin", trt.bfloat16, (-1, 1, rotary))
    for index in range(config["num_hidden_layers"]):
        print(f"Adding Clef backbone block {index + 1}/{config['num_hidden_layers']}", flush=True)
        graph.weights = checkpoint.select(f"model.language_model.layers.{index}.")
        x = decoder_block(graph, x, config, index, cos, sin)
        # Materialize the BF16 residual at each block boundary, as the original
        # module does. These device outputs are not copied to the host by the
        # family pipeline; all blocks still share one execution workspace.
        x.name = f"block_{index}"
        network.mark_output(x)
    x.name = "output"
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        tail = tuple(tensor.shape)[1:]
        profile.set_shape(tensor.name, (64, *tail), (512, *tail), (max_sequence_length, *tail))
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.add_optimization_profile(profile)
    settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    settings.clear_flag(trt.BuilderFlag.TF32)
    print("Compiling Clef backbone", flush=True)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Clef backbone build failed")
    return bytes(plan)


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.mapping = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    def select(self, prefix):
        from safetensors import safe_open

        result = {}
        names = [name for name in self.mapping if name.startswith(prefix)]
        for shard in sorted({self.mapping[name] for name in names}):
            with safe_open(self.root / shard, framework="pt") as reader:
                for name in names:
                    if self.mapping[name] == shard:
                        result[name] = reader.get_tensor(name).bfloat16()
        if not result:
            raise ValueError(f"checkpoint is missing {prefix}")
        return result


def build(request, writer):
    """Build independent TensorRT stages and native runtime data into one bundle."""
    from safetensors.torch import load_file
    from .cli import BuildRequest

    if not isinstance(request, BuildRequest):
        raise TypeError("Clef requires its family-owned BuildRequest")
    root = Path(request.model_dir)
    config = json.loads((root / "config.json").read_text())
    head_config = json.loads((root / "joint_head_config.json").read_text())
    text = config["text_config"]
    if (
        config["model_type"] != "qwen3_5"
        or text["hidden_size"] != head_config["hidden_size"]
        or text["linear_key_head_dim"] != text["linear_value_head_dim"]
    ):
        raise ValueError("incompatible Clef backbone and joint head")
    checkpoint = Checkpoint(root)
    writer.set_header(family="clef", task=request.task, backend=request.backend)
    writer.add_json(
        "runtime.json",
        {
            "text_config": text,
            "head_config": head_config,
            "max_sequence_length": request.max_sequence_length,
            "max_questions": request.max_questions,
            "max_options": request.max_options,
            "precision": request.precision,
            "backbone_sequence_alignment": 1,
            "vision_config": config["vision_config"],
            "processor_config": json.loads((root / "processor_config.json").read_text()),
        },
    )
    writer.add_bytes("tokenizer.json", (root / "tokenizer.json").read_bytes())
    for section, name in (
        ("embedding.bin", "model.language_model.embed_tokens.weight"),
        ("lexical_embedding.bin", "lm_head.weight"),
    ):
        import torch

        weight = checkpoint.select(name)[name].contiguous()
        with writer.open_section(section) as out:
            out.write(weight.view(torch.int16).numpy().tobytes())
        del weight
    writer.add_bytes("backbone.plan", build_backbone(checkpoint, text, request.max_sequence_length))
    weight = checkpoint.select("model.language_model.norm.weight")[
        "model.language_model.norm.weight"
    ]
    writer.add_bytes("norm.plan", build_final_norm(weight, text, request.max_sequence_length))
    writer.add_bytes(
        "head.plan",
        build_head(
            load_file(root / "joint_head.safetensors"),
            head_config,
            max_sequence_length=request.max_sequence_length,
            max_questions=request.max_questions,
            max_options=request.max_options,
        ),
    )
    from .vision import build_vision

    vision_weights = checkpoint.select("model.visual.")
    writer.add_bytes(
        "vision.plan",
        build_vision(vision_weights, config["vision_config"], request.max_sequence_length * 4),
    )
    import torch

    writer.add_bytes(
        "vision_positions.bin",
        vision_weights["model.visual.pos_embed.weight"]
        .contiguous()
        .view(torch.int16)
        .numpy()
        .tobytes(),
    )
