# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiLiCorr's accepted-token projector and full-vocabulary proposal support."""

import numpy as np

from .native_language import Graph
from .puzzle_moe import q8_linear


def build_projection(weights, config, *, logger):
    import tensorrt as trt

    b = trt.Builder(logger)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(net, weights)
    h = config["hidden_size"]
    layers = config["dflash_config"]["target_layer_ids"]
    x = net.add_input("target_features", trt.bfloat16, (1, h * len(layers)))
    # LiLiCorr's W4A16 FC scales the FP16 input by 1/64 before projection. Rescale
    # the result in FP32 *before* normalization, not after it overflows in FP16.
    x = graph.ew(graph.cast(x, trt.float16), graph.const(np.array([[1 / 64]], np.float16)), "PROD")
    z = graph.ew(graph.f(graph.w4a16(x, "fc")), graph.scalar(64, 2), "PROD")
    var = graph.reduce(graph.ew(z, z, "PROD"), "AVG", 1)
    root = net.add_unary(
        graph.ew(var, graph.scalar(config["rms_norm_eps"], 2), "SUM"), trt.UnaryOperation.SQRT
    ).get_output(0)
    w = weights.f32("hidden_norm.weight").astype(np.float16).astype(np.float32)[None]
    delta = graph.ew(graph.ew(z, root, "DIV"), graph.const(w), "PROD")
    graph.output(graph.cast(delta, trt.float16), "accepted_hidden")
    config_builder = b.create_builder_config()
    config_builder.builder_optimization_level = 3
    plan = b.build_serialized_network(net, config_builder)
    if plan is None:
        raise RuntimeError("Cannot build native Bolt LiLiCorr target features")
    return plan


def build_vocabulary(target_weights, config, *, logger):
    import tensorrt as trt

    b = trt.Builder(logger)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(net, target_weights)
    h = config["hidden_size"]
    x = net.add_input("prediction_hidden", trt.float16, (1, h))
    # Unlike the target's FP32 terminal logits, Q8_0 writes the draft input
    # type (FP16). The released lattice then computes log probabilities in FP32.
    logits = graph.f(q8_linear(net, x, target_weights.tensor("language_model.lm_head.weight")))
    candidates = net.add_topk(
        logits, trt.TopKOperation.MAX, config["dflash_config"]["lilicorr_candidate_topk"], 2
    )
    peak = graph.reduce(logits, "MAX", 1)
    centered = graph.ew(logits, peak, "SUB")
    exp = net.add_unary(centered, trt.UnaryOperation.EXP).get_output(0)
    logz = net.add_unary(graph.reduce(exp, "SUM", 1), trt.UnaryOperation.LOG).get_output(0)
    values = graph.ew(graph.ew(candidates.get_output(0), peak, "SUB"), logz, "SUB")
    graph.output(candidates.get_output(1), "candidate_ids")
    graph.output(values, "candidate_log_probs")
    config_builder = b.create_builder_config()
    config_builder.builder_optimization_level = 3
    plan = b.build_serialized_network(net, config_builder)
    if plan is None:
        raise RuntimeError("Cannot build native Bolt LiLiCorr vocabulary proposals")
    return plan
