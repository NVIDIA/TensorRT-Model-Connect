# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the small dynamic interaction-pair classifier as its own TRT plan."""

from __future__ import annotations

import sys
from collections.abc import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat


INTERACTION_SECTION = "sam2_hoi_interaction_engine_plan"
INTERACTION_INPUT_WIDTH = 512
INTERACTION_HIDDEN_WIDTH = 256
INTERACTION_CLASSES = 2
INTERACTION_MAX_PAIRS = 22_500
_PREFIX = "image_encoder.hoi_head.query_head.interaction_head.mlp"


def _parameter(weights: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    value = np.asarray(weights[f"{_PREFIX}.{name}"], dtype=np.float32)
    return np.ascontiguousarray(value)


def _add_constant(network, trt, values: np.ndarray):
    """Add a contiguous FP32 constant through TensorRT's explicit Weights API."""

    array = np.ascontiguousarray(values, dtype=np.float32)
    return network.add_constant(array.shape, trt.Weights(array)).get_output(0)


def interaction_mlp_numpy(
    pair_features: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Framework-independent oracle for the exact three-layer source MLP."""

    x = np.asarray(pair_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INTERACTION_INPUT_WIDTH:
        raise ValueError(f"SAM2 HOI pair features must have shape [pairs, 512], got {x.shape}")
    for index in (0, 2, 4):
        weight = _parameter(weights, f"{index}.weight")
        bias = _parameter(weights, f"{index}.bias")
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"Invalid SAM2 HOI interaction layer {index} shapes")
        if x.shape[1] != weight.shape[1]:
            raise ValueError(
                f"SAM2 HOI interaction layer {index} expects {weight.shape[1]} "
                f"features, got {x.shape[1]}"
            )
        x = x @ weight.T + bias
        if index != 4:
            x = np.maximum(x, np.float32(0.0))
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return np.ascontiguousarray(exp / exp.sum(axis=1, keepdims=True), dtype=np.float32)


def build_interaction_engine(
    weights: Mapping[str, np.ndarray],
    *,
    precision: str = "fp32",
    verbose: bool = False,
) -> bytes:
    """Build a dynamic-pair plan; NMS and pair construction stay in the runtime."""

    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"SAM2 HOI interaction precision must be fp32 or bf16, got {precision}")
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(explicit_batch=True, strongly_typed=True)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    inp = network.add_input("pair_features", trt.float32, (-1, INTERACTION_INPUT_WIDTH))
    if inp is None:
        raise RuntimeError("TensorRT failed to create the SAM2 HOI interaction input")
    learned_dtype = trt.float32
    current = inp
    if precision == "bf16":
        learned_dtype = trt.bfloat16
        current = network.add_cast(current, learned_dtype).get_output(0)

    for index in (0, 2, 4):
        weight = np.ascontiguousarray(
            _parameter(weights, f"{index}.weight").T,
            dtype=np.float32,
        )
        bias = _parameter(weights, f"{index}.bias")
        constant = _add_constant(network, trt, weight)
        if precision == "bf16":
            constant = network.add_cast(constant, learned_dtype).get_output(0)
        current = network.add_matrix_multiply(
            current,
            trt.MatrixOperation.NONE,
            constant,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        bias_tensor = _add_constant(network, trt, bias.reshape(1, -1))
        if precision == "bf16":
            bias_tensor = network.add_cast(bias_tensor, learned_dtype).get_output(0)
        current = network.add_elementwise(
            current, bias_tensor, trt.ElementWiseOperation.SUM
        ).get_output(0)
        if index != 4:
            current = network.add_activation(current, trt.ActivationType.RELU).get_output(0)

    if precision == "bf16":
        current = network.add_cast(current, trt.float32).get_output(0)
    softmax = network.add_softmax(current)
    softmax.axes = 1 << 1
    probabilities = softmax.get_output(0)
    probabilities.name = "interaction_probabilities"
    network.mark_output(probabilities)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pair_features",
        min=(1, INTERACTION_INPUT_WIDTH),
        opt=(8, INTERACTION_INPUT_WIDTH),
        max=(INTERACTION_MAX_PAIRS, INTERACTION_INPUT_WIDTH),
    )
    config.add_optimization_profile(profile)
    if verbose:
        print(
            "[trtmc build] Building SAM2 HOI dynamic interaction plan ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT SAM2 HOI interaction engine build failed")
    return bytes(plan)
