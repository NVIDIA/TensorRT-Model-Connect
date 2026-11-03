# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT selection/projection of the eight precomposed PDD heads.

This component does not implement the student's transformer or VSA operation.
Its explicit BF16 operands, FP32 accumulator and BF16 output rounding describe
the released linear epilogue. TensorRT owns fusion and GPU execution.
"""

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph_ops as op
from .reference_pdd import HEAD_FEATURES


def validate_head_banks(banks):
    if not isinstance(banks, dict) or set(banks) != set(HEAD_FEATURES):
        raise ValueError("PDD projection requires both modality head banks")
    hidden = None
    for modality, width in HEAD_FEATURES.items():
        item = banks[modality]
        if not isinstance(item, dict) or set(item) != {"weight", "bias"}:
            raise ValueError("PDD bank requires explicit weight and bias arrays")
        weight, bias = np.asarray(item["weight"]), np.asarray(item["bias"])
        if (
            weight.ndim != 3
            or weight.shape[:2] != (8, width)
            or weight.shape[2] <= 0
            or bias.shape != (8, width)
        ):
            raise ValueError("PDD bank geometry differs from eight trained output heads")
        if hidden is not None and hidden != weight.shape[2]:
            raise ValueError("PDD video/audio heads must share the transformer hidden width")
        hidden = weight.shape[2]
        for parameter in (weight, bias):
            if parameter.dtype != np.float32 or not np.isfinite(parameter).all():
                raise ValueError("PDD bank must contain finite decoded BF16 parameters")
            raw = np.ascontiguousarray(parameter).view(np.uint32)
            if np.any(raw & np.uint32(0xFFFF)):
                raise ValueError("PDD bank parameters must retain their BF16 rounding")
    return hidden


def add_output_heads(network, hidden_states, step, banks):
    """Add both heads after gathering only the requested target hidden rows.

    The family runtime must supply a validated step in 0..7. This selector is an
    internal operand, not a caller-controlled sampling or precision option.
    """
    width = validate_head_banks(banks)
    if not isinstance(hidden_states, dict) or set(hidden_states) != set(HEAD_FEATURES):
        raise ValueError("PDD heads require both target hidden-state operands")
    if step.dtype != trt.int32 or tuple(step.shape) != (1,):
        raise ValueError("PDD head selector must be one INT32 internal step")
    result = {}
    for modality, features in HEAD_FEATURES.items():
        hidden = hidden_states[modality]
        if (
            len(tuple(hidden.shape)) != 2
            or hidden.shape[-1] != width
            or hidden.dtype not in (trt.float32, trt.bfloat16)
        ):
            raise ValueError("PDD target hidden states have the wrong shape or dtype")
        # Store compact BF16 constants; widen only selected operands for the
        # linear's FP32 accumulator, including bias before output rounding.
        bank = op.weight_constant(
            network, np.asarray(banks[modality]["weight"], dtype=ml_dtypes.bfloat16)
        )
        selected = network.add_gather(bank, step, 0).get_output(0)
        reshape = network.add_shuffle(selected)
        reshape.reshape_dims = (features, width)
        weight = op.cast(network, reshape.get_output(0), trt.float32)
        hidden = op.cast(network, op.cast(network, hidden, trt.bfloat16), trt.float32)
        projected = network.add_matrix_multiply(
            hidden, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        bias_bank = op.weight_constant(
            network, np.asarray(banks[modality]["bias"], dtype=ml_dtypes.bfloat16)
        )
        bias = op.cast(network, network.add_gather(bias_bank, step, 0).get_output(0), trt.float32)
        projected = network.add_elementwise(
            projected, bias, trt.ElementWiseOperation.SUM
        ).get_output(0)
        velocity = op.cast(network, op.cast(network, projected, trt.bfloat16), trt.float32)
        velocity.name = modality + "_velocity"
        result[modality] = velocity
    return result


def _rows(value):
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(n) is not int or n <= 0 for n in value)
        or not value[0] <= value[1] <= value[2]
    ):
        raise ValueError("PDD row profile requires three ordered positive integer bounds")
    return value


def build_projection_engine(
    banks, *, video_rows=(1, 17, 97), audio_rows=(1, 5, 13), workspace_bytes=256 << 20
):
    """Serialize a standalone component plan, not a complete student bundle."""
    width = validate_head_banks(banks)
    video_rows, audio_rows = _rows(video_rows), _rows(audio_rows)
    if type(workspace_bytes) is not int or workspace_bytes <= 0:
        raise ValueError("PDD projection workspace must be a positive integer")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    try:
        config = builder.create_builder_config()
        config.builder_optimization_level = 3
        config.clear_flag(trt.BuilderFlag.TF32)
        op.configure_builder(config)
        op.configure_workspace(config, workspace_bytes, default_bytes=256 << 20)
        profile = builder.create_optimization_profile()
        inputs = {}
        for modality, bounds in [("video", video_rows), ("audio", audio_rows)]:
            name = modality + "_hidden_states"
            inputs[modality] = network.add_input(name, trt.float32, (-1, width))
            expected = [(n, width) for n in bounds]
            profile.set_shape(name, *expected)
            if [tuple(s) for s in profile.get_shape(name)] != expected:
                raise RuntimeError("TensorRT did not retain the PDD row profile")
        selector = network.add_input("pdd_step", trt.int32, (1,))
        profile.set_shape("pdd_step", (1,), (1,), (1,))
        config.add_optimization_profile(profile)
        for output in add_output_heads(network, inputs, selector, banks).values():
            network.mark_output(output)
        op.validate_native_network(network, expected_attentions=0, label="PDD projection")
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT failed to serialize the PDD projection component")
        return bytes(plan)
    finally:
        op.release_weight_buffers(network)
