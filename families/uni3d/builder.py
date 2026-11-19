# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Uni3D point-encoder engine from its ONNX export.

The ONNX graph is the complete Uni3D ``encode_pc`` path: farthest point
sampling (pointnet2_ops rules, standard ONNX ops), kNN grouping, the
mini-PointNet tokenizer and the EVA02 ViT. The TensorRT ONNX parser imports it
as a strongly typed network, so the exported dtypes are kept exactly: grouping
stays fp32 so the sampled centers and neighborhoods match the reference.

Required ONNX contract:
  - one input "points" [1, num_points, 6] float32 (xyz, rgb)
  - one output "embedding" [1, embed_dim] float32
"""

from __future__ import annotations

import sys
from pathlib import Path

import tensorrt as trt

from .config import ModelConfig


_INPUT_NAME = "points"
_OUTPUT_NAME = "embedding"


def _validate_onnx_contract(onnx_path: Path, config: ModelConfig) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {_INPUT_NAME} or set(outputs) != {_OUTPUT_NAME}:
        raise ValueError(
            f"Uni3D ONNX needs input {_INPUT_NAME!r} and output {_OUTPUT_NAME!r}, "
            f"got {sorted(inputs)} -> {sorted(outputs)}"
        )
    for value in (inputs[_INPUT_NAME], outputs[_OUTPUT_NAME]):
        if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"Uni3D ONNX {value.name!r} must be float32")
    point_dims = inputs[_INPUT_NAME].type.tensor_type.shape.dim
    if len(point_dims) != 3 or point_dims[2].dim_value != config.input_dim:
        raise ValueError(f"Uni3D ONNX {_INPUT_NAME!r} must be [1, num_points, {config.input_dim}]")
    embed_dims = outputs[_OUTPUT_NAME].type.tensor_type.shape.dim
    if len(embed_dims) != 2 or embed_dims[1].dim_value != config.embed_dim:
        raise ValueError(f"Uni3D ONNX {_OUTPUT_NAME!r} must be [1, {config.embed_dim}]")


def build_uni3d_engine(model_dir: Path, config: ModelConfig, verbose: bool) -> bytes:
    """Build one Uni3D point-encoder TensorRT engine."""
    onnx_path = model_dir / "uni3d.onnx"
    _validate_onnx_contract(onnx_path, config)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("Uni3D ONNX parse failed: " + errors)

    config_trt = builder.create_builder_config()
    # Level 3 is no faster for this graph (RTX 3090, 10k points: 6.9 vs 6.7 ms) and builds slower.
    config_trt.builder_optimization_level = 1
    profile = builder.create_optimization_profile()
    # Sampling needs at least num_group points; the checkpoint was trained on num_points.
    profile.set_shape(_INPUT_NAME, (1, config.num_group, config.input_dim),
                      (1, config.num_points, config.input_dim),
                      (1, config.num_points, config.input_dim))
    config_trt.add_optimization_profile(profile)

    if verbose:
        print(
            f"[trtmc build] Building Uni3D engine (points<={config.num_points}, "
            f"groups={config.num_group}x{config.group_size}, precision={config.precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config_trt)
    if plan is None:
        raise RuntimeError("Uni3D engine build failed")
    return bytes(plan)
