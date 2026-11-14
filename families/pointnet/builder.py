# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the PointNet semantic-segmentation engine from its ONNX weight container.

The ONNX artifact is a self-contained PointNet graph (input transform, shared
MLP, global max pooling, and the semantic-segmentation head). The TensorRT
OnnxParser imports it directly; the builder validates the tensor contract and
pins the dynamic point-count optimization profile.

Required ONNX contract:
  - one input named "point" with shape [1, input_dim, num_points] (float32)
  - one output named "pred" with shape [1, num_points, num_classes] (float32)

FP16 is implemented by rewriting the ONNX container to half precision before
import, because TensorRT 11 no longer exposes a global FP16 builder flag.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import tensorrt as trt

from .config import ModelConfig


_INPUT_NAME = "point"
_OUTPUT_NAME = "pred"


def _validate_onnx_contract(onnx_path: Path, config: ModelConfig) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=True)
    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    if set(inputs) != {_INPUT_NAME} or _OUTPUT_NAME not in outputs:
        raise ValueError(
            "PointNet ONNX contract mismatch: expected one input named " +
            repr(_INPUT_NAME) + " and an output named " + repr(_OUTPUT_NAME)
        )

    def dims(value):
        return [dim.dim_value if dim.HasField("dim_value") else -1
                for dim in value.type.tensor_type.shape.dim]

    input_dims = dims(inputs[_INPUT_NAME])
    output_dims = dims(outputs[_OUTPUT_NAME])
    if len(input_dims) != 3 or input_dims[0] not in (1, -1) or input_dims[1] != config.input_dim:
        raise ValueError(
            "PointNet ONNX input " + repr(_INPUT_NAME) + " must be [1, " +
            str(config.input_dim) + ", num_points], got " + str(input_dims)
        )
    if len(output_dims) != 3 or output_dims[0] not in (1, -1) or output_dims[2] < 1:
        raise ValueError("PointNet ONNX output has unexpected shape " + str(output_dims))
    if config.num_classes != output_dims[2]:
        raise ValueError(
            "PointNet config num_classes=" + str(config.num_classes) +
            " does not match ONNX num_classes=" + str(output_dims[2])
        )


def _fp16_onnx(onnx_path: Path, dst_path: Path) -> None:
    """Rewrite the ONNX container with half-precision weights and I/O casts."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=True)
    graph = model.graph

    for init in list(graph.initializer):
        if init.data_type == TensorProto.FLOAT:
            array = numpy_helper.to_array(init).astype("float16")
            graph.initializer.remove(init)
            graph.initializer.append(numpy_helper.from_array(array, name=init.name))

    point_name = graph.input[0].name
    graph.node.insert(
        0, helper.make_node("Cast", [point_name], ["point_fp16"], to=TensorProto.FLOAT16)
    )
    for node in graph.node[1:]:
        for index in range(len(node.input)):
            if node.input[index] == point_name:
                node.input[index] = "point_fp16"

    pred_name = graph.output[0].name
    for node in graph.node:
        for index in range(len(node.output)):
            if node.output[index] == pred_name:
                node.output[index] = "pred_fp16"
    graph.node.append(
        helper.make_node("Cast", ["pred_fp16"], [pred_name], to=TensorProto.FLOAT)
    )
    graph.output[0].type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(model)
    onnx.save(model, str(dst_path))


def build_pointnet_engine(model_dir: Path, config: ModelConfig, precision: str,
                          verbose: bool) -> bytes:
    """Build one PointNet semantic-segmentation TensorRT engine."""
    onnx_path = model_dir / "pointnet.onnx"
    _validate_onnx_contract(onnx_path, config)

    parse_path = onnx_path
    tmp = None
    if precision == "fp16":
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp.close()
        _fp16_onnx(onnx_path, Path(tmp.name))
        parse_path = Path(tmp.name)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    try:
        if not parser.parse_from_file(str(parse_path)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("PointNet ONNX parse failed: " + errors)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)

    config_trt = builder.create_builder_config()
    config_trt.builder_optimization_level = 1
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape(_INPUT_NAME, (1, config.input_dim, 1),
                      (1, config.input_dim, config.num_points),
                      (1, config.input_dim, config.num_points))
    config_trt.add_optimization_profile(profile)

    if verbose:
        print(
            "[trtmc build] Building PointNet engine (points=" + str(config.num_points) +
            ", classes=" + str(config.num_classes) + ", precision=" + precision + ")",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config_trt)
    if plan is None:
        raise RuntimeError("PointNet engine build failed")
    return bytes(plan)
