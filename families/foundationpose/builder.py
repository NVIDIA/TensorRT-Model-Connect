# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT Python graphs for FoundationPose.

The NGC ONNX files are used as structurally validated weight containers. The
TensorRT networks below are authored explicitly with the TensorRT Python API;
they never pass the ONNX computation graph to TensorRT.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorrt as trt


_INPUT_HEIGHT = 160
_INPUT_WIDTH = 160
_INPUT_CHANNELS = 6
_FEATURE_HEIGHT = 20
_FEATURE_WIDTH = 20
_TOKENS = _FEATURE_HEIGHT * _FEATURE_WIDTH
_HIDDEN = 512
_HEADS = 4
_HEAD_DIM = _HIDDEN // _HEADS
_LAYER_NORM_EPS = 1.0e-5


class _OnnxWeightArchive:
    """Read exact tensors and node connections without importing the graph into TRT."""

    def __init__(self, path: str | Path, kind: str) -> None:
        import onnx
        from onnx import numpy_helper

        self.path = Path(path)
        model = onnx.load(self.path, load_external_data=True)
        expected_outputs = {"output1", "output2"} if kind == "refiner" else {"output1"}
        inputs = {value.name for value in model.graph.input}
        outputs = {value.name for value in model.graph.output}
        if inputs != {"input1", "input2"} or outputs != expected_outputs:
            raise ValueError(
                f"FoundationPose {kind} weight artifact contract mismatch: "
                f"inputs={sorted(inputs)}, outputs={sorted(outputs)}"
            )
        self.nodes = {node.name: node for node in model.graph.node}
        if len(self.nodes) != len(model.graph.node):
            raise ValueError(f"FoundationPose {kind} weight artifact has duplicate node names")
        self.arrays = {
            value.name: np.ascontiguousarray(numpy_helper.to_array(value), dtype=np.float32)
            for value in model.graph.initializer
        }

    def named(self, name: str, expected: tuple[int, ...]) -> np.ndarray:
        try:
            value = self.arrays[name]
        except KeyError as exc:
            raise KeyError(
                f"FoundationPose weight artifact {self.path} is missing tensor {name!r}"
            ) from exc
        if tuple(value.shape) != expected:
            raise ValueError(
                f"FoundationPose tensor {name!r} has shape {tuple(value.shape)}, "
                f"expected {expected}"
            )
        return value

    def node_parameter(
        self,
        node_name: str,
        input_index: int,
        expected: tuple[int, ...],
        *,
        operation: str,
    ) -> np.ndarray:
        try:
            node = self.nodes[node_name]
        except KeyError as exc:
            raise KeyError(
                f"FoundationPose weight artifact {self.path} is missing node {node_name!r}"
            ) from exc
        if node.op_type != operation:
            raise ValueError(
                f"FoundationPose node {node_name!r} is {node.op_type}, expected {operation}"
            )
        try:
            tensor_name = node.input[input_index]
        except IndexError as exc:
            raise ValueError(
                f"FoundationPose node {node_name!r} has no input {input_index}"
            ) from exc
        return self.named(tensor_name, expected)


class _FoundationPoseGraph:
    def __init__(
        self,
        trt: Any,
        network: Any,
        archive: _OnnxWeightArchive,
        kind: str,
        work_np_dtype: type[np.float16] | type[np.float32],
    ) -> None:
        self.trt = trt
        self.network = network
        self.archive = archive
        self.kind = kind
        self.work_np_dtype = work_np_dtype
        self._host_weights: list[np.ndarray] = []

    def layer(self, value: Any, kind: str, name: str) -> Any:
        if value is None:
            raise RuntimeError(f"TensorRT rejected FoundationPose {kind} layer {name!r}")
        value.name = name
        return value

    def constant(
        self,
        value: np.ndarray | float,
        name: str,
        dtype: type[np.float16] | type[np.float32] | None = None,
    ) -> Any:
        array = np.ascontiguousarray(value, dtype=dtype or self.work_np_dtype)
        self._host_weights.append(array)
        return self.layer(
            self.network.add_constant(array.shape, self.trt.Weights(array)), "constant", name
        ).get_output(0)

    def cast(self, tensor: Any, dtype: Any, name: str) -> Any:
        if tensor.dtype == dtype:
            return tensor
        return self.layer(self.network.add_cast(tensor, dtype), "cast", name).get_output(0)

    def numpy_dtype(self, tensor: Any) -> type[np.float16] | type[np.float32]:
        return np.float32 if tensor.dtype == self.trt.float32 else np.float16

    def add(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.SUM),
            "sum",
            name,
        ).get_output(0)

    def multiply(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.PROD),
            "product",
            name,
        ).get_output(0)

    def divide(self, left: Any, right: Any, name: str) -> Any:
        return self.layer(
            self.network.add_elementwise(left, right, self.trt.ElementWiseOperation.DIV),
            "division",
            name,
        ).get_output(0)

    def reshape(
        self,
        tensor: Any,
        shape: tuple[int, ...],
        name: str,
        *,
        first_transpose: tuple[int, ...] | None = None,
        second_transpose: tuple[int, ...] | None = None,
    ) -> Any:
        shuffle = self.layer(self.network.add_shuffle(tensor), "shuffle", name)
        if first_transpose is not None:
            shuffle.first_transpose = first_transpose
        shuffle.reshape_dims = shape
        if second_transpose is not None:
            shuffle.second_transpose = second_transpose
        return shuffle.get_output(0)

    def concatenate(self, tensors: Sequence[Any], axis: int, name: str) -> Any:
        concat = self.layer(self.network.add_concatenation(list(tensors)), "concatenation", name)
        concat.axis = axis
        return concat.get_output(0)

    def relu(self, tensor: Any, name: str) -> Any:
        return self.layer(
            self.network.add_activation(tensor, self.trt.ActivationType.RELU), "ReLU", name
        ).get_output(0)

    def convolution(self, tensor: Any, node_name: str, name: str) -> Any:
        weight = self.archive.node_parameter(
            node_name, 1, self._conv_shape(node_name), operation="Conv"
        )
        bias = self.archive.node_parameter(node_name, 2, (int(weight.shape[0]),), operation="Conv")
        dtype = self.numpy_dtype(tensor)
        weight = np.ascontiguousarray(weight, dtype=dtype)
        bias = np.ascontiguousarray(bias, dtype=dtype)
        self._host_weights.extend((weight, bias))
        convolution = self.layer(
            self.network.add_convolution_nd(
                tensor,
                int(weight.shape[0]),
                tuple(int(value) for value in weight.shape[2:]),
                self.trt.Weights(weight),
                self.trt.Weights(bias),
            ),
            "convolution",
            name,
        )
        stride = (
            2
            if node_name.endswith((".0/net/net.0/Conv", ".1/net/net.0/Conv", ".2/net/net.0/Conv"))
            else 1
        )
        kernel = int(weight.shape[2])
        convolution.stride_nd = (stride, stride)
        convolution.padding_nd = (kernel // 2, kernel // 2)
        return convolution.get_output(0)

    def _conv_shape(self, node_name: str) -> tuple[int, ...]:
        if "/encodeAB/" in node_name or "/encoderAB/" in node_name:
            if node_name.endswith(".2/net/net.0/Conv"):
                return (512, 256, 3, 3)
            if ".0/" in node_name or ".1/" in node_name:
                return (256, 256, 3, 3)
            return (512, 512, 3, 3)
        if node_name.endswith(".0/net/net.0/Conv"):
            return (64, 6, 7, 7)
        if node_name.endswith(".1/net/net.0/Conv"):
            return (128, 64, 3, 3)
        return (128, 128, 3, 3)

    def residual_block(self, tensor: Any, prefix: str, name: str) -> Any:
        hidden = self.convolution(tensor, f"/{prefix}/conv1/Conv", f"{name}.conv1")
        hidden = self.relu(hidden, f"{name}.relu1")
        hidden = self.convolution(hidden, f"/{prefix}/conv2/Conv", f"{name}.conv2")
        return self.relu(self.add(hidden, tensor, f"{name}.residual"), f"{name}.relu2")

    def image_encoder(self, tensor: Any, name: str) -> Any:
        prefix = "encodeA" if self.kind == "refiner" else "encoderA"
        tensor = self.reshape(
            tensor, (-1, 6, 160, 160), f"{name}.to_nchw", first_transpose=(0, 3, 1, 2)
        )
        tensor = self.convolution(tensor, f"/{prefix}/{prefix}.0/net/net.0/Conv", f"{name}.conv0")
        tensor = self.relu(tensor, f"{name}.relu0")
        tensor = self.convolution(tensor, f"/{prefix}/{prefix}.1/net/net.0/Conv", f"{name}.conv1")
        tensor = self.relu(tensor, f"{name}.relu1")
        tensor = self.residual_block(tensor, f"{prefix}/{prefix}.2", f"{name}.block0")
        return self.residual_block(tensor, f"{prefix}/{prefix}.3", f"{name}.block1")

    def paired_features(self, input1: Any, input2: Any) -> Any:
        first = self.image_encoder(input1, "input1_encoder")
        second = self.image_encoder(input2, "input2_encoder")
        tensor = self.concatenate((first, second), 1, "paired_features")
        prefix = "encodeAB" if self.kind == "refiner" else "encoderAB"
        tensor = self.residual_block(tensor, f"{prefix}/{prefix}.0", "paired.block0")
        tensor = self.residual_block(tensor, f"{prefix}/{prefix}.1", "paired.block1")
        tensor = self.convolution(
            tensor, f"/{prefix}/{prefix}.2/net/net.0/Conv", "paired.downsample"
        )
        tensor = self.relu(tensor, "paired.downsample_relu")
        tensor = self.residual_block(tensor, f"{prefix}/{prefix}.3", "paired.block2")
        tensor = self.residual_block(tensor, f"{prefix}/{prefix}.4", "paired.block3")
        tensor = self.reshape(tensor, (-1, _HIDDEN, _TOKENS), "features.flatten")
        position = self.archive.named("/pos_embed/Slice_output_0", (1, _HIDDEN, _TOKENS))
        tensor = self.add(
            tensor, self.constant(position, "position_embedding"), "features.positioned"
        )
        return self.reshape(
            tensor, (-1, _HIDDEN, _TOKENS), "features.tokens", second_transpose=(0, 2, 1)
        )

    def linear_onnx(
        self,
        tensor: Any,
        weight: np.ndarray,
        bias: np.ndarray | None,
        name: str,
    ) -> Any:
        dtype = self.numpy_dtype(tensor)
        weight = np.ascontiguousarray(weight, dtype=dtype)
        self._host_weights.append(weight)
        weight_shape = (1,) * (len(tuple(tensor.shape)) - 2) + tuple(weight.shape)
        output = self.layer(
            self.network.add_matrix_multiply(
                tensor,
                self.trt.MatrixOperation.NONE,
                self.constant(weight.reshape(weight_shape), f"{name}.weight", dtype),
                self.trt.MatrixOperation.NONE,
            ),
            "matrix multiply",
            f"{name}.matmul",
        ).get_output(0)
        if bias is None:
            return output
        bias_shape = (1,) * (len(tuple(output.shape)) - 1) + (int(bias.shape[0]),)
        return self.add(
            output, self.constant(bias.reshape(bias_shape), f"{name}.bias", dtype), name
        )

    def linear_torch(
        self,
        tensor: Any,
        weight: np.ndarray,
        bias: np.ndarray | None,
        name: str,
    ) -> Any:
        dtype = self.numpy_dtype(tensor)
        weight = np.ascontiguousarray(weight, dtype=dtype)
        self._host_weights.append(weight)
        weight_shape = (1,) * (len(tuple(tensor.shape)) - 2) + tuple(weight.shape)
        output = self.layer(
            self.network.add_matrix_multiply(
                tensor,
                self.trt.MatrixOperation.NONE,
                self.constant(weight.reshape(weight_shape), f"{name}.weight", dtype),
                self.trt.MatrixOperation.TRANSPOSE,
            ),
            "matrix multiply",
            f"{name}.matmul",
        ).get_output(0)
        if bias is None:
            return output
        bias_shape = (1,) * (len(tuple(output.shape)) - 1) + (int(bias.shape[0]),)
        return self.add(
            output, self.constant(bias.reshape(bias_shape), f"{name}.bias", dtype), name
        )

    def attention(
        self,
        tensor: Any,
        prefix: str,
        name: str,
        *,
        dynamic_tokens: bool,
        force_fp32: bool = False,
    ) -> Any:
        if force_fp32:
            tensor = self.cast(tensor, self.trt.float32, f"{name}.to_fp32")
        dtype = self.numpy_dtype(tensor)
        root = prefix.split(".", maxsplit=1)[0]
        if prefix.endswith(".self_attn"):
            module, operation = prefix.rsplit(".", maxsplit=1)
            node_name = f"/{root}/{module}/{operation}/MatMul"
        else:
            node_name = f"/{prefix}/MatMul"
        projection = self.archive.node_parameter(
            node_name, 1, (_HIDDEN, 3 * _HIDDEN), operation="MatMul"
        )
        bias = self.archive.named(f"{prefix.replace('/', '.')}.in_proj_bias", (3 * _HIDDEN,))
        query = self.linear_onnx(tensor, projection[:, :_HIDDEN], bias[:_HIDDEN], f"{name}.query")
        key = self.linear_onnx(
            tensor,
            projection[:, _HIDDEN : 2 * _HIDDEN],
            bias[_HIDDEN : 2 * _HIDDEN],
            f"{name}.key",
        )
        value = self.linear_onnx(
            tensor, projection[:, 2 * _HIDDEN :], bias[2 * _HIDDEN :], f"{name}.value"
        )
        token_dim = 0 if dynamic_tokens else _TOKENS
        query = self.reshape(
            query,
            (-1, token_dim, _HEADS, _HEAD_DIM),
            f"{name}.query_heads",
            second_transpose=(0, 2, 1, 3),
        )
        key = self.reshape(
            key,
            (-1, token_dim, _HEADS, _HEAD_DIM),
            f"{name}.key_heads",
            second_transpose=(0, 2, 3, 1),
        )
        value = self.reshape(
            value,
            (-1, token_dim, _HEADS, _HEAD_DIM),
            f"{name}.value_heads",
            second_transpose=(0, 2, 1, 3),
        )
        if self.kind == "refiner":
            split_scale = np.array([[[[float(_HEAD_DIM) ** -0.25]]]], dtype=np.float32)
            query = self.multiply(
                query,
                self.constant(split_scale, f"{name}.query_scale", dtype),
                f"{name}.scaled_query",
            )
            key = self.multiply(
                key,
                self.constant(split_scale, f"{name}.key_scale", dtype),
                f"{name}.scaled_key",
            )
        else:
            scale = np.array([[[[math.sqrt(_HEAD_DIM)]]]], dtype=np.float32)
            query = self.divide(
                query,
                self.constant(scale, f"{name}.scale", dtype),
                f"{name}.scaled_query",
            )
        scores = self.layer(
            self.network.add_matrix_multiply(
                query,
                self.trt.MatrixOperation.NONE,
                key,
                self.trt.MatrixOperation.NONE,
            ),
            "attention scores",
            f"{name}.scores",
        ).get_output(0)
        softmax = self.layer(self.network.add_softmax(scores), "softmax", f"{name}.softmax")
        softmax.axes = 1 << 3
        context = self.layer(
            self.network.add_matrix_multiply(
                softmax.get_output(0),
                self.trt.MatrixOperation.NONE,
                value,
                self.trt.MatrixOperation.NONE,
            ),
            "attention context",
            f"{name}.context_heads",
        ).get_output(0)
        context = self.reshape(
            context,
            (-1, token_dim, _HIDDEN),
            f"{name}.context",
            first_transpose=(0, 2, 1, 3),
        )
        out_weight = self.archive.named(
            f"{prefix.replace('/', '.')}.out_proj.weight", (_HIDDEN, _HIDDEN)
        )
        out_bias = self.archive.named(f"{prefix.replace('/', '.')}.out_proj.bias", (_HIDDEN,))
        return self.linear_torch(context, out_weight, out_bias, f"{name}.output")

    def layer_norm(self, tensor: Any, prefix: str, name: str) -> Any:
        scale = self.archive.named(f"{prefix}.weight", (_HIDDEN,)).reshape(1, 1, _HIDDEN)
        bias = self.archive.named(f"{prefix}.bias", (_HIDDEN,)).reshape(1, 1, _HIDDEN)
        output_dtype = tensor.dtype
        tensor = self.cast(tensor, self.trt.float32, f"{name}.to_float32")
        normalization = self.layer(
            self.network.add_normalization_v2(
                tensor,
                self.constant(scale, f"{name}.scale", np.float32),
                self.constant(bias, f"{name}.bias", np.float32),
                1 << 2,
            ),
            "layer normalization",
            name,
        )
        normalization.epsilon = _LAYER_NORM_EPS
        return self.cast(normalization.get_output(0), output_dtype, f"{name}.to_output_dtype")

    def transformer_head(self, tokens: Any, prefix: str, name: str) -> Any:
        attention = self.attention(
            tokens, f"{prefix}.0.self_attn", f"{name}.attention", dynamic_tokens=False
        )
        hidden = self.layer_norm(
            self.add(tokens, attention, f"{name}.attention_residual"),
            f"{prefix}.0.norm1",
            f"{name}.norm1",
        )
        linear1_node = f"/{prefix.replace('.', '/')}/{prefix.split('.')[0]}.0/linear1/MatMul"
        linear2_node = f"/{prefix.replace('.', '/')}/{prefix.split('.')[0]}.0/linear2/MatMul"
        head_node = f"/{prefix.replace('.', '/')}/{prefix.split('.')[0]}.1/MatMul"
        feed_forward = self.linear_onnx(
            hidden,
            self.archive.node_parameter(linear1_node, 1, (_HIDDEN, _HIDDEN), operation="MatMul"),
            self.archive.named(f"{prefix}.0.linear1.bias", (_HIDDEN,)),
            f"{name}.linear1",
        )
        feed_forward = self.relu(feed_forward, f"{name}.relu")
        feed_forward = self.linear_onnx(
            feed_forward,
            self.archive.node_parameter(linear2_node, 1, (_HIDDEN, _HIDDEN), operation="MatMul"),
            self.archive.named(f"{prefix}.0.linear2.bias", (_HIDDEN,)),
            f"{name}.linear2",
        )
        hidden = self.layer_norm(
            self.add(hidden, feed_forward, f"{name}.feed_forward_residual"),
            f"{prefix}.0.norm2",
            f"{name}.norm2",
        )
        output = self.linear_onnx(
            hidden,
            self.archive.node_parameter(head_node, 1, (_HIDDEN, 3), operation="MatMul"),
            self.archive.named(f"{prefix}.1.bias", (3,)),
            f"{name}.output",
        )
        return self.layer(
            self.network.add_reduce(output, self.trt.ReduceOperation.AVG, 1 << 1, keep_dims=False),
            "token mean",
            f"{name}.mean",
        ).get_output(0)

    def build_refiner(self, input1: Any, input2: Any) -> tuple[Any, Any]:
        tokens = self.paired_features(input1, input2)
        translation = self.transformer_head(tokens, "trans_head", "translation_head")
        rotation = self.transformer_head(tokens, "rot_head", "rotation_head")
        return translation, rotation

    def build_scorer(self, input1: Any, input2: Any) -> Any:
        tokens = self.paired_features(input1, input2)
        tokens = self.attention(tokens, "att", "candidate_attention", dynamic_tokens=False)
        pooled = self.layer(
            self.network.add_reduce(tokens, self.trt.ReduceOperation.AVG, 1 << 1, keep_dims=False),
            "token mean",
            "candidate_attention.mean",
        ).get_output(0)
        candidates = self.reshape(pooled, (1, -1, _HIDDEN), "candidates.to_sequence")
        # Cross-hypothesis attention directly controls the score ordering and is
        # more numerically sensitive than the per-candidate feature extractor.
        candidates = self.attention(
            candidates,
            "att_cross",
            "candidate_cross_attention",
            dynamic_tokens=True,
            force_fp32=self.work_np_dtype is np.float16,
        )
        logits = self.linear_onnx(
            candidates,
            self.archive.node_parameter("/linear/MatMul", 1, (_HIDDEN, 1), operation="MatMul"),
            self.archive.named("linear.bias", (1,)),
            "score",
        )
        return self.reshape(logits, (1, -1), "score.output")


def build_foundationpose_engine(
    path: str,
    *,
    kind: str,
    max_batch: int,
    precision: str,
    verbose: bool = False,
) -> bytes:
    """Build a FoundationPose engine from family-owned TensorRT layers."""
    if precision not in {"fp16", "fp32"}:
        raise ValueError("FoundationPose engines support fp16 or fp32 builds")
    if kind not in {"refiner", "scorer"}:
        raise ValueError(f"Unsupported FoundationPose engine kind: {kind!r}")
    if max_batch <= 0:
        raise ValueError("FoundationPose max_batch must be positive")

    work_np_dtype = np.float16 if precision == "fp16" else np.float32
    work_trt_dtype = trt.float16 if precision == "fp16" else trt.float32
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    archive = _OnnxWeightArchive(path, kind)
    graph = _FoundationPoseGraph(trt, network, archive, kind, work_np_dtype)
    input_shape = (-1, _INPUT_HEIGHT, _INPUT_WIDTH, _INPUT_CHANNELS)
    input1 = network.add_input("input1", trt.float32, input_shape)
    input2 = network.add_input("input2", trt.float32, input_shape)
    if input1 is None or input2 is None:
        raise RuntimeError("TensorRT rejected the FoundationPose input contract")
    input1 = graph.cast(input1, work_trt_dtype, "input1.to_working_precision")
    input2 = graph.cast(input2, work_trt_dtype, "input2.to_working_precision")
    if kind == "refiner":
        translation, rotation = graph.build_refiner(input1, input2)
        translation = graph.cast(translation, trt.float32, "output1.to_fp32")
        rotation = graph.cast(rotation, trt.float32, "output2.to_fp32")
        translation.name = "output1"
        rotation.name = "output2"
        network.mark_output(translation)
        network.mark_output(rotation)
    else:
        score = graph.build_scorer(input1, input2)
        score = graph.cast(score, trt.float32, "output1.to_fp32")
        score.name = "output1"
        network.mark_output(score)

    profile = builder.create_optimization_profile()
    opt_batch = min(8, max_batch)
    for name in ("input1", "input2"):
        profile.set_shape(
            name,
            (1, _INPUT_HEIGHT, _INPUT_WIDTH, _INPUT_CHANNELS),
            (opt_batch, _INPUT_HEIGHT, _INPUT_WIDTH, _INPUT_CHANNELS),
            (max_batch, _INPUT_HEIGHT, _INPUT_WIDTH, _INPUT_CHANNELS),
        )
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.builder_optimization_level = 4
    config.max_aux_streams = 0
    if verbose:
        print(
            f"[trtmc build] Building native FoundationPose {kind} graph "
            f"(batch=1..{max_batch}, input=160x160x6, precision={precision}) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build the FoundationPose {kind} engine")
    return bytes(plan)
