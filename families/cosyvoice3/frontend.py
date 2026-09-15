# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native, batch-one CosyVoice3 reference-audio networks.

ONNX is a build-time weight container only. The two topologies below are
explicit family-owned graphs, not an ONNX parser or an inference fallback.
The exact published files are pinned because their folded weights have
exporter-generated names. Runtime imports neither onnx nor onnxruntime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .components import ComponentEngine, Graph


CHECKPOINTS = {
    "campplus": "campplus.onnx",
    "speech_tokenizer": "speech_tokenizer_v3.onnx",
}


@dataclass(frozen=True)
class FrontendProfile:
    min_frames: int = 4
    opt_frames: int = 500
    max_frames: int = 3000

    def __post_init__(self):
        values = (self.min_frames, self.opt_frames, self.max_frames)
        if any(type(v) is not int for v in values) or not 4 <= values[0] <= values[1] <= values[2] <= 3000:
            raise ValueError("Frontend requires 4 <= min <= opt <= max <= 3000 feature frames")


class FrontendWeights:
    """Resolve explicitly named learned layers in the pinned exported checkpoint."""

    def __init__(self, model_dir, component):
        import onnx
        from onnx import numpy_helper

        filename = CHECKPOINTS[component]
        path = Path(model_dir) / filename
        model = onnx.load(path, load_external_data=False)
        self.nodes = {n.name: n for n in model.graph.node}
        self.arrays = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
        self.initializers = set(self.arrays)
        self.used = set()
        for node in model.graph.node:
            if node.op_type == "Constant":
                self.arrays[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)

    def array(self, name):
        self.used.add(name)
        value = self.arrays[name]
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Expected finite FP32 weights: {name}")
        return np.ascontiguousarray(value)

    def parameters(self, name, kind):
        node = self.nodes[name]
        if node.op_type != kind:
            raise ValueError(f"Unexpected checkpoint layer: {name}")
        return [self.array(n) for n in node.input if n in self.arrays]

    def check_consumed(self):
        if missing := self.initializers - self.used:
            raise ValueError(f"Unmapped frontend checkpoint tensors: {sorted(missing)}")


class FrontendGraph(Graph):
    def reduce(self, x, axis, op="AVG", keep=True):
        return self.net.add_reduce(x, getattr(self.trt.ReduceOperation, op), 1 << axis, keep).get_output(0)

    def crop(self, x, size):
        rank = len(x.shape)
        layer = self.net.add_slice(x, (0,) * rank, (1,) * rank, (1,) * rank)
        layer.set_input(2, size)
        return layer.get_output(0)

    def convolution(self, x, weights, name, *, padding, stride=1, dilation=1, groups=1):
        params = weights.parameters(name + "/Conv", "Conv")
        w, b = params[0], params[1] if len(params) == 2 else None
        one_dim = w.ndim == 3
        if one_dim:
            x = self.reshape(x, self.shape(1, int(x.shape[1]), self.dim(x, 2), 1))
            w = np.ascontiguousarray(w[..., None])
            padding, stride, dilation = (padding, 0), (stride, 1), (dilation, 1)
        layer = self.net.add_convolution_nd(x, w.shape[0], w.shape[2:], w,
                                            self.trt.Weights() if b is None else b)
        self.storage.extend((w, b))
        layer.padding_nd = padding
        layer.stride_nd = stride
        layer.dilation_nd = dilation
        layer.num_groups = groups
        result = layer.get_output(0)
        return self.reshape(result, (1, w.shape[0], -1)) if one_dim else result

    def batchnorm(self, x, weights, name):
        scale, bias, mean, variance = weights.parameters(name + "/BatchNormalization", "BatchNormalization")
        shape = (1, scale.size) + (1,) * (len(x.shape) - 2)
        centered = self.ew(x, self.const(mean.reshape(shape)), "SUB")
        normalized = self.ew(centered, self.const(np.sqrt(variance + np.float32(1e-5)).reshape(shape)), "DIV")
        return self.ew(self.ew(normalized, self.const(scale.reshape(shape)), "PROD"), self.const(bias.reshape(shape)), "SUM")

    def dense(self, x, weights, name, bias=True):
        w, = weights.parameters(name + "/MatMul", "MatMul")
        b = weights.parameters(name + "/Add", "Add")[0] if bias else None
        return self.linear(x, np.ascontiguousarray(w.T), b)

    def layernorm(self, x, weights, name):
        scale, = weights.parameters(name + "/Mul", "Mul")
        bias, = weights.parameters(name + "/Add_1", "Add")
        centered = self.ew(x, self.reduce(x, 2), "SUB")
        variance = self.reduce(self.ew(centered, centered, "PROD"), 2)
        denom = self.unary(self.ew(variance, self.scalar(1e-5, 3), "SUM"), "SQRT")
        norm = self.ew(centered, denom, "DIV")
        return self.ew(self.ew(norm, self.const(scale.reshape(1, 1, -1)), "PROD"),
                       self.const(bias.reshape(1, 1, -1)), "SUM")

    def gelu(self, x):
        rank = len(x.shape)
        erf = self.unary(self.ew(x, self.scalar(np.float32(2) ** np.float32(0.5), rank), "DIV"), "ERF")
        return self.ew(self.ew(x, self.ew(erf, self.scalar(1, rank), "SUM"), "PROD"), self.scalar(0.5, rank), "PROD")


def campplus_graph(weights):
    g = FrontendGraph()
    x = g.net.add_input("features", g.trt.float32, (1, -1, 80))
    x = g.reshape(g.transpose(x, (0, 2, 1)), (1, 1, 80, -1))
    x = g.activation(g.convolution(x, weights, "/head/conv1", padding=(1, 1), stride=(1, 1), dilation=(1, 1)), "RELU")
    for stage in (1, 2):
        for block in (0, 1):
            prefix = f"/head/layer{stage}/layer{stage}.{block}"
            stride = (2, 1) if block == 0 else (1, 1)
            residual = g.convolution(x, weights, prefix + "/shortcut/shortcut.0", padding=(0, 0), stride=stride, dilation=(1, 1)) if block == 0 else x
            y = g.activation(g.convolution(x, weights, prefix + "/conv1", padding=(1, 1), stride=stride, dilation=(1, 1)), "RELU")
            y = g.convolution(y, weights, prefix + "/conv2", padding=(1, 1), stride=(1, 1), dilation=(1, 1))
            x = g.activation(g.ew(y, residual, "SUM"), "RELU")
    x = g.activation(g.convolution(x, weights, "/head/conv2", padding=(1, 1), stride=(2, 1), dilation=(1, 1)), "RELU")
    x = g.reshape(x, (1, 320, -1))
    x = g.activation(g.convolution(x, weights, "/xvector/tdnn/linear", padding=2, stride=2), "RELU")
    for stage, count in enumerate((12, 24, 16), 1):
        for block in range(1, count + 1):
            prefix = f"/xvector/block{stage}/tdnnd{block}"
            y = g.activation(g.batchnorm(x, weights, prefix + "/nonlinear1/batchnorm"), "RELU")
            y = g.activation(g.convolution(y, weights, prefix + "/linear1", padding=0), "RELU")
            local = g.convolution(y, weights, prefix + "/cam_layer/linear_local", padding=1 if stage == 1 else 2, dilation=1 if stage == 1 else 2)
            # Segment context: ceil-mode average over 100 frames, excluding the
            # incomplete segment's padding, repeated and cropped to input time.
            pool = g.net.add_pooling_nd(g.reshape(y, (1, 128, -1, 1)), g.trt.PoolingType.AVERAGE, (100, 1))
            pool.stride_nd = (100, 1)
            pool.padding_mode = g.trt.PaddingMode.EXPLICIT_ROUND_UP
            pool.average_count_excludes_padding = True
            segments = g.reshape(pool.get_output(0), (1, 128, -1))
            context = g.crop(g.repeat(segments, 100), g.shape(1, 128, g.dim(y, 2)))
            context = g.ew(context, g.reduce(y, 2), "SUM")
            gate = g.activation(g.convolution(context, weights, prefix + "/cam_layer/linear1", padding=0), "RELU")
            gate = g.activation(g.convolution(gate, weights, prefix + "/cam_layer/linear2", padding=0), "SIGMOID")
            x = g.cat([x, g.ew(local, gate, "PROD")], 1)
        prefix = f"/xvector/transit{stage}"
        x = g.activation(g.batchnorm(x, weights, prefix + "/nonlinear/batchnorm"), "RELU")
        x = g.convolution(x, weights, prefix + "/linear", padding=0)
    x = g.activation(x, "RELU")
    mean = g.reduce(x, 2)
    centered = g.ew(x, mean, "SUB")
    variance = g.reduce(g.ew(centered, centered, "PROD"), 2)
    n = g.reshape(g.net.add_cast(g.dim(x, 2), g.trt.float32).get_output(0), (1, 1, 1))
    variance = g.ew(g.ew(variance, n, "PROD"), g.ew(n, g.scalar(1, 3), "SUB"), "DIV")
    x = g.cat([mean, g.unary(variance, "SQRT")], 1)
    x = g.convolution(x, weights, "/xvector/dense/linear", padding=0)
    x = g.batchnorm(x, weights, "/xvector/dense/nonlinear/batchnorm")
    g.mark(g.reshape(x, (1, 192)), "speaker")
    weights.check_consumed()
    return g


def speech_tokenizer_graph(weights):
    """B=1, unpadded features: official length mask is identically true.

    No padded batches or shorter feats_length are accepted by this contract.
    Consequently token time is ceil(T/4), and the mask cannot change dataflow.
    """
    g = FrontendGraph()
    x = g.net.add_input("features", g.trt.float32, (1, 128, -1))
    for name in ("/conv1", "/conv2"):
        x = g.gelu(g.convolution(x, weights, name, padding=1, stride=2))
    x = g.transpose(x, (0, 2, 1))
    length = g.dim(x, 1)
    # The export deduplicates all twelve layers' identical position tables.
    cos = g.const(weights.array("/blocks.0/attn/rotary_emb/Constant_output_0"))
    sin = g.const(weights.array("/blocks.0/attn/rotary_emb/Constant_5_output_0"))
    cos = g.reshape(g.crop(cos, g.shape(length, 64)), (1, 1, -1, 64))
    sin = g.reshape(g.crop(sin, g.shape(length, 64)), (1, 1, -1, 64))
    for block in range(12):
        prefix = f"/blocks.{block}"
        y = g.layernorm(x, weights, prefix + "/attn_ln")
        q = g.dense(y, weights, prefix + "/attn/query")
        k = g.dense(y, weights, prefix + "/attn/key", bias=False)
        v = g.dense(y, weights, prefix + "/attn/value")
        memory = g.convolution(g.transpose(v, (0, 2, 1)), weights, prefix + "/attn/fsmn_block", padding=15, groups=1280)
        memory = g.ew(g.transpose(memory, (0, 2, 1)), v, "SUM")

        def heads(tensor):
            return g.transpose(g.reshape(tensor, (1, -1, 20, 64)), (0, 2, 1, 3))

        def rotary(tensor):
            tensor = heads(tensor)
            half = g.gather(tensor, g.const(np.r_[32:64, 0:32].astype(np.int32)), 3)
            half = g.ew(half, g.const(np.r_[-np.ones(32), np.ones(32)].astype(np.float32).reshape(1, 1, 1, 64)), "PROD")
            return g.ew(g.ew(tensor, cos, "PROD"), g.ew(half, sin, "PROD"), "SUM")

        scale = g.scalar(np.float32(0.125) ** np.float32(0.5), 4)
        q, k = g.ew(rotary(q), scale, "PROD"), g.ew(rotary(k), scale, "PROD")
        scores = g.net.add_matrix_multiply(q, g.trt.MatrixOperation.NONE, k, g.trt.MatrixOperation.TRANSPOSE).get_output(0)
        prob = g.net.add_softmax(scores)
        prob.axes = 1 << 3
        attended = g.net.add_matrix_multiply(prob.get_output(0), g.trt.MatrixOperation.NONE, heads(v), g.trt.MatrixOperation.NONE).get_output(0)
        attended = g.reshape(g.transpose(attended, (0, 2, 1, 3)), (1, -1, 1280))
        attended = g.dense(attended, weights, prefix + "/attn/out")
        x = g.ew(x, g.ew(attended, memory, "SUM"), "SUM")
        y = g.layernorm(x, weights, prefix + "/mlp_ln")
        y = g.gelu(g.dense(y, weights, prefix + "/mlp/mlp.0"))
        x = g.ew(x, g.dense(y, weights, prefix + "/mlp/mlp.2"), "SUM")
    projected = g.dense(x, weights, "/quantizer/project_in")
    bounded = g.ew(g.activation(projected, "TANH"), g.scalar(0.999, 3), "PROD")
    rounded = g.unary(bounded, "ROUND")
    # Preserve the published straight-through expression in the inference graph.
    codes = g.ew(bounded, g.ew(rounded, bounded, "SUB"), "SUM")
    digits = g.ew(codes, g.scalar(1, 3), "SUM")
    digits = g.ew(digits, g.const((3 ** np.arange(8)).astype(np.float32).reshape(1, 1, 8)), "PROD")
    indices = g.net.add_cast(g.reduce(digits, 2, "SUM", False), g.trt.int32).get_output(0)
    g.mark(indices, "tokens")
    weights.check_consumed()
    return g


def build_engine(model_dir, component, profile=FrontendProfile(), *, workspace_mib=256):
    weights = FrontendWeights(model_dir, component)
    graph = campplus_graph(weights) if component == "campplus" else speech_tokenizer_graph(weights)
    shape = (lambda n: (1, n, 80)) if component == "campplus" else (lambda n: (1, 128, n))
    shapes = tuple(shape(n) for n in (profile.min_frames, profile.opt_frames, profile.max_frames))
    return graph.build({"features": shapes}, workspace_mib)


class FrontendEngine(ComponentEngine):
    def __init__(self, directory, component, *, device=0):
        if component not in CHECKPOINTS:
            raise ValueError("Unknown frontend component")
        output = {"speaker": "float32"} if component == "campplus" else {"tokens": "int32"}
        super().__init__(directory, component, {"features": "float32"}, output, device=device)
        if (self.manifest.get("batch_size") != 1 or self.manifest.get("padded_input") is not False
                or self.manifest.get("learned_execution") != "native_tensorrt"):
            raise ValueError("Unsupported frontend semantics")
        self.component = component
        self.profile = FrontendProfile(**self.manifest["profile"])

    def extract(self, features):
        shape = tuple(features.shape)
        axis = 1 if self.component == "campplus" else 2
        channels = 80 if self.component == "campplus" else 128
        if len(shape) != 3 or shape[0] != 1 or shape[3 - axis] != channels:
            raise ValueError("Invalid batch-one frontend feature layout")
        if not self.profile.min_frames <= shape[axis] <= self.profile.max_frames:
            raise ValueError("Feature length outside frontend profile; no padding or truncation is applied")
        outputs = self.run(features=features)
        expected = (1, 192) if self.component == "campplus" else (1, (shape[axis] + 3) // 4)
        result = next(iter(outputs.values()))
        if tuple(result.shape) != expected:
            raise RuntimeError("Frontend output shape violates the model contract")
        if self.component == "speech_tokenizer" and ((result < 0) | (result >= 6561)).any().item():
            raise RuntimeError("Speech tokenizer produced an out-of-vocabulary token")
        return outputs
