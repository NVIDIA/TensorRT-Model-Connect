# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CosyVoice3-local graph construction and synchronous component mechanics.

These helpers do not import an inference framework's model implementation.
PyTorch in the runner owns CUDA buffers only; TensorRT executes the graphs.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading

import numpy as np

from . import trt_compat


class Graph:
    def __init__(self):
        self.trt = t = trt_compat.get_trt()
        self.logger = t.Logger(t.Logger.WARNING)
        self.builder = t.Builder(self.logger)
        self.net = self.builder.create_network(1 << int(t.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.storage = []

    def const(self, value, dtype=None):
        value = np.asarray(value, dtype=dtype)
        shape = value.shape
        value = np.ascontiguousarray(value)
        self.storage.append(value)
        return self.net.add_constant(shape, value).get_output(0)

    def scalar(self, value, rank):
        return self.const(np.full((1,) * rank, value, np.float32))

    def ew(self, a, b, op):
        return self.net.add_elementwise(a, b, getattr(self.trt.ElementWiseOperation, op)).get_output(0)

    def unary(self, x, op):
        return self.net.add_unary(x, getattr(self.trt.UnaryOperation, op)).get_output(0)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        layer.zero_is_placeholder = False
        if isinstance(shape, tuple):
            layer.reshape_dims = shape
        else:
            layer.set_input(1, shape)
        return layer.get_output(0)

    def transpose(self, x, order):
        layer = self.net.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    def gather(self, x, indices, axis):
        return self.net.add_gather(x, indices, axis).get_output(0)

    def cat(self, values, axis):
        layer = self.net.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def dim(self, x, axis):
        shape = self.net.add_shape(x).get_output(0)
        shape = self.net.add_cast(shape, self.trt.int32).get_output(0)
        return self.gather(shape, self.const([axis], np.int32), 0)

    def shape(self, *parts):
        return self.cat([self.const([p], np.int32) if isinstance(p, int) else p for p in parts], 0)

    def linear(self, x, weight, bias=None):
        rank = len(x.shape)
        w = self.const(weight.reshape((1,) * (rank - 2) + weight.shape))
        result = self.net.add_matrix_multiply(x, self.trt.MatrixOperation.NONE,
                                              w, self.trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is not None:
            result = self.ew(result, self.const(bias.reshape((1,) * (rank - 1) + bias.shape)), "SUM")
        return result

    def activation(self, x, name, alpha=None):
        layer = self.net.add_activation(x, getattr(self.trt.ActivationType, name))
        if alpha is not None:
            layer.alpha = alpha
        return layer.get_output(0)

    def conv(self, x, w, b=None, *, left=0, right=0, stride=1, dilation=1, transpose=False):
        """NCT convolution through TensorRT's NC(H)W API."""
        x = self.reshape(x, self.shape(1, int(x.shape[1]), self.dim(x, 2), 1))
        factory = self.net.add_deconvolution_nd if transpose else self.net.add_convolution_nd
        channels = w.shape[1] if transpose else w.shape[0]
        kernel = np.ascontiguousarray(w[..., None])
        layer = factory(x, channels, (w.shape[2], 1), kernel,
                        self.trt.Weights() if b is None else b)
        # Keep weights alive through serialization, including expanded DFT filters.
        self.storage.extend((kernel, b))
        layer.pre_padding, layer.post_padding = (left, 0), (right, 0)
        layer.stride_nd = (stride, 1)
        if not transpose:
            layer.dilation_nd = (dilation, 1)
        return self.reshape(layer.get_output(0), (1, channels, -1))

    def repeat(self, x, factor):
        channels = int(x.shape[1])
        x = self.reshape(x, (1, channels, -1, 1))
        x = self.gather(x, self.const(np.zeros(factor, np.int32)), 3)
        return self.reshape(x, (1, channels, -1))

    def mark(self, x, name):
        x.name = name
        self.net.mark_output(x)

    def build(self, profiles, workspace_mib=256):
        if type(workspace_mib) is not int or workspace_mib <= 0:
            raise ValueError("workspace_mib must be a positive integer")
        config = self.builder.create_builder_config()
        config.builder_optimization_level = 4
        config.clear_flag(self.trt.BuilderFlag.TF32)
        config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
        profile = self.builder.create_optimization_profile()
        for name, shapes in profiles.items():
            if profile.set_shape(name, *shapes) is False:
                raise ValueError(f"Invalid profile: {name}")
        config.add_optimization_profile(profile)
        plan = self.builder.build_serialized_network(self.net, config)
        if plan is None:
            raise RuntimeError("CosyVoice3 TensorRT component build failed")
        return bytes(plan)


class ComponentEngine:
    """One locked execution context; no implicit dtype/device conversion."""

    def __init__(self, directory, component, inputs, outputs, *, device=0):
        import torch

        path = Path(directory)
        self.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if (m.get("schema_version") != 1 or m.get("component") != f"cosyvoice3_{component}"
                or m.get("precision") != "fp32" or m.get("streaming") is not False):
            raise ValueError(f"Unsupported {component} manifest")
        plan = (path / f"{component}.plan").read_bytes()
        self.device = torch.device("cuda", device)
        self.torch = torch
        self.inputs, self.outputs = inputs, outputs
        self.lock = threading.Lock()
        t = trt_compat.get_trt()
        self.logger = t.Logger(t.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.runtime = t.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            if self.engine is None:
                raise RuntimeError(f"Cannot load {component}; rebuild for this GPU/TensorRT")
            expected = inputs | outputs
            if {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)} != set(expected):
                raise ValueError(f"Unexpected {component} engine I/O")
            for name, dtype in expected.items():
                mode = t.TensorIOMode.INPUT if name in inputs else t.TensorIOMode.OUTPUT
                if self.engine.get_tensor_dtype(name) != getattr(t, dtype) or self.engine.get_tensor_mode(name) != mode:
                    raise ValueError(f"Invalid engine I/O contract: {name}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(f"Cannot allocate {component} execution context")
            self.stream = torch.cuda.Stream(device=self.device)

    def run(self, **inputs):
        torch = self.torch
        if set(inputs) != set(self.inputs):
            raise ValueError("Incorrect component inputs")
        with self.lock, torch.cuda.device(self.device), torch.inference_mode():
            buffers = {}
            for name, tensor in inputs.items():
                if tensor.device != self.device or tensor.dtype != getattr(torch, self.inputs[name]):
                    raise ValueError(f"{name} must be {self.inputs[name]} on {self.device}")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                    raise ValueError(f"{name} contains NaN/Inf")
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"Shape outside engine profile: {name} {tuple(tensor.shape)}")
                # TensorRT still requires a non-null address for empty KV caches.
                buffers[name] = tensor.contiguous() if tensor.numel() else torch.empty(1, device=self.device, dtype=tensor.dtype)
                if not self.context.set_tensor_address(name, buffers[name].data_ptr()):
                    raise RuntimeError(f"Cannot bind {name}")
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise ValueError(f"Unresolved shapes: {unresolved}")
            outputs = {}
            for name, dtype in self.outputs.items():
                shape = tuple(self.context.get_tensor_shape(name))
                if any(n < 0 for n in shape):
                    raise RuntimeError(f"Unresolved output shape: {name}")
                outputs[name] = torch.empty(shape, dtype=getattr(torch, dtype), device=self.device)
                if not self.context.set_tensor_address(name, outputs[name].data_ptr()):
                    raise RuntimeError(f"Cannot bind {name}")
            stream = self.stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT component execution failed")
            finally:
                stream.synchronize()
            if any(not torch.isfinite(x).all().item() for x in outputs.values()):
                raise RuntimeError("TensorRT produced NaN/Inf")
            return outputs
