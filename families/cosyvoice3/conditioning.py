# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native, family-owned offline token/speaker conditioning (not full TTS).

Equations: pinned CosyVoice CausalMaskedDiffWithDiT.inference and
PreLookaheadLayer.forward. Learned computation executes only in TensorRT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading

import numpy as np

from . import trt_compat


@dataclass(frozen=True)
class TokenProfile:
    min_tokens: int = 2
    opt_tokens: int = 32
    max_tokens: int = 128

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if any(type(n) is not int for n in values) or not 1 <= values[0] <= values[1] <= values[2] <= 7500:
            raise ValueError("Require 1 <= min_tokens <= opt_tokens <= max_tokens <= 7500")


def weight_shapes():
    return {
        "input_embedding.weight": (6561, 80),
        "spk_embed_affine_layer.weight": (80, 192),
        "spk_embed_affine_layer.bias": (80,),
        "pre_lookahead_layer.conv1.weight": (1024, 80, 4),
        "pre_lookahead_layer.conv1.bias": (1024,),
        "pre_lookahead_layer.conv2.weight": (80, 1024, 3),
        "pre_lookahead_layer.conv2.bias": (80,),
    }


def validate_weights(weights):
    shapes = weight_shapes()
    if set(weights) != set(shapes):
        raise ValueError("Conditioner requires exactly the seven published weight tensors")
    for key, shape in shapes.items():
        value = weights[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid conditioner weight {key}: expected finite FP32 {shape}")


def load_weights(model_dir):
    import torch
    import yaml

    from .config import read_config

    read_config(model_dir)
    # The DiT config reader does not validate this separate graph's topology.
    raw = yaml.load((Path(model_dir) / "cosyvoice3.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    flow = raw["flow"]
    expected = {"input_size": "80", "output_size": "80", "vocab_size": "6561", "pre_lookahead_len": "3"}
    layer = flow.get("pre_lookahead_layer")
    if (any(flow.get(key) != value for key, value in expected.items())
            or flow.get("spk_embed_dim") not in ("192", "<spk_embed_dim>")
            or flow.get("token_mel_ratio") not in ("2", "<token_mel_ratio>")
            or not isinstance(layer, dict)
            or any(layer.get(key) != value for key, value in
                   {"in_channels": "80", "channels": "1024", "pre_lookahead_len": "3"}.items())):
        raise ValueError("Unsupported CosyVoice3 conditioning topology")

    state = torch.load(Path(model_dir) / "flow.pt", map_location="cpu", weights_only=True, mmap=True)
    keys = {key for key in state if not key.startswith("decoder.")}
    if keys != set(weight_shapes()):
        raise ValueError("Checkpoint has unexpected conditioning parameters")
    if any(not torch.is_tensor(state[key]) or not state[key].is_floating_point() for key in keys):
        raise ValueError("Conditioner parameters must be floating-point tensors")
    weights = {key: state[key].float().numpy().copy() for key in keys}
    validate_weights(weights)
    return weights


def build_engine(weights, profile: TokenProfile, *, workspace_mib=64):
    """Gather -> right-lookahead conv -> causal conv -> residual -> repeat x2."""
    validate_weights(weights)
    if type(workspace_mib) is not int or workspace_mib <= 0:
        raise ValueError("workspace_mib must be a positive integer")
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    storage = []

    def const(value):
        value = np.ascontiguousarray(value)
        storage.append(value)
        return net.add_constant(value.shape, value).get_output(0)

    def ew(a, b, operation):
        return net.add_elementwise(a, b, getattr(trt.ElementWiseOperation, operation)).get_output(0)

    def reshape(x, shape):
        layer = net.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(x, order):
        layer = net.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    tokens = net.add_input("tokens", trt.int32, (1, -1))
    speaker = net.add_input("speaker", trt.float32, (1, 192))
    embedded = net.add_gather(const(weights["input_embedding.weight"]), tokens, 0).get_output(0)
    x = reshape(transpose(embedded, (0, 2, 1)), (1, 80, -1, 1))
    for index, channels, kernel, left, right in ((1, 1024, 4, 0, 3), (2, 80, 3, 2, 0)):
        key = f"pre_lookahead_layer.conv{index}"
        layer = net.add_convolution_nd(x, channels, (kernel, 1),
                                      weights[key + ".weight"][..., None], weights[key + ".bias"])
        layer.pre_padding, layer.post_padding = (left, 0), (right, 0)
        x = layer.get_output(0)
        if index == 1:
            activation = net.add_activation(x, trt.ActivationType.LEAKY_RELU)
            activation.alpha = .01
            x = activation.get_output(0)
    h = ew(transpose(reshape(x, (1, 80, -1)), (0, 2, 1)), embedded, "SUM")
    # Insert a repetition axis and gather [0,0]: repeat_interleave, not tile.
    h = reshape(h, (1, -1, 1, 80))
    h = net.add_gather(h, const(np.array([0, 0], np.int32)), 2).get_output(0)
    mu = transpose(reshape(h, (1, -1, 80)), (0, 2, 1))
    norm = net.add_reduce(ew(speaker, speaker, "PROD"), trt.ReduceOperation.SUM, 2, True).get_output(0)
    norm = net.add_unary(norm, trt.UnaryOperation.SQRT).get_output(0)
    norm = ew(norm, const(np.array([[1e-12]], np.float32)), "MAX")
    normalized = ew(speaker, norm, "DIV")
    projected = net.add_matrix_multiply(normalized, trt.MatrixOperation.NONE,
                                       const(weights["spk_embed_affine_layer.weight"].T),
                                       trt.MatrixOperation.NONE).get_output(0)
    spks = ew(projected, const(weights["spk_embed_affine_layer.bias"][None]), "SUM")
    for tensor, name in ((mu, "mu"), (spks, "spks")):
        tensor.name = name
        net.mark_output(tensor)
    config = builder.create_builder_config()
    config.builder_optimization_level = 4
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    shapes = builder.create_optimization_profile()
    shapes.set_shape("tokens", *((1, n) for n in asdict(profile).values()))
    shapes.set_shape("speaker", (1, 192), (1, 192), (1, 192))
    config.add_optimization_profile(shapes)
    plan = builder.build_serialized_network(net, config)
    if plan is None:
        raise RuntimeError("TensorRT conditioner build failed")
    return bytes(plan)


class ConditioningEngine:
    """Strict offline B=1 runner; no implicit dtype/device changes or fallback."""

    def __init__(self, directory, *, device=0):
        import torch

        self.torch = torch
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1 or manifest.get("component") != "cosyvoice3_conditioner"
                or manifest.get("precision") != "fp32" or manifest.get("streaming") is not False):
            raise ValueError("Unsupported conditioner manifest")
        self.profile = TokenProfile(**manifest["profile"])
        plan = (path / "conditioning.plan").read_bytes()
        self.device = torch.device("cuda", device)
        trt = trt_compat.get_trt()
        self.logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(plan)
            if self.engine is None:
                raise RuntimeError("Cannot load conditioner; rebuild for this GPU/TensorRT")
            names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
            if names != {"tokens", "speaker", "mu", "spks"}:
                raise ValueError("Unexpected conditioner I/O")
            for name in names:
                dtype = trt.int32 if name == "tokens" else trt.float32
                mode = trt.TensorIOMode.INPUT if name in ("tokens", "speaker") else trt.TensorIOMode.OUTPUT
                if self.engine.get_tensor_dtype(name) != dtype or self.engine.get_tensor_mode(name) != mode:
                    raise ValueError(f"Unexpected conditioner type/mode: {name}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Cannot allocate conditioner context")
        self.lock = threading.Lock()

    def __call__(self, tokens, speaker):
        torch = self.torch
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("tokens must have shape [1, N]")
        count = tokens.shape[1]
        if not self.profile.min_tokens <= count <= self.profile.max_tokens:
            raise ValueError("Token count is outside conditioner profile")
        if tokens.dtype != torch.int32 or tokens.device != self.device:
            raise ValueError("tokens must be INT32 on the engine device")
        # Offline input has no padding: reject invalid IDs rather than clamping
        # padding sentinels into apparently valid speech, as training code can.
        if (tokens < 0).any().item() or (tokens >= 6561).any().item():
            raise ValueError("Speech token IDs must be in [0, 6561)")
        if speaker.shape != (1, 192) or speaker.dtype != torch.float32 or speaker.device != self.device:
            raise ValueError("speaker must be FP32 [1, 192] on the engine device")
        if not torch.isfinite(speaker).all().item():
            raise ValueError("speaker contains nonfinite values")
        with self.lock, torch.cuda.device(self.device), torch.inference_mode():
            inputs = {"tokens": tokens.contiguous(), "speaker": speaker.contiguous()}
            outputs = {"mu": torch.empty((1, 80, count * 2), device=self.device, dtype=torch.float32),
                       "spks": torch.empty((1, 80), device=self.device, dtype=torch.float32)}
            for name, value in inputs.items():
                if not self.context.set_input_shape(name, tuple(value.shape)):
                    raise RuntimeError(f"Conditioner rejected {name} shape")
            if self.context.infer_shapes():
                raise RuntimeError("Unresolved conditioner shapes")
            for name, value in outputs.items():
                if tuple(self.context.get_tensor_shape(name)) != tuple(value.shape):
                    raise RuntimeError(f"Unexpected conditioner output shape: {name}")
            for name, value in {**inputs, **outputs}.items():
                if not self.context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError(f"Conditioner rejected {name} address")
            stream = torch.cuda.current_stream(self.device)
            try:
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("Conditioner execution failed")
            finally:
                stream.synchronize()
            if any(not torch.isfinite(value).all().item() for value in outputs.values()):
                raise RuntimeError("Conditioner produced nonfinite output")
            return outputs
