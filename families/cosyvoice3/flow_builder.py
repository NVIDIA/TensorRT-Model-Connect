# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT graph for the offline CosyVoice3 DiT velocity estimator.

Equations follow FunAudioLLM/CosyVoice's flow/DiT/{dit,modules}.py. No ONNX
parser, torch export, sibling family implementation, or custom GPU kernels.
TensorRT owns lowering, fusion, tactics and execution scheduling.

This is a component engine, NOT a complete text-to-speech bundle.
"""

from __future__ import annotations

import numpy as np

from . import trt_compat
from .checkpoint_mapper import validate_weights
from .config import FlowConfig, ShapeProfile
from .constants import time_frequencies

# Number of reduction blocks for every FP32 linear layer; see _Graph.linear.
LINEAR_K_BLOCKS = 4


class _Graph:
    def __init__(self, network, weights, cfg, profile, *, debug_outputs=False):
        self.trt = trt_compat.get_trt()
        self.net, self.weights, self.cfg, self.profile = network, weights, cfg, profile
        self.debug_outputs = debug_outputs
        self.constants = []  # Keep constant storage alive throughout serialization.

    def debug(self, tensor, name):
        if self.debug_outputs:
            tensor.name = f"debug_{name}"
            self.net.mark_output(tensor)

    def const(self, values, shape=None):
        array = np.ascontiguousarray(values, dtype=np.float32)
        if shape is not None:
            array = array.reshape(shape)
        self.constants.append(array)
        return self.net.add_constant(array.shape, array).get_output(0)

    def scalar(self, value, rank):
        return self.const([value], (1,) * rank)

    def ew(self, a, b, op="SUM"):
        return self.net.add_elementwise(a, b, getattr(self.trt.ElementWiseOperation, op)).get_output(0)

    def scale(self, a, value):
        return self.ew(a, self.scalar(value, len(a.shape)), "PROD")

    def unary(self, a, op):
        return self.net.add_unary(a, getattr(self.trt.UnaryOperation, op)).get_output(0)

    def act(self, x, op):
        return self.net.add_activation(x, getattr(self.trt.ActivationType, op)).get_output(0)

    def silu(self, x):
        return self.ew(x, self.act(x, "SIGMOID"), "PROD")

    def mish(self, x):
        softplus = self.net.add_activation(x, self.trt.ActivationType.SOFTPLUS)
        softplus.alpha, softplus.beta = 1.0, 1.0
        return self.ew(x, self.act(softplus.get_output(0), "TANH"), "PROD")

    def gelu(self, x):
        cubic = self.ew(self.ew(x, x, "PROD"), x, "PROD")
        inner = self.scale(self.ew(x, self.scale(cubic, 0.044715)), np.sqrt(2 / np.pi))
        return self.scale(self.ew(x, self.ew(self.act(inner, "TANH"), self.scalar(1, len(x.shape))), "PROD"), 0.5)

    def reshape(self, x, dims):
        layer = self.net.add_shuffle(x)
        layer.reshape_dims = dims
        return layer.get_output(0)

    def transpose(self, x, order):
        layer = self.net.add_shuffle(x)
        layer.first_transpose = order
        return layer.get_output(0)

    def cat(self, values, axis):
        layer = self.net.add_concatenation(values)
        layer.axis = axis
        return layer.get_output(0)

    def slice_last(self, x, start, size):
        rank = len(x.shape)
        # Preserve dynamic time length with an explicit shape tensor.
        shape = self.net.add_shape(x).get_output(0)
        shape = self.net.add_cast(shape, self.trt.int32).get_output(0)
        leading = self.net.add_slice(shape, (0,), (rank - 1,), (1,)).get_output(0)
        n = np.array([size], dtype=np.int32)
        self.constants.append(n)
        trailing = self.net.add_constant((1,), n).get_output(0)
        layer = self.net.add_slice(x, (0,) * (rank - 1) + (start,), (1,) * rank, (1,) * rank)
        layer.set_input(2, self.cat([leading, trailing], 0))
        return layer.get_output(0)

    def linear(self, x, key):
        """x @ W^T + b with the reduction dimension accumulated in blocks.

        TensorRT's FP32 GEMM tactics differ in accumulation order. Kernels
        that accumulate the whole reduction sequentially carry about twice
        the rounding error of the sliced kernels cuBLAS uses for the PyTorch
        reference, and the choice varies between builds. Summing LINEAR_K_BLOCKS
        partial products pairwise bounds the error independent of the tactic
        selected; measured against an FP64 oracle it matches the reference.
        """
        w, b = self.weights[key + ".weight"], self.weights[key + ".bias"]
        rank = len(x.shape)
        n, k = w.shape
        blocks = LINEAR_K_BLOCKS if k % LINEAR_K_BLOCKS == 0 else 1
        step = k // blocks
        parts = []
        for i in range(blocks):
            part = x if blocks == 1 else self.slice_last(x, i * step, step)
            rhs = self.const(w.T[i * step:(i + 1) * step], (1,) * (rank - 2) + (step, n))
            parts.append(self.net.add_matrix_multiply(part, self.trt.MatrixOperation.NONE, rhs,
                                                      self.trt.MatrixOperation.NONE).get_output(0))
        while len(parts) > 1:
            parts = [self.ew(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
        return self.ew(parts[0], self.const(b, (1,) * (rank - 1) + b.shape))

    def norm(self, x):
        axes = 1 << (len(x.shape) - 1)
        shape = (1,) * (len(x.shape) - 1) + (self.cfg.dim,)
        scale = self.const(np.ones(shape, np.float32))
        bias = self.const(np.zeros(shape, np.float32))
        # Declare LayerNorm itself, as the reference does, so TensorRT owns
        # its stable reduction implementation rather than recognizing a chain
        # of separately rounded mean/subtract/square/reduce operations.
        layer = self.net.add_normalization_v2(x, scale, bias, axes)
        layer.epsilon = 1e-6
        return layer.get_output(0)

    def conv_position(self, x):
        cfg = self.cfg
        x = self.reshape(self.transpose(x, (0, 2, 1)), (2, cfg.dim, -1, 1))
        for i in (1, 2):
            key = f"input_embed.conv_pos_embed.conv{i}.0"
            w, b = self.weights[key + ".weight"], self.weights[key + ".bias"]
            layer = self.net.add_convolution_nd(x, cfg.dim, (cfg.conv_kernel, 1), w.reshape(*w.shape, 1), b)
            layer.num_groups = cfg.conv_groups
            layer.pre_padding = (cfg.conv_kernel - 1, 0)
            layer.post_padding = (0, 0)
            x = self.mish(layer.get_output(0))
            # Expose a B,T,D view only in diagnostic engines.  Keep the
            # channel-first tensor above as the production convolution path.
            debug_view = self.transpose(self.reshape(x, (2, cfg.dim, -1)), (0, 2, 1))
            self.debug(debug_view, f"conv_{i}")
        return self.transpose(self.reshape(x, (2, cfg.dim, -1)), (0, 2, 1))

    def rotary(self, x, cos, sin):
        # The published checkpoint applies RoPE BEFORE splitting heads. Only
        # the first head_dim elements of the flattened projection are rotated.
        # Applying it independently to every head changes the model's math.
        part = self.slice_last(x, 0, self.cfg.head_dim)
        # x-transformers uses adjacent pairs (-odd, even), not half-split RoPE.
        indices = np.arange(self.cfg.head_dim, dtype=np.int32) ^ 1
        self.constants.append(indices)
        index_tensor = self.net.add_constant(indices.shape, indices).get_output(0)
        rotated = self.net.add_gather(part, index_tensor, 2).get_output(0)
        signs = np.tile([-1.0, 1.0], self.cfg.head_dim // 2)
        rotated = self.ew(rotated, self.const(signs[None, None]), "PROD")
        part = self.ew(self.ew(part, cos, "PROD"), self.ew(rotated, sin, "PROD"))
        if self.cfg.dim == self.cfg.head_dim:
            return part
        return self.cat([part, self.slice_last(x, self.cfg.head_dim, self.cfg.dim - self.cfg.head_dim)], 2)

    def attention(self, x, mask, key, cos, sin):
        cfg = self.cfg
        q = self.rotary(self.linear(x, key + ".to_q"), cos, sin)
        k = self.rotary(self.linear(x, key + ".to_k"), cos, sin)
        v = self.linear(x, key + ".to_v")
        q, k, v = [self.transpose(self.reshape(y, (2, -1, cfg.heads, cfg.head_dim)), (0, 2, 1, 3)) for y in (q, k, v)]
        # The runtime reference (PyTorch FP32 SDPA selects the memory-efficient
        # kernel) applies head_dim**-0.5 to the unrounded QK^T products. Only
        # the ONNX export's math decomposition pre-scales Q and K by
        # head_dim**-0.25, rounding both operands before the matmul; matching
        # that export doubled the logit rounding error against an FP64 oracle.
        # For head_dim=64 the scale 1/8 is a power of two, so scaling Q alone
        # is exact and equals scaling the products.
        q = self.scale(q, cfg.head_dim ** -0.5)
        scores = self.net.add_matrix_multiply(q, self.trt.MatrixOperation.NONE, k, self.trt.MatrixOperation.TRANSPOSE).get_output(0)
        bool_mask = self.net.add_cast(self.reshape(mask, (2, 1, 1, -1)), self.trt.bool).get_output(0)
        scores = self.net.add_select(bool_mask, scores, self.scalar(float("-inf"), 4)).get_output(0)
        softmax = self.net.add_softmax(scores)
        softmax.axes = 1 << 3
        out = self.net.add_matrix_multiply(softmax.get_output(0), self.trt.MatrixOperation.NONE, v, self.trt.MatrixOperation.NONE).get_output(0)
        out = self.reshape(self.transpose(out, (0, 2, 1, 3)), (2, -1, cfg.dim))
        out = self.linear(out, key + ".to_out.0")
        return self.ew(out, self.transpose(mask, (0, 2, 1)), "PROD")

    def build(self):
        cfg = self.cfg
        shapes = {
            "x": (2, cfg.mel_dim, -1), "mask": (2, 1, -1),
            "mu": (2, cfg.mel_dim, -1), "t": (2,),
            "spks": (2, cfg.spk_dim), "cond": (2, cfg.mel_dim, -1),
        }
        inputs = {name: self.net.add_input(name, self.trt.float32, shape) for name, shape in shapes.items()}
        positions = self.net.add_input("positions", self.trt.int32, (-1,))
        # Name equal dynamic dimensions so invalid cross-input shapes cannot
        # silently broadcast or read past a request's actual frame count.
        for name in ("x", "mask", "mu", "cond"):
            inputs[name].set_dimension_name(2, "frames")
        positions.set_dimension_name(0, "frames")

        inv_freq = 1 / (10000 ** (np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))
        inv_freq = self.weights.get("rotary_embed.inv_freq", inv_freq)
        phases = np.arange(self.profile.max_frames, dtype=np.float32)[:, None] * inv_freq[None]
        phases = np.repeat(phases, 2, axis=-1)
        cos, sin = [self.net.add_gather(self.const(fn(phases)[None]), positions, 1).get_output(0) for fn in (np.cos, np.sin)]

        freq = time_frequencies(cfg.time_dim)
        phases = self.ew(self.scale(self.reshape(inputs["t"], (2, 1)), 1000), self.const(freq[None]), "PROD")
        time = self.cat([self.unary(phases, "SIN"), self.unary(phases, "COS")], 1)
        time = self.linear(self.silu(self.linear(time, "time_embed.time_mlp.0")), "time_embed.time_mlp.2")
        self.debug(time, "time")

        x, cond, mu = [self.transpose(inputs[name], (0, 2, 1)) for name in ("x", "cond", "mu")]
        # Use a scalar-width tensor to broadcast the speaker over dynamic time.
        zero = self.scale(self.slice_last(x, 0, 1), 0)
        spks = self.ew(zero, self.reshape(inputs["spks"], (2, 1, cfg.spk_dim)))
        x = self.linear(self.cat([x, cond, mu, spks], 2), "input_embed.proj")
        self.debug(x, "input_projection")
        x = self.ew(x, self.conv_position(x))
        self.debug(x, "input")
        activated_time = self.silu(time)
        for i in range(cfg.depth):
            key = f"transformer_blocks.{i}"
            modulation = self.linear(activated_time, key + ".attn_norm.linear")
            params = [self.reshape(self.slice_last(modulation, j * cfg.dim, cfg.dim), (2, 1, cfg.dim)) for j in range(6)]
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = params
            norm = self.ew(self.ew(self.norm(x), self.ew(scale_a, self.scalar(1, 3)), "PROD"), shift_a)
            attention = self.attention(norm, inputs["mask"], key + ".attn", cos, sin)
            x = self.ew(x, self.ew(gate_a, attention, "PROD"))
            norm = self.ew(self.ew(self.norm(x), self.ew(scale_f, self.scalar(1, 3)), "PROD"), shift_f)
            ff = self.linear(self.gelu(self.linear(norm, key + ".ff.ff.0.0")), key + ".ff.ff.2")
            x = self.ew(x, self.ew(gate_f, ff, "PROD"))
            self.debug(x, f"block_{i:02d}")
        modulation = self.linear(activated_time, "norm_out.linear")
        scale, shift = [self.reshape(self.slice_last(modulation, i * cfg.dim, cfg.dim), (2, 1, cfg.dim)) for i in range(2)]
        x = self.ew(self.ew(self.norm(x), self.ew(scale, self.scalar(1, 3)), "PROD"), shift)
        self.debug(x, "norm_out")
        out = self.transpose(self.linear(x, "proj_out"), (0, 2, 1))
        out.name = "velocity"
        self.net.mark_output(out)
        return inputs


def build_flow_engine(weights, cfg=FlowConfig(), profile=ShapeProfile(), *, workspace_mib=512, debug_outputs=False):
    """Serialize a batch-2 (conditional + unconditional), FP32 offline DiT.

    No streaming/chunk mask or hidden FP16/TF32 fallback. max_frames includes
    prompt frames, not just newly generated speech. Inputs/outputs are FP32.
    """
    if type(workspace_mib) is not int or workspace_mib < 16:
        raise ValueError("workspace_mib must be an integer >= 16")
    weights = validate_weights(weights, cfg)
    trt = trt_compat.get_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(network, weights, cfg, profile, debug_outputs=debug_outputs)
    inputs = graph.build()
    config = builder.create_builder_config()
    config.builder_optimization_level = 4
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    config.clear_flag(trt.BuilderFlag.TF32)
    opt = builder.create_optimization_profile()
    for name in ("x", "mask", "mu", "cond"):
        channels = int(inputs[name].shape[1])
        opt.set_shape(name, *[(2, channels, n) for n in (profile.min_frames, profile.opt_frames, profile.max_frames)])
    opt.set_shape("positions", *[(n,) for n in (profile.min_frames, profile.opt_frames, profile.max_frames)])
    config.add_optimization_profile(opt)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("CosyVoice3 native Flow TensorRT build failed; see TensorRT diagnostics")
    return bytes(plan)
