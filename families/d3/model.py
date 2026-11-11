# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native dense D3 graphs. Each noncausal layer consumes the complete prompt.

DeltaNet is a causal single-token graph driven in prompt order by the native
runtime. No question reuses another question's state. TensorRT owns all GPU
lowering; these graphs contain no custom kernels or platform dispatch.
"""
from __future__ import annotations
import numpy as np
import ml_dtypes
import tensorrt as trt
from safetensors import safe_open
from .config import read_config
from .support import profile_name
from . import graph_ops
from .batch_graph import delta_layer
from .weights import _open_safetensors, _load_tensor
from .vision import VisionConfig, load_vision_weights, build_vision_engine

BF16 = ml_dtypes.bfloat16


class Graph:
    """One strongly typed family-owned plan and its shape profiles."""
    def __init__(self, maximum, verbose=False, max_batch=1):
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.net = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.clear_flag(trt.BuilderFlag.TF32)
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        self.profile = self.builder.create_optimization_profile()
        self.maximum = maximum
        self.max_batch = max_batch
        self.dynamic = False
        self.constants = []

    def input(self, name, shape, dtype=trt.bfloat16, axes=None):
        x = self.net.add_input(name, dtype, shape)
        axes = axes or {}
        if -1 in shape:
            self.dynamic = True
            minimum, optimum, maximum = [], [], []
            for axis, d in enumerate(shape):
                kind = axes.get(axis, "sequence")
                limit = self.max_batch if kind == "batch" else self.maximum
                minimum.append(1 if d == -1 else d)
                optimum.append(min(4 if kind == "batch" else 128, limit) if d == -1 else d)
                maximum.append(limit if d == -1 else d)
                if d == -1:
                    x.set_dimension_name(axis, kind)
            self.profile.set_shape(name, tuple(minimum), tuple(optimum), tuple(maximum))
        return x

    def dim(self, x, axis):
        shape = self.net.add_shape(x).get_output(0)
        return self.cast(self.net.add_gather(shape, self.constant([axis], np.int32),
                                             0).get_output(0), trt.int32)

    def constant(self, data, dtype=np.float32):
        bf16 = np.dtype(dtype) == np.dtype(BF16)
        a = np.ascontiguousarray(data, dtype=np.float32 if bf16 else dtype)
        self.constants.append(a)
        out = self.net.add_constant(a.shape, a).get_output(0)
        return self.cast(out, trt.bfloat16) if bf16 else out

    def cast(self, x, dtype):
        return x if x.dtype == dtype else self.net.add_cast(x, dtype).get_output(0)

    def reshape(self, x, shape):
        layer = self.net.add_shuffle(x)
        if all(isinstance(d, int) for d in shape):
            layer.reshape_dims = shape
        else:
            pieces = [self.constant([d], np.int32) if isinstance(d, int) else d for d in shape]
            joined = self.net.add_concatenation(pieces)
            joined.axis = 0
            layer.set_input(1, joined.get_output(0))
        return layer.get_output(0)

    def transpose(self, x, axes):
        layer = self.net.add_shuffle(x)
        layer.first_transpose = trt.Permutation(axes)
        return layer.get_output(0)

    def binary(self, a, b, op):
        # Preserve BF16 model storage boundaries with FP32 elementwise opmath.
        dtype = a.dtype
        rounded = dtype == trt.bfloat16 and b.dtype == dtype
        if rounded:
            a, b = self.cast(a, trt.float32), self.cast(b, trt.float32)
        out = self.net.add_elementwise(a, b, op).get_output(0)
        return self.cast(out, dtype) if rounded else out

    def add(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def gather(self, x, indices, axis):
        return self.net.add_gather(x, self.constant(indices, np.int32), axis).get_output(0)

    def linear(self, x, weight):
        shape = (1,) * (len(x.shape) - 2) + tuple(weight.T.shape)
        w = self.cast(self.constant(weight.T.reshape(shape), BF16), x.dtype)
        return self.net.add_matrix_multiply(x, trt.MatrixOperation.NONE, w, trt.MatrixOperation.NONE).get_output(0)

    def norm(self, x, weight, centered=True):
        return graph_ops.add_rms_norm(self.net, x, weight.size,
                                      1 + weight if centered else weight,
                                      self.constant([[1e-6]]), BF16)

    def finish(self, outputs):
        for name, tensor in outputs.items():
            tensor.name = name
            self.net.mark_output(tensor)
        if self.dynamic:
            self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.net, self.config)
        if plan is None:
            raise RuntimeError("D3 TensorRT graph compilation failed")
        return bytes(plan)


def attention(g, x, w, text):
    """Independent bidirectional GQA rows, with right-padding key masks."""
    heads, kv, dim = (text[k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
    batch, seq = g.dim(x, 0), g.dim(x, 1)
    n = g.norm(x, w["input_layernorm.weight"])
    qg = g.reshape(g.linear(n, w["self_attn.q_proj.weight"]), (batch, seq, heads, 2, dim))
    q = g.reshape(g.gather(qg, [0], 3), (batch, seq, heads, dim))
    gate = g.reshape(g.gather(qg, [1], 3), (batch, seq, heads * dim))
    k = g.reshape(g.linear(n, w["self_attn.k_proj.weight"]), (batch, seq, kv, dim))
    v = g.reshape(g.linear(n, w["self_attn.v_proj.weight"]), (batch, seq, kv, dim))

    def norm_heads(t, weight, count):
        return g.reshape(g.norm(g.reshape(t, (-1, dim)), weight), (batch, seq, count, dim))

    q = norm_heads(q, w["self_attn.q_norm.weight"], heads)
    k = norm_heads(k, w["self_attn.k_norm.weight"], kv)
    angles = g.input("angles", (-1, -1, 32), trt.float32, {0: "batch", 1: "sequence"})
    cosine = g.cast(g.net.add_unary(angles, trt.UnaryOperation.COS).get_output(0), trt.bfloat16)
    sine = g.cast(g.net.add_unary(angles, trt.UnaryOperation.SIN).get_output(0), trt.bfloat16)
    cosine, sine = [g.reshape(t, (batch, seq, 1, 32)) for t in (cosine, sine)]

    def rotate(t):
        a, b, tail = [g.gather(t, list(range(lo, hi)), 3) for lo, hi in [(0, 32), (32, 64), (64, dim)]]
        first = g.binary(g.mul(a, cosine), g.mul(b, sine), trt.ElementWiseOperation.SUB)
        second = g.add(g.mul(b, cosine), g.mul(a, sine))
        concat = g.net.add_concatenation([first, second, tail])
        concat.axis = 3
        return concat.get_output(0)

    q, k = rotate(q), rotate(k)
    repeated = [i for i in range(kv) for _ in range(heads // kv)]
    k, v = g.gather(k, repeated, 2), g.gather(v, repeated, 2)
    q, k, v = [g.transpose(t, (0, 2, 1, 3)) for t in (q, k, v)]
    score = g.net.add_matrix_multiply(g.cast(q, trt.float32), trt.MatrixOperation.NONE,
                                     g.cast(k, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
    score = g.mul(score, g.constant([[[[dim ** -.5]]]]))
    mask = g.input("attention_mask", (-1, -1), trt.float32, {0: "batch", 1: "sequence"})
    score = g.add(score, g.reshape(mask, (batch, 1, 1, seq)))
    probs = g.net.add_softmax(score)
    probs.axes = 1 << 3
    value = g.net.add_matrix_multiply(probs.get_output(0), trt.MatrixOperation.NONE,
                                     g.cast(v, trt.float32), trt.MatrixOperation.NONE).get_output(0)
    value = g.cast(g.reshape(g.transpose(value, (0, 2, 1, 3)), (batch, seq, heads * dim)), trt.bfloat16)
    value = g.mul(value, g.cast(g.net.add_activation(g.cast(gate, trt.float32),
                                         trt.ActivationType.SIGMOID).get_output(0), trt.bfloat16))
    residual = g.add(x, g.linear(value, w["self_attn.o_proj.weight"]))
    n = g.norm(residual, w["post_attention_layernorm.weight"])
    gate = g.linear(n, w["mlp.gate_proj.weight"])
    up = g.linear(n, w["mlp.up_proj.weight"])
    mixed = g.mul(graph_ops.add_activation(g.net, gate, "silu", BF16), up)
    return g.add(residual, g.linear(mixed, w["mlp.down_proj.weight"]))



def build(request, writer):
    config, decision, maximum, height, width = read_config(request)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(request.model_dir / "tokenizer.json"))
    for code, token in zip(decision["codes"], decision["token_ids"]):
        if tokenizer.encode(code, add_special_tokens=False).ids != [token]:
            raise ValueError("D3 answer-code/token mapping differs from tokenizer")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    template = ImmutableSandboxedEnvironment().from_string(
        (request.model_dir / "chat_template.jinja").read_text())
    sample = template.render(messages=[{"role": "system", "content": "system"},
                                       {"role": "user", "content": "user"}],
                             add_generation_prompt=True, enable_thinking=False)
    if sample != ("<|im_start|>system\nsystem<|im_end|>\n<|im_start|>user\nuser"
                  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise ValueError("D3 checkpoint chat template differs from the native prompt contract")
    text = config["text_config"]
    writer.set_header(family="d3", task=request.task, backend=request.backend)
    stack, readers = _open_safetensors(request.model_dir)
    with stack:
        all_keys = {key for reader in readers for key in reader.keys()}
        # Qwen3_5Model saves unprefixed language_model/visual; some exports use model.*.
        prefix = "model." if "model.language_model.embed_tokens.weight" in all_keys else ""
        base = prefix + "language_model."
        g = Graph(maximum, request.verbose, request.max_batch_size)
        ids = g.input("input_ids", (-1, -1), trt.int32, {0: "batch", 1: "sequence"})
        embedding = g.constant(_load_tensor(readers, base + "embed_tokens.weight"), BF16)
        out = g.net.add_gather(embedding, ids, 0).get_output(0)
        writer.add_bytes("embedding.plan", g.finish({"hidden": out}))
        del embedding, out, g
        for i, kind in enumerate(text["layer_types"]):
            root = base + f"layers.{i}."
            weights = {key[len(root):]: _load_tensor(readers, key) for key in all_keys if key.startswith(root)}
            g = Graph(maximum, request.verbose, request.max_batch_size)
            if kind == "linear_attention":
                x = g.input("hidden", (-1, text["hidden_size"]), axes={0: "batch"})
                outputs = delta_layer(g, x, weights, text)
            else:
                x = g.input("hidden", (-1, -1, text["hidden_size"]), axes={0: "batch", 1: "sequence"})
                outputs = {"hidden_out": attention(g, x, weights, text)}
            writer.add_bytes(f"layer_{i}.plan", g.finish(outputs))
            del outputs, x, weights, g
        g = Graph(1, request.verbose, request.max_batch_size)
        last = g.input("hidden", (-1, text["hidden_size"]), axes={0: "batch"})
        last = g.norm(last, _load_tensor(readers, base + "norm.weight"))
        with safe_open(request.model_dir / "readout.safetensors", framework="numpy") as reader:
            readout = reader.get_tensor("weight")
        if readout.shape != (255, text["hidden_size"]) or readout.dtype != np.float32:
            raise ValueError("D3 requires FP32 readout [255,hidden_size]")
        matrix = g.constant(readout.T)
        logits = g.net.add_matrix_multiply(g.cast(last, trt.float32), trt.MatrixOperation.NONE,
                                          matrix, trt.MatrixOperation.NONE).get_output(0)
        writer.add_bytes("readout.plan", g.finish({"logits": logits}))
        vision = VisionConfig.from_dict(config["vision_config"], text["hidden_size"])
        visual_readers = readers
        # The visual loader uses the canonical model.visual prefix internally.
        visual_weights = load_vision_weights(request.model_dir, vision, readers=visual_readers,
                                             prefix=prefix + "visual.")
        writer.add_bytes("vision.plan", build_vision_engine(vision, visual_weights,
                                                           height * width // 256 * request.max_batch_size,
                                                           patches_per_image=height * width // 256,
                                                           verbose=request.verbose))
    writer.add_bytes("tokenizer.json", (request.model_dir / "tokenizer.json").read_bytes())
    writer.add_json("runtime.json", {"version": 3, "model_name": "d3-" + profile_name(config), "precision": "bf16", "max_sequence_length": maximum,
                                    "max_batch_size": request.max_batch_size,
                                    "image_height": height, "image_width": width,
                                    "config": config, "decision": decision})
