# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU lookup, SiLU attention, and ranking head as a native TensorRT graph.

Reference: NVIDIA/recsys-examples 97062d97eef53115105063801e35184e36186df5,
examples/hstu. Python is used to build the bundle, never to execute it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import expected_shapes, load_config


def load_weights(model_dir: Path, config: dict) -> dict[str, np.ndarray]:
    from safetensors import safe_open

    expected = expected_shapes(config)
    key_shapes = {
        f"embeddings.{table['name']}.keys": (table["num_embeddings"],)
        for table in config["embedding_tables"]
    }
    weights = {}
    # The converter writes FP32 arrays, including BF16 sources promoted directly.
    with safe_open(model_dir / "model.safetensors", framework="numpy") as reader:
        keys = set(reader.keys())
        if expected.keys() - keys or keys - (expected.keys() | key_shapes.keys()):
            raise ValueError(
                f"HSTU checkpoint missing={sorted(expected.keys() - keys)}, "
                f"unexpected={sorted(keys - (expected.keys() | key_shapes.keys()))}"
            )
        for name in keys:
            array = reader.get_tensor(name)
            shape = expected.get(name, key_shapes.get(name))
            if array.shape != shape:
                raise ValueError(f"HSTU {name} shape {array.shape}, expected {shape}")
            if name in key_shapes:
                if array.dtype != np.int64 or np.any(array[1:] <= array[:-1]):
                    raise ValueError(f"HSTU {name} must contain strictly increasing INT64 keys")
            else:
                array = np.ascontiguousarray(array, dtype=np.float32)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"HSTU {name} contains non-finite weights")
            weights[name] = array
    return weights


class _Graph:
    """Family-local TensorRT graph operations with explicit dtype boundaries."""

    def __init__(self, trt, network, config, weights, precision):
        self.trt, self.net, self.config, self.weights = trt, network, config, weights
        self.dtype = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}[precision]
        self.host_weights = []

    def out(self, layer, name):
        if layer is None:
            raise RuntimeError(f"TensorRT rejected HSTU layer {name}")
        layer.name = name
        return layer.get_output(0)

    def constant(self, data, name, dtype=None):
        array = np.ascontiguousarray(data)
        self.host_weights.append(array)
        value = self.out(self.net.add_constant(array.shape, self.trt.Weights(array)), name)
        return self.cast(value, dtype, f"{name}.cast") if dtype is not None else value

    def cast(self, value, dtype, name):
        return value if value.dtype == dtype else self.out(self.net.add_cast(value, dtype), name)

    def scalar(self, value, like, name):
        return self.constant(np.full((1,) * len(like.shape), value, np.float32), name, like.dtype)

    def binary(self, left, right, operation, name):
        return self.out(
            self.net.add_elementwise(
                left, right, getattr(self.trt.ElementWiseOperation, operation)
            ),
            name,
        )

    def unary(self, value, operation, name):
        return self.out(
            self.net.add_unary(value, getattr(self.trt.UnaryOperation, operation)), name
        )

    def reshape(self, value, shape, name, transpose=None):
        layer = self.net.add_shuffle(value)
        layer.reshape_dims = shape
        if transpose is not None:
            layer.second_transpose = transpose
        return self.out(layer, name)

    def linear(self, value, prefix, bias=True):
        matrix = self.weights[f"{prefix}.weight"]
        shape = (1,) * (len(value.shape) - 2) + matrix.shape
        weight = self.constant(matrix.reshape(shape), f"{prefix}.weight", value.dtype)
        output = self.out(
            self.net.add_matrix_multiply(
                value, self.trt.MatrixOperation.NONE, weight, self.trt.MatrixOperation.TRANSPOSE
            ),
            prefix,
        )
        if bias:
            vector = self.weights[f"{prefix}.bias"].reshape((1,) * (len(value.shape) - 1) + (-1,))
            output = self.binary(
                output,
                self.constant(vector, f"{prefix}.bias", value.dtype),
                "SUM",
                f"{prefix}.add_bias",
            )
        return output

    def silu(self, value, name):
        # PyTorch SiLU evaluates its exponential in FP32 registers.
        fp32 = self.cast(value, self.trt.float32, f"{name}.fp32")
        sigmoid = self.out(
            self.net.add_activation(fp32, self.trt.ActivationType.SIGMOID), f"{name}.sigmoid"
        )
        output = self.binary(fp32, sigmoid, "PROD", name)
        return self.cast(output, value.dtype, f"{name}.output")

    def norm(self, value, prefix, learned, *, preserve_fp32=False):
        x = self.cast(value, self.trt.float32, f"{prefix}.fp32")
        axes = 1 << (len(x.shape) - 1)
        mean = self.out(
            self.net.add_reduce(x, self.trt.ReduceOperation.AVG, axes, True), f"{prefix}.mean"
        )
        centered = self.binary(x, mean, "SUB", f"{prefix}.center")
        square = self.binary(centered, centered, "PROD", f"{prefix}.square")
        variance = self.out(
            self.net.add_reduce(square, self.trt.ReduceOperation.AVG, axes, True),
            f"{prefix}.variance",
        )
        variance = self.binary(
            variance,
            self.scalar(self.config["layer_norm_epsilon"], x, f"{prefix}.epsilon"),
            "SUM",
            f"{prefix}.stabilize",
        )
        output = self.binary(
            centered, self.unary(variance, "SQRT", f"{prefix}.stddev"), "DIV", prefix
        )
        if learned:
            for suffix, operation in (("weight", "PROD"), ("bias", "SUM")):
                array = self.weights[f"{prefix}.{suffix}"].reshape(
                    (1,) * (len(value.shape) - 1) + (-1,)
                )
                affine = self.constant(array, f"{prefix}.{suffix}", self.dtype)
                affine = self.cast(affine, self.trt.float32, f"{prefix}.{suffix}.fp32")
                output = self.binary(output, affine, operation, f"{prefix}.{suffix}.apply")
        return output if preserve_fp32 else self.cast(output, value.dtype, f"{prefix}.output")

    def l2(self, value, name):
        fp32 = self.cast(value, self.trt.float32, f"{name}.fp32")
        square = self.binary(fp32, fp32, "PROD", f"{name}.square")
        total = self.out(
            self.net.add_reduce(
                square, self.trt.ReduceOperation.SUM, 1 << (len(value.shape) - 1), True
            ),
            f"{name}.sum",
        )
        norm = self.cast(
            self.unary(total, "SQRT", f"{name}.norm"), value.dtype, f"{name}.norm.cast"
        )
        norm = self.binary(
            norm,
            self.scalar(self.config["output_norm_epsilon"], value, f"{name}.epsilon"),
            "MAX",
            f"{name}.clamp",
        )
        return self.binary(value, norm, "DIV", name)

    def block(self, value, mask, scale, index):
        c, prefix = self.config, f"blocks.{index}"
        d, h = c["head_dim"], c["num_heads"]
        x = self.norm(value, f"{prefix}.input_norm", c["learnable_input_layernorm"])
        uvqk = self.silu(self.linear(x, f"{prefix}.uvqk", c["add_uvqk_bias"]), f"{prefix}.silu")
        packed = self.reshape(uvqk, (0, 0, h, 4 * d), f"{prefix}.heads")
        parts = []
        for offset, name in enumerate(("u", "v", "q", "k")):
            indices = self.constant(
                np.arange(offset * d, (offset + 1) * d, dtype=np.int32), f"{prefix}.{name}.indices"
            )
            part = self.out(self.net.add_gather(packed, indices, 3), f"{prefix}.{name}")
            parts.append(part)
        u = self.reshape(parts[0], (0, 0, h * d), f"{prefix}.u.flatten")
        v, q, k = [
            self.reshape(p, (0, 0, h, d), f"{prefix}.{name}.transpose", (0, 2, 1, 3))
            for p, name in zip(parts[1:], ("v", "q", "k"))
        ]
        scores = self.out(
            self.net.add_matrix_multiply(
                q, self.trt.MatrixOperation.NONE, k, self.trt.MatrixOperation.TRANSPOSE
            ),
            f"{prefix}.qk",
        )
        scores = self.binary(
            scores, self.scalar(d**-0.5, scores, f"{prefix}.alpha"), "PROD", f"{prefix}.scaled_qk"
        )
        scores = self.silu(scores, f"{prefix}.attention_silu")
        scores = self.binary(
            scores, self.cast(scale, scores.dtype, f"{prefix}.scale.cast"), "DIV", f"{prefix}.scale"
        )
        scores = self.binary(
            scores, self.cast(mask, scores.dtype, f"{prefix}.mask.cast"), "PROD", f"{prefix}.mask"
        )
        attention = self.out(
            self.net.add_matrix_multiply(
                scores, self.trt.MatrixOperation.NONE, v, self.trt.MatrixOperation.NONE
            ),
            f"{prefix}.attention",
        )
        layer = self.net.add_shuffle(attention)
        layer.first_transpose = (0, 2, 1, 3)
        layer.reshape_dims = (0, 0, h * d)
        attention = self.out(layer, f"{prefix}.attention.flatten")
        attention = self.norm(
            attention, f"{prefix}.output_norm", c["learnable_output_layernorm"], preserve_fp32=True
        )
        gated = self.binary(
            self.cast(u, self.trt.float32, f"{prefix}.u.fp32"), attention, "PROD", f"{prefix}.gate"
        )
        gated = self.cast(gated, value.dtype, f"{prefix}.gate.output")
        projected = self.linear(gated, f"{prefix}.proj", False)
        return (
            self.binary(projected, value, "SUM", f"{prefix}.residual")
            if c["residual"]
            else projected
        )

    def outputs(self):
        c = self.config
        ids = self.net.add_input("token_ids", self.trt.int32, (-1, -1))
        mask = self.net.add_input("attention_mask", self.trt.float32, (-1, 1, -1, -1))
        scale = self.reshape(
            self.net.add_input("scaling_seqlen", self.trt.float32, (1,)),
            (1, 1, 1, 1),
            "scale.broadcast",
        )
        merged = np.concatenate(
            [self.weights[f"embeddings.{t['name']}.weight"] for t in c["embedding_tables"]]
        )
        table = self.constant(merged, "embedding.weight", self.dtype)
        raw = self.out(self.net.add_gather(table, ids, 0), "embedding.lookup")
        x = raw
        if c["position_buckets"]:
            # The position-only upstream kernel scales and adds in FP32 registers.
            if not c["time_buckets"]:
                x = self.cast(x, self.trt.float32, "position.input.fp32")
            x = self.binary(
                x,
                self.scalar(c["hidden_size"] ** 0.5, x, "position.scale"),
                "PROD",
                "position.scaled_embeddings",
            )
            position_parts = []
            for name, enabled in (("position", c["position_buckets"]), ("time", c["time_buckets"])):
                if enabled:
                    indices = self.net.add_input(f"{name}_ids", self.trt.int32, (-1, -1))
                    weight = self.constant(
                        self.weights[f"{name}.weight"], f"{name}.weight", self.dtype
                    )
                    embeddings = self.out(self.net.add_gather(weight, indices, 0), f"{name}.lookup")
                    position_parts.append(self.cast(embeddings, self.trt.float32, f"{name}.fp32"))
            positional = position_parts[0]
            if c["time_buckets"]:
                positional = self.binary(positional, position_parts[1], "SUM", "position.time.sum")
                positional = self.cast(positional, self.dtype, "position.time.output")
            x = self.binary(x, positional, "SUM", "position.add")
            x = self.cast(x, self.dtype, "position.output")
        for index in range(c["num_layers"]):
            x = self.block(x, mask, scale, index)
        embeddings = self.l2(x, "output.l2")
        outputs = {"embeddings": embeddings}
        if c["mode"] == "retrieval":
            candidate_ids = self.net.add_input("candidate_token_ids", self.trt.int32, (-1, -1))
            # RetrievalGR retains its supervision/item table in FP32.
            candidate_table = self.constant(merged, "candidate.weight")
            candidates = self.out(
                self.net.add_gather(candidate_table, candidate_ids, 0), "candidate.lookup"
            )
            outputs["item_embeddings"] = self.l2(candidates, "items.l2")
        else:
            # Upstream inference converts the prediction MLP to the model dtype.
            logits = embeddings
            for index in range(len(c["prediction_head"])):
                logits = self.linear(logits, f"head.{index}", c["prediction_bias"])
                if index + 1 != len(c["prediction_head"]):
                    activation = (
                        self.trt.ActivationType.RELU
                        if c["prediction_activation"] == "relu"
                        else self.trt.ActivationType.GELU_ERF
                    )
                    logits = self.out(
                        self.net.add_activation(logits, activation), f"head.{index}.activation"
                    )
            outputs["logits"] = logits
        for name, tensor in outputs.items():
            tensor = self.cast(tensor, self.trt.float32, f"{name}.float_output")
            tensor.name = name
            self.net.mark_output(tensor)


def _engine(config, weights, precision, max_batch_size, verbose):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(trt, network, config, weights, precision)
    graph.outputs()
    profile = builder.create_optimization_profile()
    max_s = config["max_sequence_length"]
    opt_b, opt_s = min(max_batch_size, 4), min(max_s, 128)
    for index in range(network.num_inputs):
        value = network.get_input(index)
        if value.name == "scaling_seqlen":
            continue
        if value.name == "attention_mask":
            shapes = (1, 1, 1, 1), (opt_b, 1, opt_s, opt_s), (max_batch_size, 1, max_s, max_s)
        else:
            shapes = (1, 1), (opt_b, opt_s), (max_batch_size, max_s)
        profile.set_shape(value.name, *shapes)
    if not profile:
        raise RuntimeError("TensorRT rejected the HSTU optimization profile")
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    builder_config.add_optimization_profile(profile)
    # HSTU FP32 reference correctness uses IEEE FP32 GEMMs, not TF32 truncation.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the HSTU engine")
    return bytes(plan)


def build(request, writer) -> None:
    """Build HSTU through the public Model Connect bundle contract."""
    if request.task != "recommendation":
        raise ValueError("HSTU requires task=recommendation")
    if request.precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("HSTU precision must be fp32, fp16, or bf16")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("HSTU supports a single-device native engine")
    if request.dynamic_kv_cache:
        raise NotImplementedError("hstu does not support dynamic_kv_cache")
    if request.quantization not in (None, "none") or request.fp32_layers:
        raise ValueError("HSTU does not define quantization or per-layer precision overrides")
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError("HSTU does not accept image or video build options")
    config = load_config(Path(request.model_dir) / "config.json")
    if request.max_sequence_length is not None:
        config["max_sequence_length"] = request.max_sequence_length
    weights = load_weights(Path(request.model_dir), config)
    runtime = {**config, "max_batch_size": request.max_batch_size}
    tables, key_bytes, offset, key_offset = [], [], 0, 0
    for table in config["embedding_tables"]:
        entry = {**table, "offset": offset}
        keys = weights.get(f"embeddings.{table['name']}.keys")
        if keys is not None:
            entry["keys_offset"] = key_offset
            key_bytes.append(keys.astype("<i8").tobytes())
            key_offset += len(keys)
        tables.append(entry)
        offset += table["num_embeddings"]
    runtime["embedding_tables"] = tables
    plan = _engine(config, weights, request.precision, request.max_batch_size, request.verbose)
    writer.set_header(family="hstu", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    writer.add_bytes("engine.plan", plan)
    if key_bytes:
        writer.add_bytes("embedding_keys.bin", b"".join(key_bytes))
