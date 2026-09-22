# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSTU lookup, SiLU attention, and ranking head as a native TensorRT graph.

Reference: NVIDIA/recsys-examples 97062d97eef53115105063801e35184e36186df5,
examples/hstu. Python is used to build the bundle, never to execute it.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import uuid

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

    def __init__(self, trt, network, config, weights, precision, *, prefill=False):
        self.trt, self.net, self.config = trt, network, config
        self.weights = dict(weights)
        # Canonical checkpoints store [head, UVQK, channel]. Put each complete
        # U/V/Q/K projection together at build time so runtime uses slices.
        h, d, e = config["num_heads"], config["head_dim"], config["hidden_size"]
        for index in range(config["num_layers"]):
            prefix = f"blocks.{index}.uvqk"
            self.weights[f"{prefix}.weight"] = np.ascontiguousarray(
                weights[f"{prefix}.weight"]
                .reshape(h, 4, d, e)
                .transpose(1, 0, 2, 3)
                .reshape(4 * h * d, e)
            )
            if config["add_uvqk_bias"]:
                self.weights[f"{prefix}.bias"] = np.ascontiguousarray(
                    weights[f"{prefix}.bias"].reshape(h, 4, d).transpose(1, 0, 2).reshape(4 * h * d)
                )
        self.dtype = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}[precision]
        self.host_weights = []
        self.prefill = prefill

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

    def last_axis_slice(self, value, offset, width, name):
        rank = len(value.shape)
        shape = self.out(self.net.add_shape(value), f"{name}.source_shape")
        front = self.out(self.net.add_slice(shape, (0,), (rank - 1,), (1,)), f"{name}.leading_dims")
        tail = self.constant(np.array([width], dtype=np.int64), f"{name}.width")
        size = self.net.add_concatenation([front, tail])
        size.axis = 0
        layer = self.net.add_slice(value, (0,) * (rank - 1) + (offset,), (1,) * rank, (1,) * rank)
        layer.set_input(2, self.out(size, f"{name}.shape"))
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

    def silu(self, value, name, *, model_dtype=False):
        # Attention keeps its explicit FP32 exponential. Projection activations
        # use the model dtype to allow TensorRT's native GEMM/SiLU fusion.
        fp32 = value if model_dtype else self.cast(value, self.trt.float32, f"{name}.fp32")
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

    def block(self, value, mask, scale, index, cache=None):
        c, prefix = self.config, f"blocks.{index}"
        d, h = c["head_dim"], c["num_heads"]
        x = self.norm(value, f"{prefix}.input_norm", c["learnable_input_layernorm"])
        uvqk = self.silu(
            self.linear(x, f"{prefix}.uvqk", c["add_uvqk_bias"]),
            f"{prefix}.silu",
            model_dtype=True,
        )
        parts = [
            self.last_axis_slice(uvqk, offset * h * d, h * d, f"{prefix}.{name}")
            for offset, name in enumerate(("u", "v", "q", "k"))
        ]
        u = parts[0]
        v, q, k = [
            self.reshape(part, (0, 0, h, d), f"{prefix}.{name}.transpose", (0, 2, 1, 3))
            for part, name in zip(parts[1:], ("v", "q", "k"))
        ]
        if cache is not None:
            from .cache_graph import add_cached_kv

            k, v = add_cached_kv(
                self.trt,
                self.net,
                k,
                v,
                c["max_sequence_length"],
                cache["write_indices"],
                cache["active_mask"],
                str(index),
                self.host_weights,
                packed_rows=cache["rows"],
                update_lengths=cache["lengths"],
                attention_from_updates=self.prefill,
                attention_length=self.out(
                    self.net.add_slice(
                        self.out(self.net.add_shape(mask), f"{prefix}.attention_shape"),
                        (2,),
                        (1,),
                        (1,),
                    ),
                    f"{prefix}.attention_length",
                ),
            )
        scores = self.out(
            self.net.add_matrix_multiply(
                k, self.trt.MatrixOperation.NONE, q, self.trt.MatrixOperation.TRANSPOSE
            ),
            f"{prefix}.qk",
        )
        scores = self.binary(
            scores, self.scalar(d**-0.5, scores, f"{prefix}.alpha"), "PROD", f"{prefix}.scaled_qk"
        )
        # Evaluate the same attention as (V^T * (SiLU(K * Q^T) * W^T))^T.
        # This layout avoids a costly first-layer TensorRT fusion while retaining
        # the FP32 SiLU boundary. W already contains visibility and sequence scale.
        scores = self.silu(scores, f"{prefix}.attention_silu")
        scores = self.binary(
            scores, self.cast(mask, scores.dtype, f"{prefix}.mask.cast"), "PROD", f"{prefix}.mask"
        )
        attention = self.out(
            self.net.add_matrix_multiply(
                v, self.trt.MatrixOperation.TRANSPOSE, scores, self.trt.MatrixOperation.NONE
            ),
            f"{prefix}.attention",
        )
        layer = self.net.add_shuffle(attention)
        layer.first_transpose = (0, 3, 1, 2)
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
        cached = c["enable_history_cache"]
        ids = self.net.add_input("token_ids", self.trt.int32, (-1, -1))
        capacity = c["max_sequence_length"]
        mask = self.net.add_input("attention_weights_transposed", self.dtype, (-1, 1, -1, -1))
        cache = None
        if cached:
            active_lengths = self.net.add_input("cache_active_lengths", self.trt.int32, (-1,))
            active_lengths = self.reshape(active_lengths, (0, 1, 1, 1), "cache.active_lengths")
            key_positions = self.constant(
                np.arange(capacity, dtype=np.int32).reshape(1, 1, 1, capacity),
                "cache.key_positions",
            )
            cache = {
                "write_indices": self.net.add_input("cache_write_indices", self.trt.int32, (-1,)),
                "rows": None
                if self.prefill
                else self.net.add_input("cache_update_rows", self.trt.int32, (-1,)),
                "lengths": None
                if self.prefill
                else self.net.add_input("cache_update_lengths", self.trt.int32, (-1,)),
                "active_mask": self.binary(key_positions, active_lengths, "LESS", "cache.active"),
            }
        scale = None  # Scaling is carried by the request's attention weights.
        merged = np.concatenate(
            [self.weights[f"embeddings.{t['name']}.weight"] for t in c["embedding_tables"]]
        )
        table = self.constant(merged, "embedding.weight", self.dtype)
        raw = self.out(self.net.add_gather(table, ids, 0), "embedding.lookup")
        x = raw
        if c["position_buckets"]:
            # Both upstream paths multiply by the full FP32 scale. In the
            # timestamp path, PyTorch materializes that product in model dtype
            # before the separate positional addition. Position-only encoding
            # keeps the product in FP32 until the combined result is stored.
            x = self.cast(x, self.trt.float32, "position.input.fp32")
            x = self.binary(
                x,
                self.scalar(c["hidden_size"] ** 0.5, x, "position.scale"),
                "PROD",
                "position.scaled_embeddings",
            )
            if c["time_buckets"]:
                x = self.cast(x, self.dtype, "position.scaled_output")
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
            x = self.block(x, mask, scale, index, cache)
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


def _engine(config, weights, precision, max_batch_size, verbose, *, prefill=False):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(trt, network, config, weights, precision, prefill=prefill)
    graph.outputs()
    profile = builder.create_optimization_profile()
    max_s = config["max_sequence_length"]
    opt_b, opt_s = min(max_batch_size, 4), min(max_s, 128)
    for index in range(network.num_inputs):
        value = network.get_input(index)
        if value.name == "scaling_seqlen":
            continue
        if value.name == "attention_weights_transposed":
            min_k, opt_k = 1, opt_s
            shapes = (1, 1, min_k, 1), (opt_b, 1, opt_k, opt_s), (max_batch_size, 1, max_s, max_s)
        elif value.name.startswith("cache_") and value.name.endswith(("_k", "_v")):
            tail = (config["num_heads"], max_s, config["head_dim"])
            shapes = (1, *tail), (opt_b, *tail), (max_batch_size, *tail)
        elif value.name == "cache_update_rows":
            shapes = (0,), (opt_b * opt_s,), (max_batch_size * max_s,)
        elif value.name == "cache_update_lengths":
            shapes = (2,), (opt_b + 1,), (max_batch_size + 1,)
        elif value.name in ("cache_active_lengths", "cache_write_indices"):
            shapes = (1,), (opt_b,), (max_batch_size,)
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


def _candidate_engine(config, weights, max_batch_size, verbose):
    """Lookup-only retrieval path when an entire history is already cached."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = _Graph(trt, network, config, weights, "fp32")
    ids = network.add_input("candidate_token_ids", trt.int32, (-1, -1))
    merged = np.concatenate(
        [weights[f"embeddings.{table['name']}.weight"] for table in config["embedding_tables"]]
    )
    table = graph.constant(merged, "candidate.weight")
    values = graph.out(network.add_gather(table, ids, 0), "candidate.lookup")
    output = graph.l2(values, "candidate.l2")
    output.name = "item_embeddings"
    network.mark_output(output)
    profile = builder.create_optimization_profile()
    maximum = config["max_sequence_length"]
    profile.set_shape(
        "candidate_token_ids",
        (1, 1),
        (min(max_batch_size, 4), min(maximum, 256)),
        (max_batch_size, maximum),
    )
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build HSTU candidate lookup engine")
    return bytes(plan)


def _native_attention(config, weights, precision, max_batch_size, verbose, model_dir, backend):
    from .native_attention_build import build_attention_library, source_directory, unsupported_reason

    choice = config["attention_implementation"]
    if choice == "tensorrt":
        return None
    if backend != "trt":
        if choice == "nvidia_hstu":
            raise ValueError("HSTU native attention requires the standard TensorRT backend")
        return None
    reason = unsupported_reason(config, precision)
    if reason:
        if choice == "nvidia_hstu":
            raise ValueError(reason)
        return None
    hint = config["native_kernel_source"]
    source = source_directory(Path(model_dir) / hint if hint is not None else None)
    if source is None:
        if choice == "nvidia_hstu":
            raise ValueError("Set native_kernel_source to the pinned original HSTU CUDA sources")
        return None
    from .dense_graph import build_dense_engine
    from .paged_graph import build_paged_engine

    with tempfile.TemporaryDirectory(prefix="trtmc-hstu-build-") as directory:
        mode = "paged" if config["enable_history_cache"] else "dense"
        library, manifest = build_attention_library(Path(directory), source,
                                                    attention_mode=mode, verbose=verbose)
        build_engine = build_paged_engine if config["enable_history_cache"] else build_dense_engine
        plan = build_engine(config, weights, max_batch_size, library, manifest["namespace"], verbose)
        notices = (Path(directory) / "attention_native.NOTICE").read_bytes()
        return plan, library.read_bytes(), manifest, notices


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
    runtime.pop("native_kernel_source", None)
    runtime["precision"] = request.precision
    if config["enable_history_cache"]:
        # A bundle build is a cache namespace. Rebuilt weights/configuration can
        # never consume another artifact's KV, even if caller labels are reused.
        runtime["cache_artifact_id"] = uuid.uuid4().hex
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
    native = _native_attention(config, weights, request.precision, request.max_batch_size,
                               request.verbose, request.model_dir, request.backend)
    plan = native[0] if native else _engine(
        config, weights, request.precision, request.max_batch_size, request.verbose
    )
    runtime["attention_implementation"] = "nvidia_hstu" if native else "tensorrt"
    writer.set_header(family="hstu", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    writer.add_bytes("engine.plan", plan)
    if native:
        writer.add_bytes("attention_native.so", native[1])
        writer.add_json("attention_native.json", native[2])
        writer.add_bytes("attention_native.NOTICE", native[3])
    if config["enable_history_cache"] and not native:
        writer.add_bytes(
            "prefill.plan",
            _engine(
                config,
                weights,
                request.precision,
                request.max_batch_size,
                request.verbose,
                prefill=True,
            ),
        )
    if config["enable_history_cache"] and config["mode"] == "retrieval":
        writer.add_bytes(
            "candidate.plan",
            _candidate_engine(config, weights, request.max_batch_size, request.verbose),
        )
    if key_bytes:
        writer.add_bytes("embedding_keys.bin", b"".join(key_bytes))
