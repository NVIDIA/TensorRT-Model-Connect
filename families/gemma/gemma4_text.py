# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 text topology, expressed only with TensorRT graph operations.

Unlike earlier Gemma decoders, learned RMS scales are direct (not 1 + scale).
Shared K/V layers reuse their owning layer's complete attention tensors; only
owners have persistent cache IO. Cache widths follow each layer's geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors


def text_config(config) -> dict:
    return {**config.raw, **config.raw.get("text_config", {})}


def cache_geometry(raw: dict) -> tuple[list[int], list[int]]:
    """Return per-layer KV owners and widths of the owning cache buffers."""
    layers = int(raw["num_hidden_layers"])
    shared = int(raw.get("num_kv_shared_layers", 0))
    types = raw["layer_types"]
    first_shared = layers - shared
    if len(types) != layers or not 0 <= shared < layers:
        raise ValueError("Gemma4 requires a valid attention schedule and KV sharing point")
    owners, widths, previous = [], [], {}
    for index, kind in enumerate(types):
        if kind not in {"sliding_attention", "full_attention"}:
            raise ValueError(f"Unsupported Gemma4 attention type: {kind}")
        if index < first_shared:
            previous[kind] = index
            head_dim, kv_heads = attention_geometry(raw, kind)
            widths.append(head_dim * kv_heads)
        if kind not in previous:
            raise ValueError(f"Gemma4 shared {kind} has no preceding KV owner")
        owners.append(previous[kind])
    return owners, widths


def attention_geometry(raw: dict, kind: str) -> tuple[int, int]:
    full = kind == "full_attention"
    head_dim = int((raw.get("global_head_dim") if full else None) or raw["head_dim"])
    kv_heads = int(raw["num_global_key_value_heads"] if full and raw.get("attention_k_eq_v")
                   else raw["num_key_value_heads"])
    if head_dim < 2 or head_dim % 2 or kv_heads < 1 or raw["num_attention_heads"] % kv_heads:
        raise ValueError("Gemma4 attention geometry is invalid")
    return head_dim, kv_heads


def rope_tables(raw: dict, kind: str, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-width rotate-half tables, including proportional RoPE's zero angles."""
    head_dim, _ = attention_geometry(raw, kind)
    params = raw["rope_parameters"][kind]
    rope_type = params.get("rope_type", "default")
    if rope_type not in {"default", "proportional"}:
        raise NotImplementedError(f"Gemma4 native RoPE does not implement {rope_type!r}")
    fraction = float(params.get("partial_rotary_factor", 1.0)) if rope_type == "proportional" else 1.0
    factor = float(params.get("factor", 1.0))
    if not 0 < fraction <= 1 or factor <= 0:
        raise ValueError("Gemma4 rotary proportion and factor must be positive")
    count = int(fraction * head_dim // 2)
    inv = np.zeros(head_dim // 2, np.float32)
    inv[:count] = 1.0 / (float(params["rope_theta"]) **
                         (np.arange(count, dtype=np.float32) * 2 / head_dim)) / factor
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(angles), np.sin(angles)


def per_layer_embedding(table: np.ndarray, index: int, width: int) -> np.ndarray:
    """Expose each decoder layer's token embedding as its own vocabulary table."""
    if (table.ndim != 2 or width <= 0 or table.shape[1] % width
            or not 0 <= index < table.shape[1] // width):
        raise ValueError("Gemma4 per-layer embedding dimensions are inconsistent")
    return np.ascontiguousarray(table[:, index * width:(index + 1) * width])


class DenseTextGraph:
    def __init__(self, network, raw, readers, precision, capacity, conditioned=False, paired=False, feature_layers=()):
        self.net, self.raw, self.readers = network, raw, readers
        self.capacity = capacity
        self.conditioned = conditioned
        self.paired = paired
        self.feature_layers = tuple(feature_layers)
        if (len(set(self.feature_layers)) != len(self.feature_layers) or
                any(type(i) is not int or not 0 <= i < raw["num_hidden_layers"] - 1
                    for i in self.feature_layers)):
            raise ValueError("Gemma draft features require distinct nonfinal target layers")
        self.dtype = {"fp16": trt.float16, "bf16": trt.bfloat16, "fp32": trt.float32}[precision]
        # BF16 source values must not round through FP16 first.
        self.storage = np.float16 if precision == "fp16" else np.float32
        self.buffers = []
        self.prefix = next((p for p in ("model.language_model", "language_model.model", "model")
                            if _has_tensor(readers, f"{p}.embed_tokens.weight")), None)
        if self.prefix is None:
            raise ValueError("Gemma4 checkpoint has no recognized text embedding")
        self.owners, self.widths = cache_geometry(raw)
        if raw.get("enable_moe_block"):
            from .gemma4_moe import expert_geometry

            expert_geometry(raw)

    def cast(self, tensor, dtype):
        return tensor if tensor.dtype == dtype else self.net.add_cast(tensor, dtype).get_output(0)

    def const(self, values, dtype=None):
        values = np.ascontiguousarray(values)
        self.buffers.append(values)  # TensorRT weight storage lives through serialization.
        value = self.net.add_constant(values.shape, trt.Weights(values)).get_output(0)
        return self.cast(value, dtype) if dtype is not None else value

    def weight(self, name, *, dtype=None):
        value = _load_tensor(self.readers, f"{self.prefix}.{name}")
        return self.const(value.astype(self.storage if dtype is None else np.float32),
                          self.dtype if dtype is None else dtype)

    def reshape(self, tensor, shape):
        layer = self.net.add_shuffle(tensor)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def op(self, lhs, rhs, op):
        return self.net.add_elementwise(lhs, rhs, op).get_output(0)

    def scale(self, value, factor):
        constant = self.const(np.full((1,) * len(value.shape), factor, np.float32), value.dtype)
        return self.op(value, constant, trt.ElementWiseOperation.PROD)

    def linear(self, value, name):
        if _has_tensor(self.readers, f"{self.prefix}.{name}.weight_scale_2"):
            from .gemma4_quant import dense_linear

            output = dense_linear(self, value, name)
        else:
            weight = self.weight(name + ".weight")
            weight = self.cast(weight, value.dtype)
            output = self.net.add_matrix_multiply(
                value, trt.MatrixOperation.NONE, weight, trt.MatrixOperation.TRANSPOSE).get_output(0)
        if _has_tensor(self.readers, f"{self.prefix}.{name}.bias"):
            bias = self.reshape(self.weight(name + ".bias", dtype=trt.float32), (1, -1))
            output = self.op(output, self.cast(bias, output.dtype), trt.ElementWiseOperation.SUM)
        return output

    def norm(self, value, name=None):
        original_dtype = value.dtype
        fp32 = self.cast(value, trt.float32)
        rank = len(value.shape)
        squared = self.op(fp32, fp32, trt.ElementWiseOperation.PROD)
        mean = self.net.add_reduce(squared, trt.ReduceOperation.AVG, 1 << (rank - 1), True).get_output(0)
        eps = self.const(np.full((1,) * rank, self.raw["rms_norm_eps"], np.float32))
        exponent = self.const(np.full((1,) * rank, -0.5, np.float32))
        inv = self.op(self.op(mean, eps, trt.ElementWiseOperation.SUM), exponent,
                      trt.ElementWiseOperation.POW)
        output = self.op(fp32, inv, trt.ElementWiseOperation.PROD)
        if name is not None:
            weight = self.reshape(self.weight(name + ".weight", dtype=trt.float32),
                                  (1,) * (rank - 1) + (value.shape[-1],))
            output = self.op(output, weight, trt.ElementWiseOperation.PROD)
        return self.cast(output, original_dtype)

    def head_norm(self, value, heads, dim, name=None):
        shaped = self.reshape(value, (-1, heads, dim))
        return self.reshape(self.norm(shaped, name), (-1, heads * dim))

    def activation(self, value):
        return graph_ops.add_activation(self.net, value, self.raw["hidden_activation"],
                                        dtype=np.float32 if value.dtype == trt.float32 else self.storage)

    def attention(self, q, k, v, heads, kv_heads, dim, mask):
        # Preserve grouped attention as mathematical primitives. TensorRT owns
        # fusion; the graph does not require any GPU-specific attention kernel.
        n = self.net
        def grouped(value, count):
            value = graph_ops.reshape_rows_to_heads_4d(
                n, self.cast(value, trt.float32), count, dim, sequence_length=None)
            return self.reshape(value, (kv_heads, count // kv_heads, -1, dim))
        scores = n.add_matrix_multiply(grouped(q, heads), trt.MatrixOperation.NONE,
                                       grouped(k, kv_heads), trt.MatrixOperation.TRANSPOSE).get_output(0)
        # Gemma4's attention score scale is exactly 1, not 1/sqrt(head_dim).
        scores = self.op(scores, mask, trt.ElementWiseOperation.SUM)
        probabilities = n.add_softmax(scores)
        probabilities.axes = 1 << 3
        context = n.add_matrix_multiply(
            probabilities.get_output(0), trt.MatrixOperation.NONE,
            grouped(v, kv_heads), trt.MatrixOperation.NONE).get_output(0)
        context = self.reshape(context, (1, heads, -1, dim))
        return self.cast(graph_ops.reshape_heads_4d_to_rows(
            n, context, heads * dim, sequence_length=None), self.dtype)

    def build(self):
        n, raw = self.net, self.raw
        hidden, layers = int(raw["hidden_size"]), int(raw["num_hidden_layers"])
        heads = int(raw["num_attention_heads"])
        tokens = n.add_input("token_id", trt.int32, (-1,))
        positions = n.add_input("position_id", trt.int32, (-1,))
        mask = n.add_input("attention_mask", trt.float32, (-1, -1))
        if self.conditioned:
            media_features = n.add_input("media_features", trt.float32, (-1, hidden))
            media_blocks = n.add_input("media_mask", trt.int32, (-1,))
            # Nonzero IDs replace embeddings; only positive IDs denote vision.
            media_valid = self.cast(media_blocks, trt.bool)
            tokens = n.add_select(media_valid, self.const(np.array([raw["pad_token_id"]], np.int32)),
                                   tokens).get_output(0)
        # Runtime caches use absolute sequence positions. Hide keys older than
        # the sliding window, in addition to the runtime's causal/validity mask.
        cache_pos = self.const(np.arange(self.capacity, dtype=np.int32))
        all_pos_layer = n.add_concatenation([cache_pos, positions])
        all_pos_layer.axis = 0
        key_pos = self.reshape(all_pos_layer.get_output(0), (1, -1))
        query_pos = self.reshape(positions, (-1, 1))
        lower_bound = self.op(query_pos, self.const(np.array([[raw["sliding_window"]]], np.int32)),
                              trt.ElementWiseOperation.SUB)
        in_window = self.op(key_pos, lower_bound, trt.ElementWiseOperation.GREATER)
        masked = self.const(np.array([[-np.inf]], np.float32))

        local_source = mask
        if self.conditioned and raw.get("use_bidirectional_attention") == "vision":
            key_blocks = n.add_concatenation([
                self.const(np.zeros(self.capacity, np.int32)), media_blocks])
            key_blocks.axis = 0
            same = self.op(self.reshape(media_blocks, (-1, 1)),
                           self.reshape(key_blocks.get_output(0), (1, -1)),
                           trt.ElementWiseOperation.EQUAL)
            vision_valid = self.op(media_blocks, self.const(np.array([0], np.int32)),
                                   trt.ElementWiseOperation.GREATER)
            same = self.op(same, self.reshape(vision_valid, (-1, 1)), trt.ElementWiseOperation.AND)
            local_source = n.add_select(same, self.const(np.zeros((1, 1), np.float32)),
                                        mask).get_output(0)
        local_mask = n.add_select(in_window, local_source, masked).get_output(0)
        masks = {}
        # Explicit reshapes below preserve the two dynamic mask axes.
        for kind, source in (("full_attention", mask), ("sliding_attention", local_mask)):
            shape = n.add_concatenation([self.const(np.array([1, 1], np.int64)),
                                         n.add_shape(source).get_output(0)])
            shape.axis = 0
            reshape = n.add_shuffle(source)
            reshape.set_input(1, shape.get_output(0))
            masks[kind] = reshape.get_output(0)
        embedding = self.weight("embed_tokens.weight")
        state = self.scale(n.add_gather(embedding, tokens, 0).get_output(0), hidden**0.5)
        if self.conditioned:
            state = n.add_select(self.reshape(media_valid, (-1, 1)),
                                 self.cast(media_features, self.dtype), state).get_output(0)
        ple_dim = int(raw.get("hidden_size_per_layer_input", 0))
        ple = None
        if ple_dim:
            # Keep the token table layer-local rather than giving Gather one
            # flattened multi-gigabyte table. The values and model math are
            # unchanged; TensorRT still owns all execution and fusion.
            ple_embedding = _load_tensor(self.readers, f"{self.prefix}.embed_tokens_per_layer.weight")
            if ple_embedding.shape != (raw["vocab_size_per_layer_input"], layers * ple_dim):
                raise ValueError("Gemma4 per-layer embedding shape does not match its config")
            projected = self.scale(self.linear(state, "per_layer_model_projection"), hidden**-0.5)
            projected = self.reshape(projected, (-1, layers, ple_dim))
            ple = self.norm(projected, "per_layer_projection_norm")
        tables = {kind: tuple(self.const(x.astype(self.storage), self.dtype)
                              for x in rope_tables(raw, kind, self.capacity))
                  for kind in set(raw["layer_types"])}
        kv, features = {}, {}
        for index, kind in enumerate(raw["layer_types"]):
            prefix = f"layers.{index}"
            head_dim, kv_heads = attention_geometry(raw, kind)
            normalized = self.norm(state, prefix + ".input_layernorm")
            q = self.head_norm(self.linear(normalized, prefix + ".self_attn.q_proj"),
                               heads, head_dim, prefix + ".self_attn.q_norm")
            cos, sin = tables[kind]
            q = graph_ops.add_apply_rope_native(n, q, heads, head_dim, cos, sin, positions,
                                               head_dim, sequence_length=None)
            owner = self.owners[index]
            if owner == index:
                k_raw = self.linear(normalized, prefix + ".self_attn.k_proj")
                v_raw = (k_raw if kind == "full_attention" and raw.get("attention_k_eq_v")
                         else self.linear(normalized, prefix + ".self_attn.v_proj"))
                k = self.head_norm(k_raw, kv_heads, head_dim, prefix + ".self_attn.k_norm")
                k = graph_ops.add_apply_rope_native(n, k, kv_heads, head_dim, cos, sin, positions,
                                                   head_dim, sequence_length=None)
                v = self.head_norm(v_raw, kv_heads, head_dim)
                combined = []
                for label, value in (("k", k), ("v", v)):
                    cache = n.add_input(f"cache_{label}_{index}", self.dtype,
                                        (self.capacity, self.widths[index]))
                    value.name = f"present_{label}_{index}"
                    n.mark_output(value)
                    concat = n.add_concatenation([cache, value])
                    concat.axis = 0
                    combined.append(concat.get_output(0))
                kv[index] = combined
            k, v = kv[owner]
            context = self.attention(q, k, v, heads, kv_heads, head_dim, masks[kind])
            attn = self.norm(self.linear(context, prefix + ".self_attn.o_proj"),
                             prefix + ".post_attention_layernorm")
            state = self.op(state, attn, trt.ElementWiseOperation.SUM)
            normalized = self.norm(state, prefix + ".pre_feedforward_layernorm")
            gate = self.linear(normalized, prefix + ".mlp.gate_proj")
            up = self.linear(normalized, prefix + ".mlp.up_proj")
            # The original model has no clamp. Keep the nonlinear product and
            # down projection in FP32 rather than copying Edge's FP16 clipping.
            mlp = self.op(self.activation(self.cast(gate, trt.float32)), self.cast(up, trt.float32),
                          trt.ElementWiseOperation.PROD)
            mlp = self.linear(mlp, prefix + ".mlp.down_proj")
            if raw.get("enable_moe_block"):
                from .gemma4_moe import add_experts

                mlp = add_experts(self, state, mlp, prefix)
            mlp = self.norm(mlp, prefix + ".post_feedforward_layernorm")
            state = self.op(state, self.cast(mlp, self.dtype), trt.ElementWiseOperation.SUM)
            if ple is not None:
                per_layer = n.add_gather(ple, self.const(np.array([index], np.int32)), 1).get_output(0)
                per_layer = self.reshape(per_layer, (-1, ple_dim))
                layer_embedding = self.const(
                    per_layer_embedding(ple_embedding, index, ple_dim).astype(self.storage), self.dtype)
                token_ple = self.scale(n.add_gather(layer_embedding, tokens, 0).get_output(0),
                                       ple_dim**0.5)
                per_layer = self.scale(self.op(per_layer, token_ple, trt.ElementWiseOperation.SUM),
                                       2**-0.5)
                gate = self.activation(self.linear(state, prefix + ".per_layer_input_gate"))
                addition = self.op(gate, per_layer, trt.ElementWiseOperation.PROD)
                addition = self.linear(addition, prefix + ".per_layer_projection")
                state = self.op(state, self.norm(addition, prefix + ".post_per_layer_input_norm"),
                                trt.ElementWiseOperation.SUM)
            state = self.op(state, self.reshape(self.weight(prefix + ".layer_scalar"), (1, 1)),
                            trt.ElementWiseOperation.PROD)
            if index in self.feature_layers:
                features[index] = self.cast(state, trt.float32)
        if self.feature_layers:
            feature_output = n.add_concatenation([features[i] for i in self.feature_layers])
            feature_output.axis = 1
            feature_output.get_output(0).name = "target_features"
            n.mark_output(feature_output.get_output(0))
        state = self.norm(state, "norm")
        if self.paired:
            indices = n.add_input("logits_indices", trt.int32, (-1,))
            selected = n.add_gather(state, indices, 0).get_output(0)
            exported = self.cast(selected, trt.float32)
            exported.name = "target_hidden"
            n.mark_output(exported)
        else:
            size = self.const(np.array([1, hidden], np.int64))
            start = self.op(n.add_shape(state).get_output(0), size, trt.ElementWiseOperation.SUB)
            last = n.add_slice(state, (0, 0), (1, hidden), (1, 1))
            last.set_input(1, start)
            selected = last.get_output(0)
        if raw.get("tie_word_embeddings", True):
            logits = n.add_matrix_multiply(selected, trt.MatrixOperation.NONE,
                                           embedding, trt.MatrixOperation.TRANSPOSE).get_output(0)
        else:
            raise NotImplementedError("Gemma4 native untied LM head is not implemented")
        logits = self.cast(logits, trt.float32)
        if raw.get("final_logit_softcapping"):
            logits = graph_ops.add_tanh_softcap(n, logits, float(raw["final_logit_softcapping"]),
                                                scalar_shape=(1, 1))
        logits.name = "logits"
        n.mark_output(logits)


def build(request, writer, config, *, companion=None, companion_variant="mtp"):
    """Build the decoder and requested native image/audio towers."""
    from .model import _BUNDLE_FILES, _runtime_config

    for name, default in (("fp32_layers", ()), ("mtp_seq_len", None),
                          ("graph_transform", None)):
        if getattr(request, name) != default:
            raise NotImplementedError(f"Gemma4 native does not support {name}")

    raw = text_config(config)
    unified = config.model_type in {"gemma4_unified", "gemma4_unified_text"}
    include_vision = request.task in {"images_text_to_text", "image_audio_text_to_text"}
    include_audio = request.task in {"audio_text_to_text", "image_audio_text_to_text"}
    conditioned = include_vision or include_audio
    if request.task not in {"text_generation", "text_continuation", "images_text_to_text",
                            "audio_text_to_text", "image_audio_text_to_text"}:
        raise NotImplementedError("Gemma4 native media topology is not implemented")

    vision = config.raw.get("vision_config")
    if include_vision and not isinstance(vision, dict):
        raise ValueError("Gemma4 image task requires the checkpoint vision configuration")
    audio = config.raw.get("audio_config")
    if include_audio and not isinstance(audio, dict):
        raise ValueError("Gemma4 audio task requires the checkpoint audio configuration")
    source = Path(request.model_dir)
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Gemma4 native quantization requires an already prepared source")
    from .gemma4_quant import configure_aux_streams, validate_source

    readers = _open_safetensors(source)
    validate_source(source, raw, readers)
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Gemma4 native tensor parallelism is not implemented")
    if raw.get("attention_bias") or raw.get("use_bidirectional_attention") == "all":
        raise NotImplementedError("Gemma4 native requires unbiased causal text attention")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError("Gemma4 precision must be fp16, bf16, or fp32")
    capacity = request.max_sequence_length or min(config.max_position_embeddings, 2048)
    if not 0 < capacity <= config.max_position_embeddings:
        raise ValueError("Gemma4 sequence capacity exceeds the original checkpoint")
    if unified:
        from .gemma4_unified import validate_media_weights

        validate_media_weights(readers, vision=include_vision, audio=include_audio)
    audio_frontend = None
    if include_audio:
        from .gemma4_audio_frontend import frontend_constants

        audio_frontend = frontend_constants(source, config.raw, capacity)
    assistant_graph = None
    assistant_raw = None
    feature_layers = ()
    zlab_draft = False
    if companion is not None:
        import json
        if companion_variant not in {"mtp", "dflash", "dspark", "eagle3"}:
            raise NotImplementedError("Gemma native companion variant is not implemented")
        if conditioned:
            raise ValueError("Gemma4 native companions currently map text-only requests")
        assistant_raw = json.loads((Path(companion) / "config.json").read_text())
        zlab_draft = (companion_variant == "dflash"
                      and assistant_raw.get("architectures") == ["DFlashDraftModel"])
        if zlab_draft:
            assistant_raw["target_layer_ids"] = assistant_raw["dflash_config"]["target_layer_ids"]
            assistant_raw["mask_token_id"] = assistant_raw["dflash_config"]["mask_token_id"]
        if companion_variant in {"dflash", "dspark", "eagle3"}:
            feature_layers = tuple(assistant_raw["target_layer_ids"])
            if (assistant_raw["hidden_size"] != raw["hidden_size"]
                    or assistant_raw["vocab_size"] != raw["vocab_size"]
                    or (companion_variant == "dspark") != bool(assistant_raw.get("markov_rank"))):
                raise ValueError("Gemma4 block draft geometry or head type does not match the pair")
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    assistant_network = None
    if companion is not None:
        assistant_network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        # Validate the shared-KV topology before building the large target.
        if companion_variant == "mtp":
            from .gemma4_assistant import AssistantGraph

            assistant_graph = AssistantGraph(
                assistant_network, assistant_raw, _open_safetensors(Path(companion)),
                raw, readers, precision, capacity)
        elif zlab_draft:
            from .gemma4_zlab_dflash import ZlabDraftGraph

            assistant_graph = ZlabDraftGraph(
                assistant_network, assistant_raw, _open_safetensors(Path(companion)),
                precision, capacity, readers)
        else:
            if companion_variant == "eagle3":
                from .gemma4_eagle3 import EagleDraftGraph as DraftGraph
            else:
                from .gemma4_dflash import BlockDraftGraph as DraftGraph

            assistant_graph = DraftGraph(
                assistant_network, assistant_raw, _open_safetensors(Path(companion)),
                precision, capacity)
    graph = DenseTextGraph(network, raw, readers, precision, capacity,
                           conditioned, paired=companion is not None, feature_layers=feature_layers)
    graph.build()
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 1
    configure_aux_streams(build_config, raw)
    if precision == "fp32":
        build_config.clear_flag(trt.BuilderFlag.TF32)
    # Expert loops can have substantial context scratch. One dynamic
    # profile covers both phases without allocating two copies of it.
    single_context = bool(raw.get("enable_moe_block"))
    profiles = [(capacity, min(capacity, 128))]
    if not single_context:
        profiles.append((1, 1))
    for maximum, optimum in profiles:
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (1,), (optimum,), (maximum,))
        profile.set_shape("attention_mask", (1, capacity + 1),
                          (optimum, capacity + optimum), (maximum, capacity + maximum))

        if companion is not None:
            verify = (2 if companion_variant == "mtp" else
                      assistant_raw.get("block_size", assistant_raw.get("ttt_length")) + (0 if zlab_draft else 1))
            profile.set_shape("logits_indices", (1,), (1,), (min(maximum, verify),))
        if conditioned:
            profile.set_shape("media_mask", (1,), (optimum,), (maximum,))
            profile.set_shape("media_features", (1, raw["hidden_size"]),
                              (optimum, raw["hidden_size"]), (maximum, raw["hidden_size"]))
        build_config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Gemma4 native text engine build failed")
    writer.set_header(family="gemma", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", bytes(plan))
    writer.add_json("runtime.json", _runtime_config(
        source, config, precision=precision, max_cache_length=capacity,
        decoder_engine_layout="single_profile" if single_context else "dual_profile",
        tensor_parallel_size=1,
        tensor_parallel_mode="single", kv_cache_dimensions=graph.widths))

    if assistant_graph is not None:
        # Release target graph storage before constructing the companion.
        del graph, network, plan
        assistant_graph.build()
        assistant_config = builder.create_builder_config()
        assistant_config.builder_optimization_level = 1
        if precision == "fp32":
            assistant_config.clear_flag(trt.BuilderFlag.TF32)
        if companion_variant in {"dflash", "dspark", "eagle3"}:
            draft_profile = builder.create_optimization_profile()
            block = min(assistant_raw.get("block_size", assistant_raw.get("ttt_length")), capacity)
            optimum = min(capacity - 1, 128)
            for name in ("draft_tokens", "draft_positions"):
                if companion_variant == "eagle3":
                    draft_profile.set_shape(name, (1,), (optimum,), (capacity - 1,))
                else:
                    draft_profile.set_shape(name, (1,), (block,), (block,))
            width = len(feature_layers) * raw["hidden_size"]
            if companion_variant != "eagle3":
                draft_profile.set_shape("context_positions", (1,), (optimum,), (capacity - 1,))
            draft_profile.set_shape("target_features", (1, width), (optimum, width),
                                    (capacity - 1, width))
            assistant_config.add_optimization_profile(draft_profile)
        assistant_plan = builder.build_serialized_network(assistant_network, assistant_config)
        if assistant_plan is None:
            raise RuntimeError("Gemma4 native companion engine build failed")
        writer.add_bytes("assistant.plan", bytes(assistant_plan))
        metadata = {
            "variant": companion_variant, "hidden_size": raw["hidden_size"],
            "vocab_size": raw["vocab_size"], "target_layers": assistant_graph.target_layers,
        }
        if companion_variant in {"dflash", "dspark", "eagle3"}:
            metadata["block_size"] = assistant_raw.get("block_size", assistant_raw.get("ttt_length"))
            if companion_variant != "eagle3":
                metadata["mask_token_id"] = assistant_raw["mask_token_id"]
            if zlab_draft:
                metadata["skip_first"] = True
        writer.add_json("assistant.json", metadata)

    if include_vision:
        from .gemma4_vision import VisionGraph

        visual = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        soft_tokens = int(config.raw.get("vision_soft_tokens_per_image", 280))
        if unified:
            from .gemma4_unified import UnifiedMediaGraph

            visual_graph = UnifiedMediaGraph(visual, config.raw, readers, precision)
            visual_graph.image(soft_tokens)
        else:
            visual_graph = VisionGraph(visual, vision, readers, precision, soft_tokens)
            visual_graph.build()
        visual_config = builder.create_builder_config()
        visual_config.builder_optimization_level = 1
        if precision == "fp32":
            visual_config.clear_flag(trt.BuilderFlag.TF32)
        visual_plan = builder.build_serialized_network(visual, visual_config)
        if visual_plan is None:
            raise RuntimeError("Gemma4 native vision engine build failed")
        writer.add_bytes("vision.plan", bytes(visual_plan))
        writer.add_json("vision.json", {
            "patch_size": vision["patch_size"], "pooling_kernel": vision["pooling_kernel_size"],
            "soft_tokens": soft_tokens, "hidden_size": raw["hidden_size"],
            "merged_patches": unified,
            "image_token": config.raw["image_token_id"], "begin_token": config.raw["boi_token_id"],
            "end_token": config.raw["eoi_token_id"],
        })
    if include_audio:
        from .gemma4_audio import AudioGraph

        frontend, window, filters = audio_frontend
        acoustic = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        if unified:
            from .gemma4_unified import UnifiedMediaGraph

            audio_graph = UnifiedMediaGraph(acoustic, config.raw, readers, precision)
            audio_graph.audio(frontend["max_frames"])
        else:
            audio_graph = AudioGraph(acoustic, audio, readers, precision,
                                     frontend["max_frames"], frontend["mel_bins"])
            audio_graph.build()
        audio_config = builder.create_builder_config()
        audio_config.builder_optimization_level = 1
        if precision == "fp32":
            audio_config.clear_flag(trt.BuilderFlag.TF32)
        audio_plan = builder.build_serialized_network(acoustic, audio_config)
        if audio_plan is None:
            raise RuntimeError("Gemma4 native audio engine build failed")
        writer.add_bytes("audio.plan", bytes(audio_plan))
        writer.add_json("audio.json", frontend)
        if not unified:
            writer.add_bytes("audio_window.f32", window.astype("<f4").tobytes())
            writer.add_bytes("audio_mel.f64", filters.astype("<f8").tobytes())
    for name in _BUNDLE_FILES:
        path = source / name
        if path.is_file():
            writer.add_bytes(name, path.read_bytes())
