# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 native DSpark graph and checkpoint contract.

Reproject confirmed target features on each draft round. Proposal KV is local to
that round, so rejected proposals never enter persistent recurrent target state.
The target supplies token embeddings and its actual (possibly quantized) LM head.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_blocks, graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors


@dataclass(frozen=True)
class DraftConfig:
    raw: dict
    feature_layers: tuple[int, ...]
    hidden: int
    vocab: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    block_size: int
    mask_token: int
    epsilon: float
    markov_rank: int
    sample_from_anchor: bool
    causal: bool
    confidence: bool
    confidence_with_markov: bool

    @property
    def query_rows(self):
        return self.block_size + (not self.sample_from_anchor)

    @classmethod
    def from_dict(cls, raw, target):
        policy = raw.get("dspark_config") or {}

        def field(name, default=None):
            return policy.get(name, raw.get(name, default))

        feature_layers = tuple(field("target_layer_ids", ()))
        h, v = int(raw["hidden_size"]), int(raw["vocab_size"])
        if (
            raw.get("architectures") != ["DSparkDraftModel"]
            or h != target["hidden_size"]
            or v != target["vocab_size"]
            or raw.get("num_target_layers") != target["num_hidden_layers"]
            or not feature_layers
            or len(set(feature_layers)) != len(feature_layers)
            or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"]
                   for i in feature_layers)
            or field("block_size", 7) != 7
        ):
            raise ValueError("Qwen3.8 DSpark companion does not match its target/block7 contract")
        layers = int(raw["num_hidden_layers"])
        if (
            raw.get("attention_bias", False)
            or raw.get("hidden_act", "silu") != "silu"
            or field("attention_mode", "gqa") != "gqa"
            or field("projector_type", "dspark") != "dspark"
            or field("attention_sink_bias", False)
            or raw.get("attention_k_eq_v", False)
            or raw.get("sliding_window") not in (None, 0)
            or raw.get("layer_types", ["full_attention"] * layers) != ["full_attention"] * layers
            or field("markov_head_type", "vanilla") != "vanilla"
            or raw.get("draft_vocab_size", v) != v
            or raw.get("target_hidden_size") not in (None, h)
            or raw.get("quantization_config")
        ):
            raise ValueError("Unsupported Qwen3.8 DSpark projection/attention contract")
        heads, kv = int(raw["num_attention_heads"]), int(raw["num_key_value_heads"])
        dim = int(raw.get("head_dim", h // max(heads, 1)))
        width, rank = int(raw["intermediate_size"]), int(field("markov_rank", 0))
        mask, eps = int(field("mask_token_id", -1)), float(raw.get("rms_norm_eps", 1e-6))
        if (min(h, v, heads, kv, dim, layers, width, rank) <= 0 or heads % kv or dim % 2
                or not 0 <= mask < v or not np.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid Qwen3.8 DSpark geometry")
        return cls(raw, feature_layers, h, v, layers, heads, kv, dim, width, 7, mask, eps,
                   rank, bool(field("sample_from_anchor", True)),
                   bool(policy.get("causal", raw.get("dflash_query_causal",
                             (raw.get("dflash_config") or {}).get("causal", False)))),
                   bool(field("enable_confidence_head", False)),
                   bool(field("confidence_head_with_markov", False)))


def rotary_tables(raw, head_dim, capacity):
    """Official YaRN NTK-by-parts interpolation, or ordinary RoPE, in FP32."""
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    kind = rope.get("rope_type", rope.get("type", "default"))
    theta = float(rope.get("rope_theta", raw.get("rope_theta", 10000.0)))
    if head_dim <= 0 or head_dim % 2 or not np.isfinite(theta) or theta <= 1:
        raise ValueError("Invalid Qwen3.8 draft rotary geometry")
    indices = np.arange(head_dim // 2, dtype=np.float32)
    inverse = 1.0 / np.power(np.float32(theta), 2 * indices / head_dim)
    amplitude = 1.0
    if kind == "yarn":
        factor = float(rope["factor"])
        original = int(rope["original_max_position_embeddings"])
        fast, slow = float(rope.get("beta_fast", 32)), float(rope.get("beta_slow", 1))
        if min(factor, original, fast, slow) <= 0 or not np.isfinite([factor, fast, slow]).all():
            raise ValueError("Invalid Qwen3.8 draft YaRN scaling")

        def correction(rotations):
            return head_dim * np.log(original / (rotations * 2 * np.pi)) / (2 * np.log(theta))

        low = max(np.floor(correction(fast)), 0)
        high = min(np.ceil(correction(slow)), head_dim - 1)
        ramp = np.clip((indices - low) / (high - low if high != low else 0.001), 0, 1)
        inverse = inverse / factor * ramp + inverse * (1 - ramp)
        supplied_amplitude = rope.get("attention_factor")
        amplitude = float(
            supplied_amplitude
            if supplied_amplitude is not None
            else 0.1 * np.log(factor) + 1
            if factor > 1
            else 1
        )
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("Invalid Qwen3.8 draft YaRN attention factor")
    elif kind != "default":
        raise ValueError(f"Unsupported Qwen3.8 draft rotary scaling: {kind}")
    phases = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]
    return (
        (np.cos(phases) * amplitude).astype(np.float32),
        (np.sin(phases) * amplitude).astype(np.float32),
    )


def load_draft_weights(model_dir: Path, config: DraftConfig):
    """Read dense draft tensors once, preserving the upstream FP16 export rounding."""
    readers = _open_safetensors(model_dir)
    weights = {}

    def load(name, shape, projection=False):
        value = _load_tensor(readers, name)
        if value.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        weights[name] = np.ascontiguousarray(value.T if projection else value, dtype=np.float16)

    h, v, d = config.hidden, config.vocab, config.head_dim
    load("fc.weight", (h, len(config.feature_layers) * h), True)
    load("hidden_norm.weight", (h,))
    load("norm.weight", (h,))
    for i in range(config.layers):
        prefix = f"layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            load(f"{prefix}.{norm}.weight", (h,))
        for projection, out_size, in_size in (
            ("q_proj", config.heads * d, h), ("k_proj", config.kv_heads * d, h),
            ("v_proj", config.kv_heads * d, h), ("o_proj", h, config.heads * d),
        ):
            load(f"{prefix}.self_attn.{projection}.weight", (out_size, in_size), True)
        for norm in ("q_norm", "k_norm"):
            load(f"{prefix}.self_attn.{norm}.weight", (d,))
        for projection, out_size, in_size in (
            ("gate_proj", config.intermediate, h), ("up_proj", config.intermediate, h),
            ("down_proj", h, config.intermediate),
        ):
            load(f"{prefix}.mlp.{projection}.weight", (out_size, in_size), True)
    load("markov_head.markov_w1.weight", (v, config.markov_rank))
    load("markov_head.markov_w2.weight", (v, config.markov_rank), True)
    if config.confidence:
        width = h + (config.markov_rank if config.confidence_with_markov else 0)
        load("confidence_head.proj.weight", (1, width), True)
        load("confidence_head.proj.bias", (1,))
    if _has_tensor(readers, "lm_head.weight") or _has_tensor(readers, "embed_tokens.weight"):
        raise ValueError("Qwen3.8 DSpark requires the inherited target head and embeddings")
    return weights


def build_draft_engine(
    config, weights, target_lm_head, max_cache_length, *, quant_ctx=None, verbose=False, debug_layer_outputs=False
):
    """Build the real block draft, with no plugins, ONNX or target substitution."""
    if max_cache_length < config.query_rows:
        raise ValueError("Native Qwen3.8 draft context is smaller than its proposal block")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    h, heads, kv, dim = config.hidden, config.heads, config.kv_heads, config.head_dim
    rows, capacity = config.query_rows, max_cache_length
    feature_width = len(config.feature_layers) * h
    embeddings = network.add_input("inputs_embeds", trt.float16, (rows, h))
    positions = network.add_input("draft_positions", trt.int32, (rows,))
    features = network.add_input("target_hidden", trt.float16, (capacity, feature_width))
    mask = network.add_input("draft_attention_mask", trt.float32, (rows, capacity + rows))

    def constant(value, dtype=np.float32):
        value = np.asarray(value, dtype=dtype)
        return graph_ops.add_constant(network, value.shape, value, dtype=dtype)

    def cast(value, dtype):
        return value if value.dtype == dtype else network.add_cast(value, dtype).get_output(0)

    def reshape(value, shape):
        op = network.add_shuffle(value)
        op.reshape_dims = shape
        return op.get_output(0)

    def binary(a, b, operation):
        return network.add_elementwise(a, b, operation).get_output(0)

    def linear(value, weight, in_width, out_width, nrows=rows):
        return graph_ops.add_matmul_rhs_constant(
            network, value, in_width, out_width, weight, dtype=np.float16
        )

    def norm(value, weight, width):
        # Match source RMSNorm: FP32 reduction; cast normalized values before
        # multiplying the source gamma in the original activation dtype.
        dtype = value.dtype
        x = cast(value, trt.float32)
        squared = binary(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
        denom = binary(mean, constant([[config.epsilon]]), trt.ElementWiseOperation.SUM)
        root = network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
        normalized = cast(binary(x, root, trt.ElementWiseOperation.DIV), dtype)
        gamma = cast(constant(np.asarray(weight).reshape(1, width)), dtype)
        return binary(normalized, gamma, trt.ElementWiseOperation.PROD)

    def debug(value, name):
        if debug_layer_outputs:
            output = network.add_identity(value).get_output(0)
            output.name = "debug_" + name
            network.mark_output(output)

    cos, sin = rotary_tables(config.raw, dim, capacity + rows)
    cosine, sine = constant(cos), constant(sin)
    context_positions = constant(np.arange(capacity), np.int32)

    def rotate(value, nrows, nheads, pos):
        x = reshape(cast(value, trt.float32), (nrows, nheads, dim))
        left = network.add_slice(x, (0, 0, 0), (nrows, nheads, dim // 2), (1, 1, 1)).get_output(0)
        right = network.add_slice(
            x, (0, 0, dim // 2), (nrows, nheads, dim // 2), (1, 1, 1)
        ).get_output(0)
        c = reshape(network.add_gather(cosine, pos, 0).get_output(0), (nrows, 1, dim // 2))
        s = reshape(network.add_gather(sine, pos, 0).get_output(0), (nrows, 1, dim // 2))
        a = binary(
            binary(left, c, trt.ElementWiseOperation.PROD),
            binary(right, s, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUB,
        )
        b = binary(
            binary(right, c, trt.ElementWiseOperation.PROD),
            binary(left, s, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM,
        )
        joined = network.add_concatenation([a, b])
        joined.axis = 2
        return cast(reshape(joined.get_output(0), (nrows, nheads * dim)), trt.float16)

    def head_norm(value, weight, nrows, nheads):
        return reshape(
            norm(reshape(value, (nrows * nheads, dim)), weight, dim), (nrows, nheads * dim)
        )

    def head_rows(value, nrows, nheads):
        op = network.add_shuffle(value)
        op.reshape_dims = (nrows, nheads, dim)
        op.second_transpose = trt.Permutation([1, 0, 2])
        return op.get_output(0)

    hidden = embeddings
    # The Qwen DSpark projector must remain FP32 through RMSNorm. Features can
    # exceed FP16 FC output range; promoting an already overflowed result fails.
    fused = graph_ops.add_matmul_rhs_constant(
        network, cast(features, trt.float32), feature_width, h,
        weights["fc.weight"].astype(np.float32), dtype=np.float32,
    )
    fused = cast(norm(fused, weights["hidden_norm.weight"], h), trt.float16)
    debug(fused, "features")
    for i in range(config.layers):
        prefix = f"layers.{i}"
        attn = f"{prefix}.self_attn"
        normalized = norm(hidden, weights[f"{prefix}.input_layernorm.weight"], h)

        def project(value, name, out_width, nrows=rows):
            return linear(value, weights[f"{attn}.{name}.weight"], h, out_width, nrows)

        q = head_norm(
            project(normalized, "q_proj", heads * dim),
            weights[f"{attn}.q_norm.weight"],
            rows,
            heads,
        )
        k = head_norm(
            project(normalized, "k_proj", kv * dim), weights[f"{attn}.k_norm.weight"], rows, kv
        )
        v = project(normalized, "v_proj", kv * dim)
        context_k = head_norm(
            project(fused, "k_proj", kv * dim, capacity), weights[f"{attn}.k_norm.weight"], capacity, kv
        )
        context_v = project(fused, "v_proj", kv * dim, capacity)
        q = rotate(q, rows, heads, positions)
        k = rotate(k, rows, kv, positions)
        context_k = rotate(context_k, capacity, kv, context_positions)
        keys = network.add_concatenation([context_k, k])
        keys.axis = 0
        values = network.add_concatenation([context_v, v])
        values.axis = 0
        group, total = heads // kv, capacity + rows
        q = reshape(cast(head_rows(q, rows, heads), trt.float32), (kv, group, rows, dim))
        k = reshape(
            cast(head_rows(keys.get_output(0), total, kv), trt.float32), (kv, 1, total, dim)
        )
        v = reshape(
            cast(head_rows(values.get_output(0), total, kv), trt.float32), (kv, 1, total, dim)
        )
        scores = network.add_matrix_multiply(
            q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        scores = binary(
            scores, constant(np.full((1, 1, 1, 1), 1 / np.sqrt(dim))), trt.ElementWiseOperation.PROD
        )
        scores = binary(scores, reshape(mask, (1, 1, rows, total)), trt.ElementWiseOperation.SUM)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 3
        probabilities = probabilities.get_output(0)
        probabilities = cast(cast(probabilities, trt.float16), trt.float32)
        context = network.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
        ).get_output(0)
        context = reshape(context, (heads, rows, dim))
        shuffle = network.add_shuffle(context)
        shuffle.first_transpose = trt.Permutation([1, 0, 2])
        shuffle.reshape_dims = (rows, heads * dim)
        projected = linear(
            cast(shuffle.get_output(0), trt.float16),
            weights[f"{attn}.o_proj.weight"],
            heads * dim,
            h,
        )
        hidden = binary(hidden, projected, trt.ElementWiseOperation.SUM)
        debug(hidden, f"attention_{i}")
        normalized = norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], h)
        gate = linear(normalized, weights[f"{prefix}.mlp.gate_proj.weight"], h, config.intermediate)
        up = linear(normalized, weights[f"{prefix}.mlp.up_proj.weight"], h, config.intermediate)
        # Source F.silu evaluates sigmoid and product in FP32 for FP16
        # inputs, then rounds once before the separate SwiGLU product.
        gate = cast(
            graph_ops.add_activation(network, cast(gate, trt.float32), "silu", dtype=np.float32),
            trt.float16,
        )
        down = linear(
            binary(gate, up, trt.ElementWiseOperation.PROD),
            weights[f"{prefix}.mlp.down_proj.weight"],
            config.intermediate,
            h,
        )
        hidden = binary(hidden, down, trt.ElementWiseOperation.SUM)
        debug(hidden, f"layer_{i}")
    hidden = norm(hidden, weights["norm.weight"], h)
    hidden.name = "draft_hidden"
    network.mark_output(hidden)
    head = graph_blocks.make_matmul_fn(network, np.float16, quant_ctx)
    logits = cast(head(hidden, h, config.vocab, target_lm_head, "w_lm_head"), trt.float32)
    logits.name = "draft_logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Qwen3.8 draft engine build failed")
    return bytes(plan)


def build_markov_engine(config, weights, *, verbose=False):
    """FP32 Markov bias and optional confidence from the source FP16 sidecars.

    Conditioning uses the previous sampled token, not the proposal at this row.
    Confidence is exposed separately; it never modifies the target distribution.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3

    def fp32(value):
        return network.add_cast(value, trt.float32).get_output(0)

    def constant(value):
        value = np.ascontiguousarray(value, dtype=np.float16)
        return graph_ops.add_constant(network, value.shape, value, dtype=np.float16)

    def matmul(a, b):
        return network.add_matrix_multiply(
            a, trt.MatrixOperation.NONE, b, trt.MatrixOperation.NONE
        ).get_output(0)

    previous = network.add_input("previous_token", trt.int32, (1,))
    row = network.add_gather(constant(weights["markov_head.markov_w1.weight"]), previous, 0).get_output(0)
    bias = matmul(fp32(row), fp32(constant(weights["markov_head.markov_w2.weight"])))
    bias.name = "markov_bias"
    network.mark_output(bias)
    if config.confidence:
        hidden = network.add_input("draft_hidden", trt.float16, (1, config.hidden))
        value = hidden
        if config.confidence_with_markov:
            concat = network.add_concatenation([hidden, row])
            concat.axis = 1
            value = concat.get_output(0)
        score = matmul(fp32(value), fp32(constant(weights["confidence_head.proj.weight"])))
        score = network.add_elementwise(
            score, fp32(constant(weights["confidence_head.proj.bias"].reshape(1, 1))),
            trt.ElementWiseOperation.SUM,
        ).get_output(0)
        confidence = network.add_activation(score, trt.ActivationType.SIGMOID).get_output(0)
        confidence.name = "confidence"
        network.mark_output(confidence)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Qwen3.8 Markov engine build failed")
    return bytes(plan)
