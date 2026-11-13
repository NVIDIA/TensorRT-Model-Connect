# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8 DFlash2 native backbone and conditional candidate-selector graphs.

All convolution, attention, quantized head and selector math uses TensorRT Python
operators. No ONNX frontend, custom GPU helper, or cross-family implementation.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_blocks, graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors
from .draft import rotary_tables


@dataclass(frozen=True)
class DFlash2Config:
    raw: dict
    feature_layers: tuple[int, ...]
    hidden: int
    vocab: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    mask_token: int
    epsilon: float
    window: int
    group_size: int
    kernel_size: int
    selector_rank: int
    selector_top_k: int
    query_rows: int = 8

    @classmethod
    def from_dict(cls, raw, target):
        policy = raw.get("dflash_config") or {}
        feature_layers = tuple(policy.get("target_layer_ids", ()))
        h, v, layers = int(raw["hidden_size"]), int(raw["vocab_size"]), int(raw["num_hidden_layers"])
        if (raw.get("architectures") != ["DFlash2DraftModel"]
                or h != target["hidden_size"] or v != target["vocab_size"]
                or raw.get("num_target_layers") != target["num_hidden_layers"]
                or policy.get("block_size") != 8 or not feature_layers
                or len(set(feature_layers)) != len(feature_layers)
                or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"]
                       for i in feature_layers)):
            raise ValueError("Qwen3.8 DFlash2 companion does not match target/block8 contract")
        if (raw.get("attention_bias", False) or raw.get("hidden_act", "silu") != "silu"
                or raw.get("is_causal", False) or raw.get("attention_sink_bias", False)
                or raw.get("layer_types") != ["sliding_attention"] * layers
                or raw.get("quantization_config")):
            raise ValueError("Unsupported Qwen3.8 DFlash2 attention/projection contract")
        heads, kv = int(raw["num_attention_heads"]), int(raw["num_key_value_heads"])
        dim = int(raw.get("head_dim", h // max(heads, 1)))
        width, window = int(raw["intermediate_size"]), int(raw["sliding_window"])
        group, kernel = int(policy["conv_group_size"]), int(policy["conv_kernel_size"])
        rank, top_k = int(policy["selector_rank"]), int(policy["selector_top_k"])
        mask, eps = int(policy["mask_token_id"]), float(raw.get("rms_norm_eps", 1e-6))
        if (min(h, v, layers, heads, kv, dim, width, window, group, rank) <= 0
                or heads % kv or dim % 2 or h % group or kernel != 2 or top_k != 16
                or top_k > v or not 0 <= mask < v or not np.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid Qwen3.8 DFlash2 geometry")
        return cls(raw, feature_layers, h, v, layers, heads, kv, dim, width,
                   mask, eps, window, group, kernel, rank, top_k)


def load_draft_weights(model_dir: Path, config: DFlash2Config):
    """Read the original unquantized companion, with Edge's FP16 export rounding."""
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
        for conv in ("attention_conv", "mlp_conv"):
            load(f"{prefix}.{conv}.kernel_projection.weight",
                 (2 * config.kernel_size * h // config.group_size, h), True)
            load(f"{prefix}.{conv}.base_kernel", (2, config.kernel_size, h))
    load("candidate_selector.hidden_projection.weight", (config.selector_rank, h), True)
    load("candidate_selector.predecessor_codebook", (v, config.selector_rank))
    load("candidate_selector.successor_codebook", (v, config.selector_rank))
    if _has_tensor(readers, "lm_head.weight") or _has_tensor(readers, "embed_tokens.weight"):
        raise ValueError("Qwen3.8 DFlash2 requires inherited target head and embeddings")
    return weights


def build_draft_engine(
    config, weights, target_lm_head, max_cache_length, *, quant_ctx=None, verbose=False, debug_layer_outputs=False
):
    """Build the real block draft, with no plugins, ONNX or target substitution."""
    if max_cache_length < config.query_rows:
        raise ValueError("Native Qwen3.8 DFlash2 draft context is smaller than its proposal block")
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

    def convolution(value, coefficients, base, residual=None):
        # Kernel semantics: (base[channel] + delta[group]) * hidden[token-tap],
        # accumulated in FP32. Pre-conv saturates to FP16; post-conv adds the
        # FP32 residual with no FP16 rounding or saturation.
        groups = h // config.group_size
        terms = []
        for tap in range(config.kernel_size):
            dynamic = network.add_slice(
                coefficients, (0, tap * groups), (rows, groups), (1, 1)
            ).get_output(0)
            dynamic = reshape(cast(dynamic, trt.float32), (rows, groups, 1))
            fixed = constant(base[tap].reshape(1, groups, config.group_size))
            shifted = value
            if tap:
                prefix_zeros = constant(np.zeros((tap, h)), np.float16)
                prefix_values = network.add_slice(value, (0, 0), (rows - tap, h), (1, 1)).get_output(0)
                concat = network.add_concatenation([prefix_zeros, prefix_values])
                concat.axis = 0
                shifted = concat.get_output(0)
            term = binary(binary(fixed, dynamic, trt.ElementWiseOperation.SUM),
                          reshape(cast(shifted, trt.float32), (rows, groups, config.group_size)),
                          trt.ElementWiseOperation.PROD)
            terms.append(term)
        output = terms[0]
        for term in terms[1:]:
            output = binary(output, term, trt.ElementWiseOperation.SUM)
        output = reshape(output, (rows, h))
        if residual is not None:
            return binary(output, residual, trt.ElementWiseOperation.SUM)
        clipped = network.add_activation(output, trt.ActivationType.CLIP)
        clipped.alpha, clipped.beta = -65504.0, 65504.0
        return cast(clipped.get_output(0), trt.float16)

    def conv_prepare(value, prefix):
        width = config.kernel_size * h // config.group_size
        coefficients = linear(value, weights[f"{prefix}.kernel_projection.weight"], h, 2 * width)
        pre = network.add_slice(coefficients, (0, 0), (rows, width), (1, 1)).get_output(0)
        post = network.add_slice(coefficients, (0, width), (rows, width), (1, 1)).get_output(0)
        return convolution(value, pre, weights[f"{prefix}.base_kernel"][0]), post

    hidden = cast(embeddings, trt.float32)
    # The Qwen DFlash2 projector must remain FP32 through RMSNorm. Features can
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
        residual = hidden
        normalized = cast(norm(hidden, weights[f"{prefix}.input_layernorm.weight"], h), trt.float16)
        normalized, attention_post = conv_prepare(normalized, f"{prefix}.attention_conv")
        debug(normalized, f"attention_pre_{i}")

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
        hidden = convolution(projected, attention_post,
                             weights[f"{prefix}.attention_conv.base_kernel"][1], residual)
        debug(hidden, f"attention_{i}")
        residual = hidden
        normalized = cast(norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], h), trt.float16)
        normalized, mlp_post = conv_prepare(normalized, f"{prefix}.mlp_conv")
        debug(normalized, f"mlp_pre_{i}")
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
        hidden = convolution(down, mlp_post, weights[f"{prefix}.mlp_conv.base_kernel"][1], residual)
        debug(hidden, f"layer_{i}")
    hidden = cast(norm(hidden, weights["norm.weight"], h), trt.float16)
    hidden = network.add_slice(hidden, (1, 0), (rows - 1, h), (1, 1)).get_output(0)
    debug(hidden, "prediction_hidden")
    projected = linear(hidden, weights["candidate_selector.hidden_projection.weight"],
                       h, config.selector_rank, rows - 1)
    projected.name = "projected_hidden"
    network.mark_output(projected)
    head = graph_blocks.make_matmul_fn(network, np.float16, quant_ctx)
    logits = cast(head(hidden, h, config.vocab, target_lm_head, "w_lm_head"), trt.float32)
    debug(logits, "unary_logits")
    candidates = network.add_topk(logits, trt.TopKOperation.MAX, config.selector_top_k, 1 << 1)
    values, ids = candidates.get_output(0), candidates.get_output(1)
    values.name, ids.name = "unary_values", "candidate_ids"
    network.mark_output(values)
    network.mark_output(ids)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Qwen3.8 DFlash2 draft engine build failed")
    return bytes(plan)


def build_selector_engine(config, weights, *, verbose=False):
    """One conditional selector step: unary + sum(pred * projected * succ).

    Sampling is orchestrated in C++, passing the actual preceding proposal. The
    codebooks remain original FP16 and their product/reduction stays FP32.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    # The predecessor/hidden product has FP32 information beyond FP16/TF32.
    # Preserve the source FP32 selector reduction rather than rounding that
    # product to TF32 before the successor-codebook dot product.
    build_config.clear_flag(trt.BuilderFlag.TF32)
    k, rank = config.selector_top_k, config.selector_rank
    previous = network.add_input("previous_token", trt.int32, (1,))
    ids = network.add_input("candidate_ids", trt.int32, (k,))
    unary = network.add_input("unary_values", trt.float32, (1, k))
    projected = network.add_input("projected_hidden", trt.float16, (1, rank))

    def table(name):
        value = weights[f"candidate_selector.{name}_codebook"]
        return graph_ops.add_constant(network, value.shape, value, dtype=np.float16)

    def fp32(value):
        return network.add_cast(value, trt.float32).get_output(0)

    predecessor = network.add_gather(table("predecessor"), previous, 0).get_output(0)
    successor = network.add_gather(table("successor"), ids, 0).get_output(0)
    conditioned = network.add_elementwise(fp32(predecessor), fp32(projected),
                                          trt.ElementWiseOperation.PROD).get_output(0)
    correction = network.add_matrix_multiply(
        conditioned, trt.MatrixOperation.NONE, fp32(successor), trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(unary, correction, trt.ElementWiseOperation.SUM).get_output(0)
    scores.name = "selector_scores"
    network.mark_output(scores)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Qwen3.8 DFlash2 selector build failed")
    return bytes(plan)
