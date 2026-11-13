# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-MoE DFlash block graph, owned independently of other model families.

Only confirmed target features enter the draft. Reprojecting those features
avoids a second persistent cache and never commits rejected proposals. Both
linear and DDTree verification consume this same bidirectional block graph.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors
from .quantization import PackedLinear, load_projection


@dataclass(frozen=True)
class DraftConfig:
    raw: dict
    variant: str
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

    @classmethod
    def from_dict(cls, raw, target, variant):
        if variant not in {"dflash", "dflash_ddtree"}:
            raise ValueError("Unsupported native Qwen3.5-MoE DFlash variant")
        policy = raw.get("dflash_config") or {}
        feature_layers = tuple(policy.get("target_layer_ids", ()))
        hidden, vocab = int(raw["hidden_size"]), int(raw["vocab_size"])
        if (
            raw.get("architectures") != ["DFlashDraftModel"]
            or hidden != target["hidden_size"]
            or vocab != target["vocab_size"]
            or raw.get("num_target_layers") != target["num_hidden_layers"]
            or not feature_layers
            or len(set(feature_layers)) != len(feature_layers)
            or any(type(i) is not int or not 0 <= i < target["num_hidden_layers"]
                   for i in feature_layers)
            or policy.get("block_size") != 16
        ):
            raise ValueError("Qwen3.5-MoE DFlash companion metadata does not match its target")
        if (raw.get("attention_bias", False)
                or raw.get("hidden_act", "silu") != "silu"
                or raw.get("attention_sink_bias", False)
                or policy.get("causal", False)):
            raise ValueError("Qwen3.5-MoE DFlash requires bias-free bidirectional attention and SwiGLU")
        heads, kv = int(raw["num_attention_heads"]), int(raw["num_key_value_heads"])
        if min(heads, kv) <= 0:
            raise ValueError("Invalid Qwen3.5-MoE DFlash head counts")
        dim = int(raw.get("head_dim", hidden // heads))
        layers, width = int(raw["num_hidden_layers"]), int(raw["intermediate_size"])
        mask = int(policy.get("mask_token_id", -1))
        eps = float(raw.get("rms_norm_eps", 1e-6))
        if (min(hidden, vocab, dim, layers, width) <= 0 or heads % kv or dim % 2
                or not 0 <= mask < vocab or not np.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid Qwen3.5-MoE DFlash geometry")
        # The published drafts carry Qwen3 SWA labels. The pinned DFlash
        # computation does not use them: non-Gemma draft attention is full,
        # bidirectional attention, not the ordinary Qwen3 causal decoder.
        return cls(raw, variant, feature_layers, hidden, vocab, layers, heads,
                   kv, dim, width, 16, mask, eps)


def rotary_tables(raw, head_dim, capacity):
    """The published Qwen3.5-MoE DFlash companions use full-head default RoPE."""
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Qwen3.5-MoE DFlash requires default rotary scaling")
    theta = float(rope.get("rope_theta", raw.get("rope_theta", 10000.0)))
    if head_dim <= 0 or head_dim % 2 or not np.isfinite(theta) or theta <= 1:
        raise ValueError("Invalid Qwen3.5-MoE DFlash rotary geometry")
    inverse = 1.0 / np.power(np.float32(theta), np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    phases = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]
    return np.cos(phases).astype(np.float32), np.sin(phases).astype(np.float32)


def load_draft_weights(model_dir: Path, config: DraftConfig):
    """Preserve packed projections and the draft's separately calibrated head."""
    readers = _open_safetensors(model_dir, prepared=True)
    weights = {}

    def load(name, shape, projection=False):
        value = (load_projection(readers, name.removesuffix(".weight"))
                 if projection else _load_tensor(readers, name).astype(np.float32))
        expected = shape[::-1] if projection else shape
        if value.shape != expected:
            raise ValueError(f"{name}: expected {expected}, got {value.shape}")
        weights[name] = value

    h, v, d = config.hidden, config.vocab, config.head_dim
    load("fc.weight", (h, len(config.feature_layers) * h), True)
    if isinstance(weights["fc.weight"], PackedLinear):
        raise ValueError("Qwen3.5-MoE DFlash requires dense FC weights for the FP32 feature projection")
    load("hidden_norm.weight", (h,))
    load("norm.weight", (h,))
    for i in range(config.layers):
        prefix = f"layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            load(f"{prefix}.{norm}.weight", (h,))
        for name, out_size, in_size in (
            ("q_proj", config.heads * d, h), ("k_proj", config.kv_heads * d, h),
            ("v_proj", config.kv_heads * d, h), ("o_proj", h, config.heads * d),
        ):
            load(f"{prefix}.self_attn.{name}.weight", (out_size, in_size), True)
        for norm in ("q_norm", "k_norm"):
            load(f"{prefix}.self_attn.{norm}.weight", (d,))
        for name, out_size, in_size in (
            ("gate_proj", config.intermediate, h), ("up_proj", config.intermediate, h),
            ("down_proj", h, config.intermediate),
        ):
            load(f"{prefix}.mlp.{name}.weight", (out_size, in_size), True)
    if _has_tensor(readers, "lm_head.weight"):
        load("lm_head.weight", (v, h), True)
    return weights


def build_draft_engine(
    config, weights, target_embedding, target_lm_head, max_cache_length, *, verbose=False, debug_layer_outputs=False
):
    """Build the real block draft, with no plugins, ONNX or target substitution."""
    if target_embedding.shape != (config.vocab, config.hidden):
        raise ValueError("Qwen3.5-MoE DFlash target embedding shape mismatch")
    if max_cache_length < config.block_size:
        raise ValueError("Native Qwen3.5-MoE DFlash draft context is smaller than its proposal block")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    build_config.clear_flag(trt.BuilderFlag.TF32)
    h, heads, kv, dim = config.hidden, config.heads, config.kv_heads, config.head_dim
    rows, capacity = config.block_size, max_cache_length
    feature_width = len(config.feature_layers) * h
    ids = network.add_input("draft_token_ids", trt.int32, (rows,))
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
        if isinstance(weight, PackedLinear):
            return weight.matmul(network, value, in_width, out_width, fp32_accumulation=True)
        # Match FP16 linear arithmetic: FP16 inputs/weights, FP32 dot-product
        # accumulation, and one FP16 output rounding before the next operator.
        rhs = constant(np.asarray(weight, dtype=np.float16).astype(np.float32))
        result = network.add_matrix_multiply(
            cast(value, trt.float32), trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE
        ).get_output(0)
        return cast(result, trt.float16)

    def norm(value, weight, width):
        # Match source RMSNorm: FP32 reduction; cast normalized values before
        # multiplying the source gamma in the original activation dtype.
        dtype = value.dtype
        x = cast(value, trt.float32)
        squared = binary(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(squared, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
        denom = binary(mean, constant([[config.epsilon]]), trt.ElementWiseOperation.SUM)
        root = network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
        inverse = network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
        normalized = cast(binary(x, inverse, trt.ElementWiseOperation.PROD), dtype)
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

    hidden = network.add_gather(
        constant(target_embedding, np.float16), ids, 0
    ).get_output(0)
    # The official Qwen DFlash projector accumulates AND outputs in FP32.
    # Casting an already-overflowed FP16 projection afterwards is incorrect.
    fc = constant(np.asarray(weights["fc.weight"], dtype=np.float16).astype(np.float32))
    fused = network.add_matrix_multiply(
        cast(features, trt.float32), trt.MatrixOperation.NONE, fc, trt.MatrixOperation.NONE
    ).get_output(0)
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
        debug(normalized, f"mlp_norm_{i}")
        gate = linear(normalized, weights[f"{prefix}.mlp.gate_proj.weight"], h, config.intermediate)
        up = linear(normalized, weights[f"{prefix}.mlp.up_proj.weight"], h, config.intermediate)
        debug(gate, f"gate_{i}")
        debug(up, f"up_{i}")
        # Source F.silu evaluates sigmoid and product in FP32 for FP16
        # inputs, then rounds once before the separate SwiGLU product.
        gate = cast(
            graph_ops.add_activation(network, cast(gate, trt.float32), "silu", dtype=np.float32),
            trt.float16,
        )
        debug(gate, f"silu_{i}")
        mlp_product = binary(gate, up, trt.ElementWiseOperation.PROD)
        debug(mlp_product, f"product_{i}")
        down = linear(
            mlp_product,
            weights[f"{prefix}.mlp.down_proj.weight"],
            config.intermediate,
            h,
        )
        hidden = binary(hidden, down, trt.ElementWiseOperation.SUM)
        debug(hidden, f"layer_{i}")
    hidden = norm(hidden, weights["norm.weight"], h)
    hidden.name = "draft_hidden"
    network.mark_output(hidden)
    # Prepared drafts can have their own LM-head activation scale. Reusing
    # a quantized target head would silently discard that calibration.
    lm_head = weights.get("lm_head.weight", target_lm_head)
    logits = cast(linear(hidden, lm_head, h, config.vocab), trt.float32)
    logits.name = "draft_logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Qwen3.5-MoE DFlash draft engine build failed")
    return bytes(plan)
