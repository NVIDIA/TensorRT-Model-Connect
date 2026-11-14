# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Nemotron paired drafts: prepared projections and transformer math.

The draft reprojects retained target features, rather than maintaining a second
persistent KV cache. This preserves the checkpoint computation while keeping
rejected proposals out of the recurrent target state.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph_ops
from .checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors, _transpose_2d
from .prepared_quantization import PreparedLinear


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
    query_rows: int
    mask_token: int
    epsilon: float
    sliding_window: int
    markov_rank: int

    @classmethod
    def from_dict(cls, raw, target, variant):
        if variant not in {"dflash", "dspark", "dspark_tree"}:
            raise ValueError("Unsupported native Nemotron paired variant")
        dspark = variant.startswith("dspark")
        policy = raw.get("dspark_config") or raw.get("dflash_config") or {}
        feature_layers = tuple(policy.get("target_layer_ids", raw.get("target_layer_ids", ())))
        architecture = "Qwen3DSparkModel" if dspark else "DFlashDraftModel"
        block = 8 if dspark else 16
        hidden, vocab = int(raw["hidden_size"]), int(raw["vocab_size"])
        if (
            raw.get("architectures") != [architecture]
            or hidden != target["hidden_size"]
            or vocab != target["vocab_size"]
            or not feature_layers
            or len(set(feature_layers)) != len(feature_layers)
            or any(
                type(i) is not int or not 0 <= i < target["num_hidden_layers"]
                for i in feature_layers
            )
            or policy.get("block_size", raw.get("block_size", block)) != block
            or policy.get("sample_from_anchor", raw.get("sample_from_anchor")) is not False
        ):
            raise ValueError("Nemotron companion metadata does not match its target")
        if raw.get("attention_bias", False) or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("Nemotron draft requires bias-free projections and SwiGLU")
        if raw.get("has_lm_head", False) or not raw.get("has_embed_tokens", True):
            raise ValueError("Nemotron draft requires its own embeddings and the target LM head")
        if dspark:
            if (
                policy.get("causal") is not True
                or raw.get("sliding_window") != 1024
                or raw.get("attention_sink_bias") is not True
                or raw.get("dspark_bonus_anchor") is not True
                or raw.get("markov_head_type") != "vanilla"
            ):
                raise ValueError(
                    "Nemotron DSpark requires the published anchor/SWA/Markov contract"
                )
        elif policy.get("causal", False) or raw.get("attention_sink_bias", False):
            raise ValueError("Nemotron DFlash requires bidirectional attention without sinks")
        heads, kv = int(raw["num_attention_heads"]), int(raw["num_key_value_heads"])
        if heads <= 0 or kv <= 0:
            raise ValueError("Invalid Nemotron draft head counts")
        dim = int(raw.get("head_dim", hidden // heads))
        layers, width = int(raw["num_hidden_layers"]), int(raw["intermediate_size"])
        mask = int(policy.get("mask_token_id", raw.get("mask_token_id", -1)))
        eps = float(raw.get("rms_norm_eps", 1e-6))
        rank = int(raw.get("markov_rank", 0) or 0)
        if (
            min(hidden, vocab, heads, kv, dim, layers, width) <= 0
            or heads % kv
            or dim % 2
            or not 0 <= mask < vocab
            or not np.isfinite(eps)
            or eps <= 0
            or dspark
            and rank <= 0
        ):
            raise ValueError("Invalid Nemotron draft geometry")
        return cls(
            raw,
            variant,
            feature_layers,
            hidden,
            vocab,
            layers,
            heads,
            kv,
            dim,
            width,
            block,
            block + 1 if dspark else block,
            mask,
            eps,
            1024 if dspark else 0,
            rank,
        )


def rotary_tables(raw, head_dim, capacity):
    """Official YaRN NTK-by-parts interpolation, or ordinary RoPE, in FP32."""
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    kind = rope.get("rope_type", rope.get("type", "default"))
    theta = float(rope.get("rope_theta", raw.get("rope_theta", 10000.0)))
    if head_dim <= 0 or head_dim % 2 or not np.isfinite(theta) or theta <= 1:
        raise ValueError("Invalid Nemotron draft rotary geometry")
    indices = np.arange(head_dim // 2, dtype=np.float32)
    inverse = 1.0 / np.power(np.float32(theta), 2 * indices / head_dim)
    amplitude = 1.0
    if kind == "yarn":
        factor = float(rope["factor"])
        original = int(rope["original_max_position_embeddings"])
        fast, slow = float(rope.get("beta_fast", 32)), float(rope.get("beta_slow", 1))
        if min(factor, original, fast, slow) <= 0 or not np.isfinite([factor, fast, slow]).all():
            raise ValueError("Invalid Nemotron draft YaRN scaling")

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
            raise ValueError("Invalid Nemotron draft YaRN attention factor")
    elif kind != "default":
        raise ValueError(f"Unsupported Nemotron draft rotary scaling: {kind}")
    phases = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]
    return (
        (np.cos(phases) * amplitude).astype(np.float32),
        (np.sin(phases) * amplitude).astype(np.float32),
    )


def load_draft_weights(model_dir: Path, config: DraftConfig):
    """Reuse only this family's exact prepared safetensors storage/scales."""
    readers = _open_safetensors(model_dir)
    weights = {}

    def load(name, shape, projection=False):
        value = _load_tensor(readers, name)
        if value.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {value.shape}")
        weights[name] = _transpose_2d(value, name) if projection else value.astype(np.float32)

    h, v, d = config.hidden, config.vocab, config.head_dim
    load("embed_tokens.weight", (v, h))
    load("fc.weight", (h, len(config.feature_layers) * h), True)
    load("hidden_norm.weight", (h,))
    load("norm.weight", (h,))
    for i in range(config.layers):
        prefix = f"layers.{i}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            load(f"{prefix}.{norm}.weight", (h,))
        for projection, out_size, in_size in (
            ("q_proj", config.heads * d, h),
            ("k_proj", config.kv_heads * d, h),
            ("v_proj", config.kv_heads * d, h),
            ("o_proj", h, config.heads * d),
        ):
            load(f"{prefix}.self_attn.{projection}.weight", (out_size, in_size), True)
        for norm in ("q_norm", "k_norm"):
            load(f"{prefix}.self_attn.{norm}.weight", (d,))
        for projection, out_size, in_size in (
            ("gate_proj", config.intermediate, h),
            ("up_proj", config.intermediate, h),
            ("down_proj", h, config.intermediate),
        ):
            load(f"{prefix}.mlp.{projection}.weight", (out_size, in_size), True)
        if config.variant.startswith("dspark"):
            load(f"{prefix}.self_attn.attention_sink_bias", (config.heads,))
    if config.markov_rank:
        load("markov_head.markov_w1.weight", (v, config.markov_rank))
        load("markov_head.markov_w2.weight", (v, config.markov_rank), True)
    if _has_tensor(readers, "lm_head.weight"):
        raise ValueError("Nemotron paired draft must use the actual untied target LM head")
    return weights


def build_draft_engine(
    config, weights, target_lm_head, max_cache_length, *, verbose=False, debug_layer_outputs=False
):
    """Build the real block draft, with no plugins, ONNX or target substitution."""
    if max_cache_length < config.query_rows:
        raise ValueError("Native Nemotron draft context is smaller than its proposal block")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    h, heads, kv, dim = config.hidden, config.heads, config.kv_heads, config.head_dim
    rows, capacity = config.query_rows, max_cache_length
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
        if isinstance(weight, PreparedLinear):
            return weight.matmul(
                network, value, in_width, out_width, activation_rows=nrows
            )
        return graph_ops.add_matmul_rhs_constant(
            network, value, in_width, out_width, weight, dtype=np.float16, rank=2
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

    hidden = network.add_gather(
        constant(weights["embed_tokens.weight"], np.float16), ids, 0
    ).get_output(0)
    # Nemotron companions explicitly use their native prepared FC precision.
    fused = linear(features, weights["fc.weight"], feature_width, h, capacity)
    fused = norm(fused, weights["hidden_norm.weight"], h)
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
        sink = weights.get(f"{attn}.attention_sink_bias")
        if sink is not None:
            sink_tensor = constant(
                np.broadcast_to(sink.reshape(kv, group, 1, 1), (kv, group, rows, 1)).copy()
            )
            concat = network.add_concatenation([scores, sink_tensor])
            concat.axis = 3
            scores = concat.get_output(0)
        probabilities = network.add_softmax(scores)
        probabilities.axes = 1 << 3
        probabilities = probabilities.get_output(0)
        if sink is not None:
            probabilities = network.add_slice(
                probabilities, (0, 0, 0, 0), (kv, group, rows, total), (1, 1, 1, 1)
            ).get_output(0)
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
    logits = cast(linear(hidden, target_lm_head, h, config.vocab), trt.float32)
    logits.name = "draft_logits"
    network.mark_output(logits)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Nemotron draft engine build failed")
    return bytes(plan)


def build_markov_engine(config, weights, *, verbose=False):
    """Compute the source FP32 logit bias conditioned on the previous token.

    The prepared W2 remains packed in the graph. Its dequantized values and W1
    are FP16, matching the upstream head export, with FP32 accumulation/output.
    """
    if not config.variant.startswith("dspark") or config.markov_rank <= 0:
        raise ValueError("Native Nemotron Markov head requires a DSpark companion")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    previous = network.add_input("previous_token", trt.int32, (1,))
    table = graph_ops.add_constant(
        network,
        (config.vocab, config.markov_rank),
        weights["markov_head.markov_w1.weight"],
        dtype=np.float16,
    )
    row = network.add_gather(table, previous, 0).get_output(0)
    weight = weights["markov_head.markov_w2.weight"]
    if isinstance(weight, PreparedLinear):
        bias = weight.matmul(network, row, config.markov_rank, config.vocab, output_fp32=True)
    else:
        rhs = graph_ops.add_constant(
            network, (config.markov_rank, config.vocab), weight, dtype=np.float16
        )
        bias = network.add_matrix_multiply(
            network.add_cast(row, trt.float32).get_output(0),
            trt.MatrixOperation.NONE,
            network.add_cast(rhs, trt.float32).get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
    bias.name = "markov_bias"
    network.mark_output(bias)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Nemotron Markov engine build failed")
    return bytes(plan)
