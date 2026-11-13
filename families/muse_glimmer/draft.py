# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-owned block-16 DFlash V1/V2 graphs, without ONNX or custom GPU kernels.

Confirmed target features are reprojected on each round; proposal state never
enters the persistent target cache. The original packed target embedding and
MXFP8 head are inherited without conversion to a different precision recipe.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import tensorrt as trt

from .quantization import constant as add_constant, linear as quantized_linear, weight_tensor


@dataclass(frozen=True)
class DraftConfig:
    raw: dict
    version: int
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
    query_rows: int = 16

    @classmethod
    def from_dict(cls, raw, target):
        version = {("MuseGlimmerAssistantModel",): 1, ("DFlash2DraftModel",): 2}.get(
            tuple(raw.get("architectures", ())))
        policy = raw.get("dflash_config") or raw
        layers = tuple(policy.get("target_layer_ids", ()))
        h, v = int(raw["hidden_size"]), int(raw.get("vocab_size", target.vocab_size))
        heads, kv, dim = (int(raw[k]) for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
        n, width = int(raw["num_hidden_layers"]), int(raw["intermediate_size"])
        eps, mask = float(raw.get("rms_norm_eps", 1e-5)), int(policy.get("mask_token_id", -1))
        if (version is None or h != target.hidden_size or v != target.vocab_size
                or policy.get("block_size") != 16 or not layers or len(set(layers)) != len(layers)
                or any(type(i) is not int or not 0 <= i < target.num_hidden_layers for i in layers)
                or raw.get("num_target_layers", target.num_hidden_layers) != target.num_hidden_layers
                or min(h, v, heads, kv, dim, n, width) <= 0 or heads % kv or dim % 2
                or not 0 <= mask < v or not np.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid Muse DFlash companion geometry or target pairing")
        if (raw.get("attention_bias", False) or raw.get("hidden_act", "silu") != "silu"
                or raw.get("is_causal", False) or raw.get("quantization_config")
                or raw.get("layer_types") != ["sliding_attention"] * n):
            raise ValueError("Unsupported Muse DFlash attention or precision contract")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default" or float(rope.get("rope_theta", 0)) <= 1:
            raise ValueError("Muse companions require ordinary RoPE")
        window = int(raw.get("sliding_window", 0)) if version == 2 else 0
        group = int(policy.get("conv_group_size", 1))
        kernel = int(policy.get("conv_kernel_size", 2))
        rank, top_k = int(policy.get("selector_rank", 1)), int(policy.get("selector_top_k", 16))
        if (version == 2 and (window <= 0 or group <= 0 or h % group or kernel != 2
                             or rank <= 0 or top_k != 16 or top_k > v)):
            raise ValueError("Invalid Muse DFlash2 convolution or selector geometry")
        return cls(raw, version, layers, h, v, n, heads, kv, dim, width, mask, eps,
                   window, group, kernel, rank, top_k)


def rotary_tables(raw, head_dim, capacity):
    theta = float(raw["rope_parameters"]["rope_theta"])
    inverse = 1 / np.power(np.float32(theta), np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    phase = np.arange(capacity, dtype=np.float32)[:, None] * inverse[None, :]
    return np.cos(phase), np.sin(phase)


def load_draft_weights(model_dir: Path, config: DraftConfig):
    """Open each source shard once; validate all inherited and dense tensor roles."""
    from safetensors import safe_open

    path = Path(model_dir)
    index_path = path / "model.safetensors.index.json"
    weights = {}
    with ExitStack() as stack:
        readers = {}
        if index_path.is_file():
            index = json.loads(index_path.read_text())["weight_map"]
            if any(Path(name).name != name for name in index.values()):
                raise ValueError("Muse companion weight index must use local shard filenames")
            files = sorted(set(index.values()))
        else:
            files = [p.name for p in sorted(path.glob("*.safetensors"))]
            index = {}
        for filename in files:
            reader = stack.enter_context(safe_open(path / filename, framework="pt", device="cpu"))
            readers[filename] = reader
            if not index_path.is_file():
                for key in reader.keys():
                    if key in index:
                        raise ValueError(f"Duplicate Muse companion tensor: {key}")
                    index[key] = filename
        def load(name, shape, projection=False):
            source = name
            if config.version == 1:
                if name.startswith("fc."):
                    source = "encoder." + name
                if name.startswith("hidden_norm."):
                    source = name.replace("hidden_norm.", "encoder.output_norm_enc.", 1)
            if source not in index:
                raise ValueError(f"Muse companion tensor missing: {source}")
            value = readers[index[source]].get_tensor(source).float().numpy()
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid Muse companion tensor: {source}: {value.shape} != {shape}")
            weights[name] = np.ascontiguousarray(value.T if projection else value, dtype=np.float16)
        h, v, d = config.hidden, config.vocab, config.head_dim
        load("fc.weight", (h, len(config.feature_layers) * h), True)
        load("hidden_norm.weight", (h,))
        load("norm.weight", (h,))
        for i in range(config.layers):
            prefix = f"layers.{i}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                load(f"{prefix}.{norm}.weight", (h,))
            for name, out_width, in_width in (("q_proj",config.heads*d,h), ("k_proj",config.kv_heads*d,h),
                                             ("v_proj",config.kv_heads*d,h), ("o_proj",h,config.heads*d)):
                load(f"{prefix}.self_attn.{name}.weight", (out_width,in_width), True)
            for norm in ("q_norm", "k_norm"):
                load(f"{prefix}.self_attn.{norm}.weight", (d,))
            for name, out_width, in_width in (("gate_proj",config.intermediate,h),
                                             ("up_proj",config.intermediate,h), ("down_proj",h,config.intermediate)):
                load(f"{prefix}.mlp.{name}.weight", (out_width,in_width), True)
            if config.version == 2:
                for conv in ("attention_conv", "mlp_conv"):
                    load(f"{prefix}.{conv}.kernel_projection.weight", (2*config.kernel_size*h//config.group_size,h), True)
                    load(f"{prefix}.{conv}.base_kernel", (2,config.kernel_size,h))
        if config.version == 2:
            load("candidate_selector.hidden_projection.weight", (config.selector_rank,h), True)
            load("candidate_selector.predecessor_codebook", (v,config.selector_rank))
            load("candidate_selector.successor_codebook", (v,config.selector_rank))
        if any(key in index for key in ("lm_head.weight", "embed_tokens.weight")):
            raise ValueError("Muse companions must inherit the target head and embedding")
    return weights


def build_draft_engine(
    config, weights, target_lm_head, target_embedding, max_cache_length, *, verbose=False, debug_layer_outputs=False
):
    """Build the real block draft, with no plugins, ONNX or target substitution."""
    if max_cache_length < config.query_rows:
        raise ValueError("Native Muse-Glimmer DFlash2 draft context is smaller than its proposal block")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 1
    builder.max_threads = min(8, os.cpu_count() or 1)
    h, heads, kv, dim = config.hidden, config.heads, config.kv_heads, config.head_dim
    rows, capacity = config.query_rows, max_cache_length
    feature_width = len(config.feature_layers) * h
    keep_alive = []
    tokens = network.add_input("draft_tokens", trt.int32, (rows,))
    embedding = weight_tensor(network, target_embedding, keep_alive, trt.float16)
    embeddings = network.add_gather(embedding, tokens, 0).get_output(0)
    positions = network.add_input("draft_positions", trt.int32, (rows,))
    features = network.add_input("target_hidden", trt.float16, (capacity, feature_width))
    mask = network.add_input("draft_attention_mask", trt.float32, (rows, capacity + rows))

    def constant(value, dtype=np.float32):
        value = np.asarray(value, dtype=dtype)
        return add_constant(network, value, keep_alive)

    def cast(value, dtype):
        return value if value.dtype == dtype else network.add_cast(value, dtype).get_output(0)

    def reshape(value, shape):
        op = network.add_shuffle(value)
        op.reshape_dims = shape
        return op.get_output(0)

    def binary(a, b, operation):
        return network.add_elementwise(a, b, operation).get_output(0)

    def linear(value, weight, in_width, out_width, nrows=rows):
        rhs = add_constant(network, np.ascontiguousarray(weight, dtype=np.float16), keep_alive)
        return network.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                           rhs, trt.MatrixOperation.NONE).get_output(0)

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

    hidden = cast(embeddings, trt.float32) if config.version == 2 else embeddings
    # The Muse DFlash projector must remain FP32 through RMSNorm. Features can
    # exceed FP16 FC output range; promoting an already overflowed result fails.
    fused = network.add_matrix_multiply(
        cast(features, trt.float32), trt.MatrixOperation.NONE,
        constant(weights["fc.weight"].astype(np.float32)), trt.MatrixOperation.NONE,
    ).get_output(0)
    fused = cast(norm(fused, weights["hidden_norm.weight"], h), trt.float16)
    debug(fused, "features")
    for i in range(config.layers):
        prefix = f"layers.{i}"
        attn = f"{prefix}.self_attn"
        residual = hidden
        normalized = cast(norm(hidden, weights[f"{prefix}.input_layernorm.weight"], h), trt.float16)
        if config.version == 2:
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
        hidden = (convolution(projected, attention_post,
                              weights[f"{prefix}.attention_conv.base_kernel"][1], residual)
                  if config.version == 2 else binary(residual, projected, trt.ElementWiseOperation.SUM))
        debug(hidden, f"attention_{i}")
        residual = hidden
        normalized = cast(norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], h), trt.float16)
        if config.version == 2:
            normalized, mlp_post = conv_prepare(normalized, f"{prefix}.mlp_conv")
        debug(normalized, f"mlp_pre_{i}")
        gate = linear(normalized, weights[f"{prefix}.mlp.gate_proj.weight"], h, config.intermediate)
        up = linear(normalized, weights[f"{prefix}.mlp.up_proj.weight"], h, config.intermediate)
        # Source F.silu evaluates sigmoid and product in FP32 for FP16
        # inputs, then rounds once before the separate SwiGLU product.
        gate32 = cast(gate, trt.float32)
        sigmoid = network.add_activation(gate32, trt.ActivationType.SIGMOID).get_output(0)
        gate = cast(binary(gate32, sigmoid, trt.ElementWiseOperation.PROD), trt.float16)
        down = linear(
            binary(gate, up, trt.ElementWiseOperation.PROD),
            weights[f"{prefix}.mlp.down_proj.weight"],
            config.intermediate,
            h,
        )
        hidden = (convolution(down, mlp_post, weights[f"{prefix}.mlp_conv.base_kernel"][1], residual)
                  if config.version == 2 else binary(residual, down, trt.ElementWiseOperation.SUM))
        debug(hidden, f"layer_{i}")
    hidden = cast(norm(hidden, weights["norm.weight"], h), trt.float16)
    if config.version == 2:
        hidden = network.add_slice(hidden, (1, 0), (rows - 1, h), (1, 1)).get_output(0)
        projected = linear(hidden, weights["candidate_selector.hidden_projection.weight"],
                           h, config.selector_rank, rows - 1)
        projected.name = "projected_hidden"
        network.mark_output(projected)
    debug(hidden, "prediction_hidden")
    logits = cast(quantized_linear(network, hidden, target_lm_head, keep_alive), trt.float32)
    debug(logits, "unary_logits")
    if config.version == 2:
        candidates = network.add_topk(logits, trt.TopKOperation.MAX, config.selector_top_k, 1 << 1)
        values, ids = candidates.get_output(0), candidates.get_output(1)
        values.name, ids.name = "unary_values", "candidate_ids"
        network.mark_output(values)
        network.mark_output(ids)
    else:
        logits.name = "draft_logits"
        network.mark_output(logits)
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("Native Muse-Glimmer DFlash2 draft engine build failed")
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
    build_config.builder_optimization_level = 1
    builder.max_threads = min(8, os.cpu_count() or 1)
    # The predecessor/hidden product has FP32 information beyond FP16/TF32.
    # Preserve the source FP32 selector reduction rather than rounding that
    # product to TF32 before the successor-codebook dot product.
    build_config.clear_flag(trt.BuilderFlag.TF32)
    keep_alive = []
    k, rank = config.selector_top_k, config.selector_rank
    previous = network.add_input("previous_token", trt.int32, (1,))
    ids = network.add_input("candidate_ids", trt.int32, (k,))
    unary = network.add_input("unary_values", trt.float32, (1, k))
    projected = network.add_input("projected_hidden", trt.float16, (1, rank))

    def table(name):
        value = weights[f"candidate_selector.{name}_codebook"]
        return add_constant(network, np.asarray(value, dtype=np.float16), keep_alive)

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
        raise RuntimeError("Native Muse-Glimmer DFlash2 selector build failed")
    return bytes(plan)
