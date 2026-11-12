# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a TrOCR bundle: a ViT encoder engine and a one-token TrOCR decoder engine.

``encoder.plan`` runs once per image. It encodes the preprocessed pixels and
projects the final encoder states through every decoder layer's
cross-attention key and value weights, so the decoder never recomputes them:
``cross_key`` / ``cross_value`` are ``[layers, heads, patches + 1, head_dim]``.

``decoder.plan`` runs once per generated token. It takes the token id, its
position and the self-attention key/value cache ``[layers, heads, capacity,
head_dim]`` (rows at and after the position are masked out) and returns the
next-token logits and the token's own key/value rows, which the runtime writes
into the cache at that position. Greedy decoding starts from the checkpoint's
``decoder_start_token_id`` and stops at ``eos_token_id``.

All weights are baked into the plans. ``runtime.json`` carries the
preprocessing, the special token ids and the decode capacity;
``id_to_token.json`` lists the byte-level BPE token of every id, which the
runtime turns generated ids into text with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, graph as g

_TASK = "images_text_to_text"
# The checkpoint's stored precision, and the only one built: an fp16 graph kept
# the reference tokens on the E2E lines but moved the logits two to three times
# as far as the reference's own float16 run does.
_PRECISION = "fp32"
_DTYPE = np.float32
_ENCODER = "encoder."
_DECODER = "decoder.model.decoder."
# The learned decoder position table starts two rows in (fairseq's padding offset).
_POSITION_OFFSET = 2


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise RuntimeError("trocr requires safetensors in the build environment") from error
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"trocr safetensors checkpoint is missing: {path}")
    return dict(load_file(str(path)))


class _Weights:
    """Checkpoint tensors by name; every lookup is checked once, at use."""

    def __init__(self, tensors: dict) -> None:
        self._tensors = tensors
        self._used: set[str] = set()

    def __call__(self, name: str) -> np.ndarray:
        if name not in self._tensors:
            raise KeyError(f"trocr checkpoint is missing tensor {name}")
        self._used.add(name)
        return np.asarray(self._tensors[name])

    def check_all_used(self, ignored_prefixes: tuple[str, ...]) -> None:
        unused = sorted(
            name for name in self._tensors
            if name not in self._used and not name.startswith(ignored_prefixes))
        if unused:
            raise ValueError("trocr does not know these checkpoint tensors: " + ", ".join(unused))


def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would cap an fp32 build near 1e-3 relative error in every matmul.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, name: str) -> bytes:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"trocr {name} engine build failed")
    return bytes(plan)


def _encoder_layer(network, hidden, w, cfg, index: int):
    enc = cfg.encoder
    prefix = f"{_ENCODER}encoder.layer.{index}."
    normed = g.add_layer_norm(network, hidden, w(prefix + "layernorm_before.weight"),
                              w(prefix + "layernorm_before.bias"), enc.layer_norm_eps)

    def projection(name):
        bias = w(f"{prefix}attention.attention.{name}.bias") if enc.qkv_bias else None
        return g.add_linear(network, normed, w(f"{prefix}attention.attention.{name}.weight"),
                            bias, dtype=_DTYPE)

    scale = g.add_scalar(network, 2, enc.head_dim ** -0.5, _DTYPE)
    query = g.to_heads(network, g.add_prod(network, projection("query"), scale),
                       enc.num_heads, enc.head_dim)
    key = g.to_heads(network, projection("key"), enc.num_heads, enc.head_dim)
    value = g.to_heads(network, projection("value"), enc.num_heads, enc.head_dim)
    context = g.from_heads(network, g.add_attention(network, query, key, value))
    attention = g.add_linear(network, context, w(prefix + "attention.output.dense.weight"),
                             w(prefix + "attention.output.dense.bias"), dtype=_DTYPE)
    hidden = g.add_sum(network, hidden, attention)
    normed = g.add_layer_norm(network, hidden, w(prefix + "layernorm_after.weight"),
                              w(prefix + "layernorm_after.bias"), enc.layer_norm_eps)
    inner = g.add_linear(network, normed, w(prefix + "intermediate.dense.weight"),
                         w(prefix + "intermediate.dense.bias"), dtype=_DTYPE)
    inner = g.add_linear(network, g.add_gelu(network, inner, dtype=_DTYPE),
                         w(prefix + "output.dense.weight"), w(prefix + "output.dense.bias"),
                         dtype=_DTYPE)
    return g.add_sum(network, hidden, inner)


def _cross_projection(network, states, w, cfg, kind: str):
    """Every decoder layer's cross-attention ``kind`` projection of the encoder states.

    One matmul for all layers, laid out ``[layers, heads, patches + 1, head_dim]``.
    """
    dec = cfg.decoder
    weights = [w(f"{_DECODER}layers.{i}.encoder_attn.{kind}_proj.weight")
               for i in range(dec.num_layers)]
    biases = [w(f"{_DECODER}layers.{i}.encoder_attn.{kind}_proj.bias")
              for i in range(dec.num_layers)]
    rows = g.add_linear(network, states, np.concatenate(weights, axis=0),
                        np.concatenate(biases, axis=0), dtype=_DTYPE)
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (cfg.encoder.sequence_length, dec.num_layers, dec.num_heads,
                          dec.head_dim)
    layer.second_transpose = (1, 2, 0, 3)
    return layer.get_output(0)


def build_encoder_engine(w: _Weights, cfg, *, verbose: bool = False) -> bytes:
    """``pixel_values [1, 3, size, size]`` -> ``cross_key``, ``cross_value``."""
    enc = cfg.encoder
    builder, network, config = _builder(verbose)
    pixels = network.add_input("pixel_values", trt.float32,
                               (1, 3, enc.image_size, enc.image_size))
    patches = network.add_convolution_nd(
        pixels, enc.hidden_size,
        (enc.patch_size, enc.patch_size),
        trt.Weights(np.ascontiguousarray(
            w(_ENCODER + "embeddings.patch_embeddings.projection.weight").astype(_DTYPE))),
        trt.Weights(np.ascontiguousarray(
            w(_ENCODER + "embeddings.patch_embeddings.projection.bias").astype(_DTYPE))))
    patches.stride_nd = (enc.patch_size, enc.patch_size)
    flat = network.add_shuffle(patches.get_output(0))
    flat.reshape_dims = (enc.hidden_size, enc.grid * enc.grid)
    flat.second_transpose = (1, 0)
    # The class token is prepended, so row 0 of the position table is its row.
    class_token = g.add_constant(network, (1, enc.hidden_size),
                                 w(_ENCODER + "embeddings.cls_token"), dtype=_DTYPE)
    concat = network.add_concatenation([class_token, flat.get_output(0)])
    concat.axis = 0
    positions = w(_ENCODER + "embeddings.position_embeddings").reshape(
        enc.sequence_length, enc.hidden_size)
    hidden = g.add_sum(network, concat.get_output(0),
                       g.add_constant(network, positions.shape, positions, dtype=_DTYPE))
    for index in range(enc.num_layers):
        hidden = _encoder_layer(network, hidden, w, cfg, index)
    states = g.add_layer_norm(network, hidden, w(_ENCODER + "layernorm.weight"),
                              w(_ENCODER + "layernorm.bias"), enc.layer_norm_eps)
    for kind, name in (("k", "cross_key"), ("v", "cross_value")):
        output = _cross_projection(network, states, w, cfg, kind)
        output.name = name
        network.mark_output(output)
    return _serialize(builder, network, config, "encoder")


def _layer_slice(network, tensor, index: int):
    """Layer ``index`` of a ``[layers, heads, rows, head_dim]`` tensor, as ``[1, heads, rows, head_dim]``."""
    _, heads, rows, head_dim = (int(value) for value in tensor.shape)
    return network.add_slice(tensor, start=(index, 0, 0, 0), shape=(1, heads, rows, head_dim),
                             stride=(1, 1, 1, 1)).get_output(0)


def _self_attention(network, query, key, value, past_key, past_value, past_valid):
    """Attention over the cached rows the mask keeps plus this token's own row."""
    capacity = int(past_key.shape[2])
    heads = int(query.shape[1])
    past_scores = network.add_matrix_multiply(query, trt.MatrixOperation.NONE, past_key,
                                              trt.MatrixOperation.TRANSPOSE).get_output(0)
    blocked = g.add_scalar(network, 4, -np.inf, _DTYPE)
    past_scores = network.add_select(past_valid, past_scores, blocked).get_output(0)
    own_score = network.add_matrix_multiply(query, trt.MatrixOperation.NONE, key,
                                            trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_concatenation([past_scores, own_score])
    scores.axis = 3
    weights = g.add_softmax_last(network, scores.get_output(0))
    past_weights = network.add_slice(weights, start=(0, 0, 0, 0), shape=(1, heads, 1, capacity),
                                     stride=(1, 1, 1, 1)).get_output(0)
    own_weight = network.add_slice(weights, start=(0, 0, 0, capacity), shape=(1, heads, 1, 1),
                                   stride=(1, 1, 1, 1)).get_output(0)
    past = network.add_matrix_multiply(past_weights, trt.MatrixOperation.NONE, past_value,
                                       trt.MatrixOperation.NONE).get_output(0)
    return g.add_sum(network, past, g.add_prod(network, own_weight, value))


def _decoder_layer(network, hidden, w, cfg, index, caches, past_valid):
    dec = cfg.decoder
    prefix = f"{_DECODER}layers.{index}."
    cache_key, cache_value, cross_key, cross_value = caches
    scale = g.add_scalar(network, 2, dec.head_dim ** -0.5, _DTYPE)

    def project(source, name):
        return g.add_linear(network, source, w(prefix + name + ".weight"),
                            w(prefix + name + ".bias"), dtype=_DTYPE)

    def heads(rows):
        return g.to_heads(network, rows, dec.num_heads, dec.head_dim)

    query = heads(g.add_prod(network, project(hidden, "self_attn.q_proj"), scale))
    key = heads(project(hidden, "self_attn.k_proj"))
    value = heads(project(hidden, "self_attn.v_proj"))
    context = _self_attention(network, query, key, value, _layer_slice(network, cache_key, index),
                              _layer_slice(network, cache_value, index), past_valid)
    attended = project(g.from_heads(network, context), "self_attn.out_proj")
    hidden = g.add_layer_norm(network, g.add_sum(network, hidden, attended),
                              w(prefix + "self_attn_layer_norm.weight"),
                              w(prefix + "self_attn_layer_norm.bias"), dec.layer_norm_eps)

    query = heads(g.add_prod(network, project(hidden, "encoder_attn.q_proj"), scale))
    context = g.add_attention(network, query, _layer_slice(network, cross_key, index),
                              _layer_slice(network, cross_value, index))
    attended = project(g.from_heads(network, context), "encoder_attn.out_proj")
    hidden = g.add_layer_norm(network, g.add_sum(network, hidden, attended),
                              w(prefix + "encoder_attn_layer_norm.weight"),
                              w(prefix + "encoder_attn_layer_norm.bias"), dec.layer_norm_eps)

    inner = g.add_gelu(network, project(hidden, "fc1"), dtype=_DTYPE)
    hidden = g.add_layer_norm(network, g.add_sum(network, hidden, project(inner, "fc2")),
                              w(prefix + "final_layer_norm.weight"),
                              w(prefix + "final_layer_norm.bias"), dec.layer_norm_eps)
    return hidden, key, value


def build_decoder_engine(w: _Weights, cfg, capacity: int, *, verbose: bool = False) -> bytes:
    """One token at ``position`` -> next-token ``logits`` and the token's ``new_key``/``new_value``."""
    dec, enc = cfg.decoder, cfg.encoder
    builder, network, config = _builder(verbose)
    token = network.add_input("token_id", trt.int32, (1,))
    position = network.add_input("position", trt.int32, (1,))
    cache_shape = (dec.num_layers, dec.num_heads, capacity, dec.head_dim)
    cross_shape = (dec.num_layers, dec.num_heads, enc.sequence_length, dec.head_dim)
    caches = (
        network.add_input("cache_key", trt.float32, cache_shape),
        network.add_input("cache_value", trt.float32, cache_shape),
        network.add_input("cross_key", trt.float32, cross_shape),
        network.add_input("cross_value", trt.float32, cross_shape),
    )

    # One table serves the token lookup and, transposed, the tied output projection.
    table = g.add_constant(network, (dec.vocab_size, dec.hidden_size),
                           w(_DECODER + "embed_tokens.weight"), dtype=_DTYPE)
    hidden = network.add_gather(table, token, 0).get_output(0)
    offset = network.add_constant((1,), np.array([_POSITION_OFFSET], dtype=np.int32))
    row = network.add_elementwise(position, offset.get_output(0),
                                  trt.ElementWiseOperation.SUM).get_output(0)
    positions = w(_DECODER + "embed_positions.weight")
    if positions.shape[0] < capacity + _POSITION_OFFSET:
        raise ValueError("trocr position table is shorter than the decode capacity")
    hidden = g.add_sum(network, hidden, network.add_gather(
        g.add_constant(network, positions.shape, positions, dtype=_DTYPE), row, 0).get_output(0))
    hidden = g.add_layer_norm(network, hidden, w(_DECODER + "layernorm_embedding.weight"),
                              w(_DECODER + "layernorm_embedding.bias"), dec.layer_norm_eps)

    # Cached rows before the position are this sequence's earlier tokens; rows
    # at or after it are stale and must not be attended to.
    rows = network.add_constant((capacity,), np.arange(capacity, dtype=np.int32)).get_output(0)
    valid = network.add_elementwise(rows, position, trt.ElementWiseOperation.LESS).get_output(0)
    valid_layer = network.add_shuffle(valid)
    valid_layer.reshape_dims = (1, 1, 1, capacity)
    past_valid = valid_layer.get_output(0)

    keys, values = [], []
    for index in range(dec.num_layers):
        hidden, key, value = _decoder_layer(network, hidden, w, cfg, index, caches, past_valid)
        keys.append(key)
        values.append(value)

    logits = network.add_matrix_multiply(hidden, trt.MatrixOperation.NONE, table,
                                         trt.MatrixOperation.TRANSPOSE).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for name, rows_per_layer in (("new_key", keys), ("new_value", values)):
        stacked = network.add_concatenation(rows_per_layer)
        stacked.axis = 0
        output = stacked.get_output(0)
        output.name = name
        network.mark_output(output)
    return _serialize(builder, network, config, "decoder")


def _runtime_config(cfg, capacity: int, precision: str) -> dict:
    pre = cfg.preprocess
    return {
        "precision": precision,
        "image_height": pre.height,
        "image_width": pre.width,
        "resample": pre.resample,
        "rescale_factor": pre.rescale_factor,
        "image_mean": list(pre.image_mean),
        "image_std": list(pre.image_std),
        "decoder_start_token_id": cfg.decoder_start_token_id,
        "eos_token_id": cfg.eos_token_id,
        "special_token_ids": list(cfg.tokenizer.special_token_ids),
        "vocab_size": cfg.decoder.vocab_size,
        "decoder_layers": cfg.decoder.num_layers,
        "decoder_heads": cfg.decoder.num_heads,
        "head_dim": cfg.decoder.head_dim,
        "encoder_sequence_length": cfg.encoder.sequence_length,
        "max_sequence_length": capacity,
    }


def build(request, writer) -> None:
    """Build one TrOCR bundle."""
    if request.task != _TASK:
        raise ValueError(f"trocr supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("trocr supports only backend=trt")
    precision = str(request.precision).lower()
    if precision != _PRECISION:
        raise ValueError(f"trocr builds at the checkpoint's precision, {_PRECISION}")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("trocr does not support quantized builds")
    if request.fp32_layers:
        raise NotImplementedError("trocr does not support mixed-precision layers")
    if request.max_batch_size != 1:
        raise NotImplementedError("trocr reads one image per request (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("trocr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("trocr does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("trocr does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("trocr does not support video_num_frames")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("trocr does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    size = cfg.encoder.image_size
    for name, value in (("image_height", request.image_height),
                        ("image_width", request.image_width)):
        if value is not None and int(value) != size:
            # The position table covers exactly the checkpoint's patch grid.
            raise NotImplementedError(f"trocr encodes {size}x{size} images; {name}={value}")
    capacity = cfg.decoder.max_positions
    if request.max_sequence_length is not None:
        capacity = int(request.max_sequence_length)
        if not 2 <= capacity <= cfg.decoder.max_positions:
            raise ValueError(
                f"trocr max_sequence_length must be in [2, {cfg.decoder.max_positions}]")

    weights = _Weights(_open_checkpoint(model_dir))
    verbose = bool(request.verbose)
    encoder_plan = build_encoder_engine(weights, cfg, verbose=verbose)
    decoder_plan = build_decoder_engine(weights, cfg, capacity, verbose=verbose)
    # ViTModel's pooler is not part of the image-to-text graph.
    weights.check_all_used(ignored_prefixes=(_ENCODER + "pooler.",))

    writer.set_header(family="trocr", task=request.task, backend=request.backend)
    writer.add_bytes("encoder.plan", encoder_plan)
    writer.add_bytes("decoder.plan", decoder_plan)
    writer.add_json("runtime.json", _runtime_config(cfg, capacity, precision))
    writer.add_bytes("id_to_token.json", json.dumps(list(cfg.tokenizer.vocab),
                                                    ensure_ascii=False).encode("utf-8"))
