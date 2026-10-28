# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a GTE-v1.5 bundle.

One engine, one output: the final hidden rows. The pooling that turns them
into an embedding is a single row - ``1_Pooling/config.json`` sets
``pooling_mode_cls_token`` - and it is done in the runtime rather than the
graph, so the graph stays a plain encoder and the pooling contract lives
next to the thing that reports it.

The engine runs at one fixed sequence length and takes the attention mask as
an input. The mask is not decoration: a padded row that is not masked out
attends to every real row and changes their output, [CLS] included, so a
build without it would still produce an embedding of the right shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g, rope

_SUPPORTED_TASKS = ("text_to_embedding",)

# Masked-out scores are driven to this before the softmax. The reference
# uses the float32 minimum; this is far below any real score and far above
# the point where exp() underflows the subtract-the-max softmax, so the
# padded columns come out as exactly zero weight without putting a value
# near the edge of the format into the graph.
_MASK_FLOOR = -1.0e4


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError("gte needs safetensors to read its checkpoint") from error
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing gte checkpoint: {path}")
    return load_file(str(path))


def _require_weights(weights: dict) -> None:
    required = ("embeddings.word_embeddings.weight",
                "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias")
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("gte checkpoint is missing expected tensors: " + ", ".join(missing))


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a matmul graph near 0.99998 cosine by default, which hides
    # real graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def build_engine(weights, cfg, *, seq_len, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    work_np = np.float32
    head_dim = cfg["head_dim"]

    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    table = np.asarray(weights["embeddings.word_embeddings.weight"], dtype=work_np)
    hidden = network.add_gather(
        g.add_constant(network, table.shape, table, dtype=work_np), input_ids, 0).get_output(0)

    # No token type embedding: type_vocab_size is 0, so the reference's
    # token-type branch never runs and the word embedding goes straight into
    # the norm. Adding a row-0 shift the way a BERT-shaped encoder does
    # would move every feature by a constant this checkpoint does not carry.
    hidden = g.add_layer_norm(network, hidden, weights["embeddings.LayerNorm.weight"],
                              weights["embeddings.LayerNorm.bias"], cfg["eps"])

    # (1 - mask) * floor, shaped to broadcast over [1, heads, seq, seq]: the
    # mask selects which *keys* may be attended, so it varies along the last
    # axis and is flat along the query axis.
    mask_float = network.add_cast(attention_mask, trt.float32).get_output(0)
    keep = network.add_shuffle(mask_float)
    keep.reshape_dims = (1, 1, 1, seq_len)
    ones = g.add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0), dtype=work_np)
    dropped = network.add_elementwise(ones, keep.get_output(0),
                                      trt.ElementWiseOperation.SUB).get_output(0)
    additive = g.add_product(
        network, dropped,
        g.add_constant(network, (1, 1, 1, 1), np.full((1,), _MASK_FLOOR), dtype=work_np))

    cos_table, sin_table = rope.tables(seq_len, head_dim, cfg["rope_theta"],
                                       cfg["rope_scaling_factor"])
    shape = (1, 1, seq_len, head_dim)
    cos = g.add_constant(network, shape, cos_table.reshape(shape), dtype=work_np)
    sin = g.add_constant(network, shape, sin_table.reshape(shape), dtype=work_np)

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, mask=additive,
                                          cos=cos, sin=sin, dtype=work_np)
    hidden.name = "last_hidden_state"
    network.mark_output(hidden)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("gte engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one GTE-v1.5 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("gte supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("gte supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("gte does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("gte does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("gte does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("gte does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("gte does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("gte does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("gte does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("gte does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("gte does not support MTP speculative decoding")

    # fp32 only, for the same reason the rotary encoder next door is fp32
    # only: no fp16 number has been measured against the reference here, and
    # an accepted-but-unmeasured precision builds a bundle nothing checks.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"gte builds fp32 only; {precision} is not qualified for this family")

    # Checked before the checkpoint is opened: the request is either
    # buildable or it is not, and that does not depend on the weights.
    if request.max_sequence_length is None:
        raise ValueError("gte needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 2:
        raise ValueError("gte needs room for at least [CLS] and [SEP]")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, weights)

    if seq_len > cfg["max_positions"]:
        # The checkpoint states 8192. The NTK-scaled rotary cache the
        # reference builds happens to run to twice that, but the model was
        # published for 8192 and claiming more is inventing a capability.
        raise NotImplementedError(
            f"gte was trained to {cfg['max_positions']} positions, not {seq_len}")

    writer.set_header(family="gte", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, seq_len=seq_len, verbose=bool(request.verbose)))
    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"missing gte tokenizer: {tokenizer}")
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "hidden_size": cfg["hidden_size"],
            "pad_token_id": cfg["pad_token_id"],
        },
    )
