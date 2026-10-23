# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a nomic-embed-text bundle.

One engine, one output: the final hidden rows. The mean pooling that turns
them into an embedding is done in the runtime rather than the graph, because
it depends on how many of the padded rows are real and the runtime is what
knows that.

The engine runs at one fixed sequence length and takes the attention mask as
an input. The mask is not decoration: a padded row that is not masked out
attends to every real row and changes their output, so a build without it
would still produce an embedding of the right shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g, rope

_SUPPORTED_TASKS = ("text_to_embedding",)

# Masked-out scores are driven to this before the softmax. It is far below
# any real score and far above the point where exp() underflows the subtract-
# the-max softmax, so the padded columns come out as exactly zero weight
# without putting an inf into the graph.
_MASK_FLOOR = -1.0e4


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:  # pragma: no cover - dependency is pinned
        raise RuntimeError("nomic_bert needs safetensors to read its checkpoint") from error
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing nomic_bert checkpoint: {path}")
    return load_file(str(path))


def _require_weights(weights: dict) -> None:
    required = ("embeddings.word_embeddings.weight",
                "embeddings.token_type_embeddings.weight",
                "emb_ln.weight", "emb_ln.bias")
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("nomic_bert checkpoint is missing expected tensors: " + ", ".join(missing))


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
    width = cfg["hidden_size"]
    head_dim = cfg["head_dim"]

    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    table = np.asarray(weights["embeddings.word_embeddings.weight"], dtype=work_np)
    embedded = network.add_gather(
        g.add_constant(network, table.shape, table, dtype=work_np), input_ids, 0).get_output(0)

    # token_type_ids is never supplied by this model's tokenizer, and the
    # reference then fills it with zeros - so every row gets row 0 of the
    # table. That makes it a constant shift, but it is the checkpoint's
    # constant and dropping it moves every feature.
    token_type = np.asarray(weights["embeddings.token_type_embeddings.weight"],
                            dtype=work_np)[0].reshape(1, width)
    hidden = g.add_sum(network, embedded,
                       g.add_constant(network, (1, width), token_type, dtype=work_np))
    hidden = g.add_layer_norm(network, hidden, weights["emb_ln.weight"],
                              weights["emb_ln.bias"], cfg["eps"])

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

    cos_table, sin_table = rope.tables(seq_len, head_dim, cfg["rotary_base"])
    shape = (1, 1, seq_len, head_dim)
    cos = g.add_constant(network, shape, cos_table.reshape(shape), dtype=work_np)
    sin = g.add_constant(network, shape, sin_table.reshape(shape), dtype=work_np)

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, mask=additive,
                                          cos=cos, sin=sin, dtype=work_np)
    hidden.name = "last_hidden_state"
    network.mark_output(hidden)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("nomic_bert engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one nomic-embed-text bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("nomic_bert supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("nomic_bert supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("nomic_bert does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("nomic_bert does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("nomic_bert does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("nomic_bert does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("nomic_bert does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("nomic_bert does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("nomic_bert does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("nomic_bert does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("nomic_bert does not support MTP speculative decoding")

    # fp32 only, for the same reason the three vision families next door are
    # fp32 only: no fp16 number has been measured against the reference here,
    # and an accepted-but-unmeasured precision builds a bundle nothing checks.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"nomic_bert builds fp32 only; {precision} is not qualified for this family")

    # Checked before the checkpoint is opened: the request is either
    # buildable or it is not, and that does not depend on the weights.
    if request.max_sequence_length is None:
        raise ValueError("nomic_bert needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 2:
        raise ValueError("nomic_bert needs room for at least [CLS] and [SEP]")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, weights)

    if seq_len > cfg["max_positions"]:
        # The rotary table would extend past where the checkpoint was
        # trained. Building it is easy and claiming it works is not.
        raise NotImplementedError(
            f"nomic_bert was trained to {cfg['max_positions']} positions, not {seq_len}")

    writer.set_header(family="nomic_bert", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, seq_len=seq_len, verbose=bool(request.verbose)))
    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"missing nomic_bert tokenizer: {tokenizer}")
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "hidden_size": cfg["hidden_size"],
            "pad_token_id": cfg["pad_token_id"],
        },
    )
