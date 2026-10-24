# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a DeBERTa-v2/v3 bundle.

One engine, one output: the final hidden rows. Which rows are real is the
runtime's business, so the engine takes the attention mask as an input and
returns the whole fixed-length window.

Two things the embeddings here do **not** do, both read from the config
rather than inferred from the tensors:

* no absolute position embeddings, although the table is in the file;
* no token type embeddings, because ``type_vocab_size`` is 0.

So the embedding is the word lookup, a layer norm, and then a multiply by the
mask - the reference zeroes the padded rows at this point, and a build that
skips it carries nonzero rows through every layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g, rel_position

_SUPPORTED_TASKS = ("text_to_token_features",)

# Masked-out scores are driven to this before the softmax: far below any real
# score and far above where a subtract-the-max softmax underflows.
_MASK_FLOOR = -1.0e4


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the checkpoint, whichever of the two formats it is published in.

    The v2 and v3 checkpoints ship ``pytorch_model.bin`` and no safetensors,
    so this family reads a torch state dict. ``weights_only=True`` keeps it to
    tensors rather than arbitrary pickled objects.
    """
    directory = Path(model_dir)
    safetensors = directory / "model.safetensors"
    if safetensors.is_file():
        from safetensors.numpy import load_file

        return load_file(str(safetensors))
    binary = directory / "pytorch_model.bin"
    if not binary.is_file():
        raise FileNotFoundError(
            f"missing deberta_v2 checkpoint: neither {safetensors} nor {binary}")
    import torch

    state = torch.load(str(binary), map_location="cpu", weights_only=True)
    return {key: value.detach().to(torch.float32).numpy() for key, value in state.items()}


def _prefix(weights: dict) -> str:
    """``DebertaV2Model`` saves unprefixed; the pretraining checkpoints do not."""
    for candidate in ("deberta.", ""):
        if f"{candidate}embeddings.word_embeddings.weight" in weights:
            return candidate
    raise KeyError("deberta_v2 checkpoint has no embeddings.word_embeddings.weight")


def _require_weights(weights: dict, prefix: str) -> None:
    required = (f"{prefix}embeddings.word_embeddings.weight",
                f"{prefix}embeddings.LayerNorm.weight",
                f"{prefix}embeddings.LayerNorm.bias",
                f"{prefix}encoder.rel_embeddings.weight",
                f"{prefix}encoder.LayerNorm.weight",
                f"{prefix}encoder.LayerNorm.bias")
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("deberta_v2 checkpoint is missing expected tensors: " + ", ".join(missing))


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
    prefix = cfg["prefix"]
    num_heads = cfg["num_heads"]

    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    table = np.asarray(weights[f"{prefix}embeddings.word_embeddings.weight"], dtype=work_np)
    embedded = network.add_gather(
        g.add_constant(network, table.shape, table, dtype=work_np), input_ids, 0).get_output(0)
    hidden = g.add_layer_norm(network, embedded,
                              weights[f"{prefix}embeddings.LayerNorm.weight"],
                              weights[f"{prefix}embeddings.LayerNorm.bias"], cfg["eps"])

    keep = network.add_cast(attention_mask, trt.float32).get_output(0)
    column = network.add_shuffle(keep)
    column.reshape_dims = (seq_len, 1)
    # The reference zeroes the padded rows right after the embedding norm.
    hidden = g.add_product(network, hidden, column.get_output(0))

    # mask[i] * mask[j]: a pair counts only if both ends are real. Masking the
    # keys alone would leave the padded query rows attending normally.
    row = network.add_shuffle(keep)
    row.reshape_dims = (1, seq_len)
    pair = g.add_product(network, column.get_output(0), row.get_output(0))
    square = network.add_shuffle(pair)
    square.reshape_dims = (1, seq_len, seq_len)
    ones = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0), dtype=work_np)
    dropped = network.add_elementwise(ones, square.get_output(0),
                                      trt.ElementWiseOperation.SUB).get_output(0)
    additive = g.add_product(
        network, dropped,
        g.add_constant(network, (1, 1, 1), np.full((1,), _MASK_FLOOR), dtype=work_np))

    # The relative table is normalised once for the whole encoder, not per
    # layer: norm_rel_ebd names the encoder's own LayerNorm.
    relative = g.add_layer_norm(
        network,
        g.add_constant(
            network,
            np.asarray(weights[f"{prefix}encoder.rel_embeddings.weight"]).shape,
            weights[f"{prefix}encoder.rel_embeddings.weight"], dtype=work_np),
        weights[f"{prefix}encoder.LayerNorm.weight"],
        weights[f"{prefix}encoder.LayerNorm.bias"], cfg["eps"])

    c2p, p2c = rel_position.gather_indices(seq_len, cfg["position_buckets"],
                                           cfg["max_relative_positions"], cfg["span"])
    shape = (num_heads, seq_len, seq_len)
    c2p_index = g.add_constant(network, shape, np.broadcast_to(c2p, shape), dtype=np.int32)
    p2c_index = g.add_constant(network, shape, np.broadcast_to(p2c, shape), dtype=np.int32)

    hidden = encoder_builder.build_layers(network, hidden, relative, weights, cfg,
                                          mask=additive, c2p_index=c2p_index,
                                          p2c_index=p2c_index, dtype=work_np)
    hidden.name = "last_hidden_state"
    network.mark_output(hidden)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("deberta_v2 engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one DeBERTa-v2/v3 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("deberta_v2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("deberta_v2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("deberta_v2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("deberta_v2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("deberta_v2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("deberta_v2 does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("deberta_v2 does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("deberta_v2 does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("deberta_v2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("deberta_v2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("deberta_v2 does not support MTP speculative decoding")

    # fp32 only, for the reason the neighbouring families are fp32 only: no
    # fp16 number has been measured against the reference here, and an
    # accepted-but-unmeasured precision builds a bundle nothing checks.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"deberta_v2 builds fp32 only; {precision} is not qualified for this family")

    if request.max_sequence_length is None:
        raise ValueError("deberta_v2 needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 2:
        raise ValueError("deberta_v2 needs room for at least [CLS] and [SEP]")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    prefix = _prefix(weights)
    _require_weights(weights, prefix)
    cfg = config_module.resolve(model_dir, weights, prefix)
    if seq_len > cfg["max_positions"]:
        raise NotImplementedError(
            f"deberta_v2 was trained to {cfg['max_positions']} positions, not {seq_len}")

    writer.set_header(family="deberta_v2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, seq_len=seq_len, verbose=bool(request.verbose)))
    tokenizer_json, framing = _tokenizer(model_dir)
    writer.add_bytes("tokenizer.json", tokenizer_json)
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "hidden_size": cfg["hidden_size"],
            "pad_token_id": framing["pad_token_id"],
            "cls_token_id": framing["cls_token_id"],
            "sep_token_id": framing["sep_token_id"],
        },
    )


def _tokenizer(model_dir: Path) -> tuple[bytes, dict]:
    """The bundle's tokenizer, plus the three ids the runtime frames with.

    These checkpoints publish ``spm.model`` and no ``tokenizer.json``, so the
    conversion happens here, at build time, where the reference's own loader
    can do it - rather than teaching the runtime a second tokenizer format.

    The framing ids are carried separately because the converted
    ``tokenizer.json`` declares a post-processor with **no** special tokens in
    it: nothing in the tokenizer would add ``[CLS]`` and ``[SEP]``, and this
    checkpoint is trained with both. The runtime adds them from these ids.
    """
    directory = Path(model_dir)
    if not (directory / "tokenizer.json").is_file() and not (directory / "spm.model").is_file():
        raise FileNotFoundError(
            f"deberta_v2 found neither tokenizer.json nor spm.model in {model_dir}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    if not tokenizer.is_fast:
        raise RuntimeError("deberta_v2 needs the fast tokenizer to read this checkpoint")
    framing = {
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
    }
    for name, value in framing.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"deberta_v2 tokenizer does not define {name}")
    return tokenizer.backend_tokenizer.to_str().encode("utf-8"), framing
