# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an ESM-2 bundle.

One engine, one output: the vocabulary logits for every row of the window.
Which rows the caller asked about is the runtime's business, so the engine
returns the whole fixed-length window and the pipeline selects from it.

The embedding is where this model stops looking like a BERT:

* ``embeddings.position_embeddings.weight`` is in the file - 1026 rows of
  it - and is **not** used. ``position_embedding_type`` is ``rotary``, so
  position enters only as the rotation inside each layer. A reader who wires
  the table up the way every other BERT wires it up gets a model that loads
  every tensor, runs, and is wrong;
* ``token_dropout`` rescales the *entire* embedding by a factor that depends
  on how many ``<mask>`` ids the input carries. It is therefore computed in
  the graph from ``input_ids``, not folded into a constant at build time.
  Folding it would be correct only for inputs with no mask, which is the one
  kind of input this task does not get.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g

_SUPPORTED_TASKS = ("masked_text_to_token_scores",)

# Masked-out scores are driven to this before the softmax: far below any real
# score and far above where a subtract-the-max softmax underflows.
_MASK_FLOOR = -1.0e4

# ESM's own constant: the fraction of tokens replaced by <mask> during
# training, 0.15 * 0.8. It is hard-coded in the reference for every ESM run.
_MASK_RATIO_TRAIN = 0.15 * 0.8


def _open_checkpoint(model_dir: Path) -> dict:
    directory = Path(model_dir)
    safetensors = directory / "model.safetensors"
    if safetensors.is_file():
        from safetensors.numpy import load_file

        return load_file(str(safetensors))
    binary = directory / "pytorch_model.bin"
    if not binary.is_file():
        raise FileNotFoundError(f"missing esm checkpoint: neither {safetensors} nor {binary}")
    import torch

    state = torch.load(str(binary), map_location="cpu", weights_only=True)
    return {key: value.detach().to(torch.float32).numpy() for key, value in state.items()}


def _prefix(weights: dict) -> str:
    """``EsmModel`` saves unprefixed; ``EsmForMaskedLM`` saves under ``esm.``."""
    for candidate in ("esm.", ""):
        if f"{candidate}embeddings.word_embeddings.weight" in weights:
            return candidate
    raise KeyError("esm checkpoint has no embeddings.word_embeddings.weight")


def _require_weights(weights: dict, prefix: str) -> None:
    required = (f"{prefix}embeddings.word_embeddings.weight",
                f"{prefix}encoder.emb_layer_norm_after.weight",
                f"{prefix}encoder.emb_layer_norm_after.bias",
                "lm_head.dense.weight", "lm_head.dense.bias",
                "lm_head.layer_norm.weight", "lm_head.layer_norm.bias",
                "lm_head.bias")
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("esm checkpoint is missing expected tensors: " + ", ".join(missing))


def _decoder_weight(weights: dict, prefix: str) -> np.ndarray:
    """The lm_head projection, which this checkpoint does not store.

    ``lm_head.decoder.weight`` is tied to the word embedding table and is
    absent from the file. Reading it as a missing tensor and zero-filling, or
    refusing the checkpoint, would both be wrong: the reference ties them, so
    the table *is* the projection. Its shape is already ``[vocab, hidden]``,
    which is the layout ``add_linear`` wants.
    """
    stored = weights.get("lm_head.decoder.weight")
    if stored is not None:
        return np.asarray(stored, dtype=np.float32)
    return np.asarray(weights[f"{prefix}embeddings.word_embeddings.weight"], dtype=np.float32)


def _rotary_inv_freq(weights: dict, cfg: dict) -> np.ndarray:
    """The frequency schedule, taken from the checkpoint when it ships one.

    This is **not** the schedule recomputed from the head width, and the
    difference is not cosmetic. ESM registers ``inv_freq`` as a persistent
    buffer, so it is written into the checkpoint and read back by
    ``from_pretrained``; the model then rotates with the stored values, not
    with freshly computed ones. In this checkpoint the stored values are the
    RoFormer schedule rounded to float16 - ``0.42163086`` where the exact
    schedule says ``0.4216965`` - a relative error of 4e-4.

    That rounding is harmless at position 0 and grows with position, because
    the angle is ``position * frequency``. By the end of a 130-residue
    sequence it is worth about 0.014 radians, which moves the logits well
    past any tolerance a parity gate would set. Recomputing the schedule
    therefore produces a model that is wrong only on long inputs, which is
    the hardest kind of wrong to notice.

    The layers are required to agree with each other, and the whole schedule
    is required to be close to the RoFormer one, so a checkpoint carrying a
    genuinely different rotation is still refused.
    """
    half = cfg["head_dim"] // 2
    reference = g.rotary_inv_freq(cfg["head_dim"])
    chosen = None
    for index in range(cfg["num_layers"]):
        key = f"{cfg['prefix']}encoder.layer.{index}.attention.self.rotary_embeddings.inv_freq"
        stored = weights.get(key)
        if stored is None:
            continue
        stored = np.asarray(stored, dtype=np.float64)
        if stored.shape != (half,):
            raise ValueError(
                f"esm layer {index} stores {stored.size} rotary frequencies, not {half}")
        if not np.allclose(stored, reference, rtol=1e-2, atol=1e-8):
            raise ValueError(
                f"esm layer {index} rotary frequencies are not the RoFormer schedule")
        if chosen is None:
            chosen = stored
        elif not np.array_equal(stored, chosen):
            raise ValueError("esm layers disagree about the rotary frequencies")
    if chosen is None:
        # Nothing shipped a buffer, so the reference would have kept the
        # schedule it computed at construction time.
        return reference
    return chosen


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


def _token_dropout(network, embedded, input_ids, keep_column, cfg, work_np):
    """ESM's mask-dropout compensation, computed from the ids at run time.

    Masked rows are zeroed, and then **every** row - masked or not - is scaled
    by ``(1 - 0.12) / (1 - observed mask fraction)``. The denominator counts
    the mask ids against the number of real tokens, so it changes with the
    input and cannot be a build-time constant.
    """
    seq_len = int(input_ids.shape[0])
    mask_id = g.add_constant(network, (seq_len,),
                             np.full((seq_len,), cfg["mask_token_id"]), dtype=np.int32)
    is_mask = network.add_elementwise(input_ids, mask_id,
                                      trt.ElementWiseOperation.EQUAL).get_output(0)
    is_mask = network.add_cast(is_mask, trt.float32).get_output(0)

    column = network.add_shuffle(is_mask)
    column.reshape_dims = (seq_len, 1)
    ones_row = g.add_constant(network, (1, 1), np.full((1,), 1.0), dtype=work_np)
    # Zero the rows the caller masked, exactly as the reference does before it
    # computes the compensation.
    embedded = g.add_product(network, embedded,
                             g.add_sub(network, ones_row, column.get_output(0)))

    masked_count = network.add_reduce(is_mask, trt.ReduceOperation.SUM, 1 << 0,
                                      True).get_output(0)
    real_count = network.add_reduce(keep_column, trt.ReduceOperation.SUM, 1 << 0,
                                    True).get_output(0)
    real_count = network.add_shuffle(real_count)
    real_count.reshape_dims = (1,)
    observed = g.add_div(network, masked_count, real_count.get_output(0))
    ones = g.add_constant(network, (1,), np.full((1,), 1.0), dtype=work_np)
    kept = g.add_sub(network, ones, observed)
    trained = g.add_constant(network, (1,), np.full((1,), 1.0 - _MASK_RATIO_TRAIN), dtype=work_np)
    factor = network.add_shuffle(g.add_div(network, trained, kept))
    factor.reshape_dims = (1, 1)
    return g.add_product(network, embedded, factor.get_output(0))


def build_engine(weights, cfg, *, seq_len, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    work_np = np.float32
    prefix = cfg["prefix"]
    head_dim = cfg["head_dim"]

    input_ids = network.add_input("input_ids", trt.int32, (seq_len,))
    attention_mask = network.add_input("attention_mask", trt.int32, (seq_len,))

    table = np.asarray(weights[f"{prefix}embeddings.word_embeddings.weight"], dtype=work_np)
    embedded = network.add_gather(
        g.add_constant(network, table.shape, table, dtype=work_np), input_ids, 0).get_output(0)

    keep = network.add_cast(attention_mask, trt.float32).get_output(0)
    column = network.add_shuffle(keep)
    column.reshape_dims = (seq_len, 1)
    keep_column = column.get_output(0)

    if cfg["token_dropout"]:
        embedded = _token_dropout(network, embedded, input_ids, keep_column, cfg, work_np)
    # The reference zeroes the padded rows last, after the compensation.
    hidden = g.add_product(network, embedded, keep_column)

    # mask[i] * mask[j]: a pair counts only if both ends are real. Masking the
    # keys alone would leave the padded query rows attending normally.
    row = network.add_shuffle(keep)
    row.reshape_dims = (1, seq_len)
    pair = g.add_product(network, keep_column, row.get_output(0))
    square = network.add_shuffle(pair)
    square.reshape_dims = (1, seq_len, seq_len)
    ones = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0), dtype=work_np)
    dropped = g.add_sub(network, ones, square.get_output(0))
    additive = g.add_product(
        network, dropped,
        g.add_constant(network, (1, 1, 1), np.full((1,), _MASK_FLOOR), dtype=work_np))

    cos_table, sin_table = g.rotary_tables(seq_len, head_dim,
                                           inv_freq=_rotary_inv_freq(weights, cfg),
                                           dtype=work_np)
    cos = g.add_constant(network, (1, seq_len, head_dim), cos_table, dtype=work_np)
    sin = g.add_constant(network, (1, seq_len, head_dim), sin_table, dtype=work_np)

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, mask=additive,
                                          cos=cos, sin=sin, dtype=work_np)

    head = g.add_gelu(network, g.add_linear(network, hidden, weights["lm_head.dense.weight"],
                                            weights["lm_head.dense.bias"], dtype=work_np),
                      dtype=work_np)
    head = g.add_layer_norm(network, head, weights["lm_head.layer_norm.weight"],
                            weights["lm_head.layer_norm.bias"], cfg["eps"])
    logits = g.add_linear(network, head, _decoder_weight(weights, prefix),
                          weights["lm_head.bias"], dtype=work_np)
    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("esm engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one ESM-2 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("esm supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("esm supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("esm does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("esm does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("esm does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("esm does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("esm does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("esm does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("esm does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("esm does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("esm does not support MTP speculative decoding")

    # fp32 only, for the reason the neighbouring encoders are fp32 only: no
    # fp16 number has been measured against the reference here, and an
    # accepted-but-unmeasured precision builds a bundle nothing checks.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"esm builds fp32 only; {precision} is not qualified for this family")

    if request.max_sequence_length is None:
        raise ValueError("esm needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 2:
        raise ValueError("esm needs room for at least <cls> and <eos>")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    prefix = _prefix(weights)
    _require_weights(weights, prefix)
    cfg = config_module.resolve(model_dir, weights, prefix)
    if seq_len > cfg["max_positions"]:
        raise NotImplementedError(
            f"esm was trained to {cfg['max_positions']} positions, not {seq_len}")

    vocabulary, framing = _vocabulary(model_dir, cfg)

    writer.set_header(family="esm", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, seq_len=seq_len, verbose=bool(request.verbose)))
    writer.add_bytes("vocab.txt", vocabulary)
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "vocab_size": cfg["vocab_size"],
            "pad_token_id": framing["pad_token_id"],
            "cls_token_id": framing["cls_token_id"],
            "eos_token_id": framing["eos_token_id"],
            "mask_token_id": framing["mask_token_id"],
            "unk_token_id": framing["unk_token_id"],
            "vocabulary_id": "esm",
        },
    )


def _vocabulary(model_dir: Path, cfg: dict) -> tuple[bytes, dict]:
    """The bundle's vocabulary, plus the ids the runtime frames and selects with.

    ESM publishes ``vocab.txt`` and no ``tokenizer.json``: its tokenizer is
    not a fast one, so there is nothing to convert. The file is one token per
    line and the line number is the id, which is the whole format - the
    runtime reads it directly rather than carrying a second tokenizer.
    """
    path = Path(model_dir) / "vocab.txt"
    if not path.is_file():
        raise FileNotFoundError(f"esm found no vocab.txt in {model_dir}")
    tokens = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(tokens) != cfg["vocab_size"]:
        raise ValueError(
            f"esm vocab.txt has {len(tokens)} tokens but the tensors say {cfg['vocab_size']}")
    index = {token: position for position, token in enumerate(tokens)}
    framing = {}
    for name, token in (("pad_token_id", "<pad>"), ("cls_token_id", "<cls>"),
                        ("eos_token_id", "<eos>"), ("mask_token_id", "<mask>"),
                        ("unk_token_id", "<unk>")):
        if token not in index:
            raise ValueError(f"esm vocab.txt does not define {token}")
        framing[name] = index[token]
    if framing["mask_token_id"] != cfg["mask_token_id"]:
        raise ValueError(
            f"esm config says mask_token_id={cfg['mask_token_id']} but vocab.txt puts <mask> at "
            f"{framing['mask_token_id']}")
    if framing["pad_token_id"] != cfg["pad_token_id"]:
        raise ValueError(
            f"esm config says pad_token_id={cfg['pad_token_id']} but vocab.txt puts <pad> at "
            f"{framing['pad_token_id']}")
    return ("\n".join(tokens) + "\n").encode("utf-8"), framing
