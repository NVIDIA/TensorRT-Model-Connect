# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a MobileBERT pretraining bundle.

One fixed-length engine carries both heads of ``MobileBertForPreTraining``:
``prediction_logits`` (the masked-LM scores of every row of the window) and
``seq_relationship_logits`` (the next-sentence scores of the first row). The
runtime serves ``masked_text_to_token_scores`` from the first and
``text_pair_to_pretraining_relation_scores`` from the second, whichever of the
two is the bundle's primary task.

This family requires torch in the build environment: the published
``google/mobilebert-uncased`` repository ships ``pytorch_model.bin`` and no
safetensors file.

Precision is float32 only. The encoder has no normalisation statistics
anywhere (``no_norm``), and the published checkpoint's last layer leaves the
stream at magnitudes near 6e7, which float16 cannot represent. The
next-sentence logits that read that row directly are near 2.7e8 and differ by
a few 1e4, a relative gap of about 1e-4 that bfloat16 cannot resolve either.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, graph
from .reorderable_marks import RANGES as _REORDERABLE

_SUPPORTED_TASKS = ("masked_text_to_token_scores", "text_pair_to_pretraining_relation_scores")

# The pretraining next-sentence head: index 0 means the second text continues
# the first, index 1 that it is a random text.
_RELATION_LABELS = ("next", "not_next")

_SPECIAL_TOKENS = {
    "pad_token_id": "[PAD]",
    "unk_token_id": "[UNK]",
    "cls_token_id": "[CLS]",
    "sep_token_id": "[SEP]",
    "mask_token_id": "[MASK]",
}


def _open_torch_checkpoint(model_dir: Path) -> dict:
    """Open the family's required pytorch_model.bin checkpoint as float32 arrays."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This family requires torch in the build environment") from error

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    weights = {}
    for key, value in state.items():
        if not value.is_floating_point():
            # The only integer tensor is the position_ids buffer; the reference
            # does not load it (it is not persistent) and counts from zero.
            if key.endswith("embeddings.position_ids"):
                continue
            raise ValueError(f"mobilebert checkpoint tensor {key} is not floating point")
        weights[key] = value.detach().to(torch.float32).numpy()
    return weights


def _layer_keys(cfg: dict, index: int) -> list[str]:
    layer = f"mobilebert.encoder.layer.{index}."
    modules = [
        "bottleneck.input", "bottleneck.attention", "attention.output", "output",
        "output.bottleneck",
        *(f"ffn.{ffn}.output" for ffn in range(cfg["extra_ffns"])),
    ]
    keys = [f"{layer}{module}.{part}" for module in modules
            for part in ("dense.weight", "dense.bias", "LayerNorm.weight", "LayerNorm.bias")]
    keys += [f"{layer}attention.self.{name}.{part}" for name in ("query", "key", "value")
             for part in ("weight", "bias")]
    keys += [f"{layer}intermediate.dense.weight", f"{layer}intermediate.dense.bias"]
    keys += [f"{layer}ffn.{ffn}.intermediate.dense.{part}" for ffn in range(cfg["extra_ffns"])
             for part in ("weight", "bias")]
    return keys


def _require_weights(weights: dict, cfg: dict) -> None:
    """Every tensor the graph reads, and the ties the reference applies."""
    required = [
        "mobilebert.embeddings.embedding_transformation.bias",
        "mobilebert.embeddings.LayerNorm.weight",
        "mobilebert.embeddings.LayerNorm.bias",
        "cls.predictions.transform.dense.weight",
        "cls.predictions.transform.dense.bias",
        "cls.predictions.transform.LayerNorm.weight",
        "cls.predictions.transform.LayerNorm.bias",
        "cls.predictions.bias",
        "cls.seq_relationship.weight",
        "cls.seq_relationship.bias",
    ]
    for index in range(cfg["num_layers"]):
        required += _layer_keys(cfg, index)
    if not cfg["tie_word_embeddings"]:
        required.append("cls.predictions.decoder.weight")
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("mobilebert checkpoint is missing expected tensors: "
                       + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
    relation = np.asarray(weights["cls.seq_relationship.weight"]).shape
    if relation != (len(_RELATION_LABELS), cfg["hidden_size"]):
        raise ValueError(f"mobilebert next-sentence head is {relation}, not "
                         f"{(len(_RELATION_LABELS), cfg['hidden_size'])}")
    # The reference ties the decoder to the word table and its bias to
    # cls.predictions.bias. A file that stores a different copy of either is
    # ambiguous about which one it means, so it is refused rather than guessed.
    table = weights["mobilebert.embeddings.word_embeddings.weight"]
    stored = weights.get("cls.predictions.decoder.weight")
    if cfg["tie_word_embeddings"] and stored is not None and not np.array_equal(stored, table):
        raise ValueError("mobilebert decoder weight differs from the tied word embedding table")
    stored_bias = weights.get("cls.predictions.decoder.bias")
    if stored_bias is not None and not np.array_equal(stored_bias, weights["cls.predictions.bias"]):
        raise ValueError("mobilebert decoder bias differs from the tied cls.predictions.bias")


def _template_pieces(template: list) -> list[tuple[str, int]]:
    pieces = []
    for item in template:
        (kind, body), = item.items()
        pieces.append((("special:" if kind == "SpecialToken" else "") + str(body["id"]),
                       int(body["type_id"])))
    return pieces


def _check_framing(post_processor) -> None:
    """The runtime frames with BERT's template; refuse a tokenizer that frames otherwise."""
    single = [("special:[CLS]", 0), ("A", 0), ("special:[SEP]", 0)]
    pair = single + [("B", 1), ("special:[SEP]", 1)]
    if not isinstance(post_processor, dict):
        raise ValueError("mobilebert tokenizer.json has no post_processor to frame with")
    kind = post_processor.get("type")
    if kind == "TemplateProcessing":
        if (_template_pieces(post_processor.get("single", [])) == single
                and _template_pieces(post_processor.get("pair", [])) == pair):
            return
    elif kind == "BertProcessing":
        if (post_processor.get("cls", [None])[0] == "[CLS]"
                and post_processor.get("sep", [None])[0] == "[SEP]"):
            return
    raise NotImplementedError(
        "mobilebert frames inputs as [CLS] A [SEP] and [CLS] A [SEP] B [SEP]; this "
        "tokenizer.json frames them differently")


def _reorderable(char: str) -> bool:
    index = bisect.bisect_left(_REORDERABLE, (ord(char) + 1,)) - 1
    return index >= 0 and _REORDERABLE[index][0] <= ord(char) <= _REORDERABLE[index][1]


def _check_vocabulary(vocab: dict) -> None:
    """Refuse what the runtime would refuse when it loads the bundle.

    The runtime's Unicode tables do not reproduce the reference's canonical
    reordering of combining marks. That never changes ids while no vocabulary
    token holds such a mark (see tests/unicode_tables.py), so a vocabulary that
    holds one is refused here, before a bundle is written.
    """
    for token in vocab:
        if any(_reorderable(char) for char in token):
            raise NotImplementedError(
                f"mobilebert tokenizer vocabulary token {token!r} holds a combining mark "
                "this tokenizer cannot order")


def _added_tokens(entries: list, vocab: dict) -> dict:
    """Added tokens as content -> id; the runtime matches them in the raw text only."""
    added = {}
    for entry in entries:
        content, token_id = entry.get("content"), int(entry.get("id", -1))
        if not content or not 0 <= token_id < len(vocab) or vocab.get(content, token_id) != token_id:
            raise ValueError(f"mobilebert tokenizer added token {content!r} has an invalid id")
        options = [key for key in ("normalized", "single_word", "lstrip", "rstrip") if entry.get(key)]
        if options:
            raise NotImplementedError(
                f"mobilebert tokenizer added token {content!r} uses {options}, which the "
                "runtime does not implement")
        added[content] = token_id
    return added


def _tokenizer(model_dir: Path, cfg: dict) -> tuple[bytes, dict]:
    """The bundle's tokenizer.json, plus the ids the runtime frames and selects with."""
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"mobilebert found no tokenizer.json in {model_dir}")
    data = path.read_bytes()
    tokenizer = json.loads(data.decode("utf-8"))
    model = tokenizer.get("model") or {}
    # Older files, this checkpoint's among them, carry no model "type"; the
    # reference then recognises WordPiece by its fields.
    word_piece = (model.get("type") == "WordPiece" if "type" in model else
                  {"continuing_subword_prefix", "max_input_chars_per_word"} <= model.keys()
                  and "merges" not in model)
    if not word_piece:
        raise NotImplementedError(
            f"mobilebert reads a WordPiece tokenizer, not {model.get('type', 'an untyped model')!r}")
    if tokenizer.get("pre_tokenizer", {}).get("type") != "BertPreTokenizer":
        raise NotImplementedError("mobilebert reads a tokenizer with the BertPreTokenizer only")
    normalizer = tokenizer.get("normalizer") or {}
    # The runtime's Unicode tables reproduce the uncased BertNormalizer only;
    # strip_accents null follows lowercase.
    uncased = (normalizer.get("type") == "BertNormalizer"
               and all(normalizer.get(key, True) is True
                       for key in ("clean_text", "handle_chinese_chars", "lowercase"))
               and normalizer.get("strip_accents") in (None, True))
    if not uncased:
        raise NotImplementedError(
            "mobilebert reads a tokenizer with the uncased BertNormalizer only (clean text, "
            "separate CJK, strip accents, lowercase)")
    vocab = model.get("vocab") or {}
    if len(vocab) != cfg["vocab_size"]:
        raise ValueError(
            f"mobilebert tokenizer has {len(vocab)} tokens but the tensors say {cfg['vocab_size']}")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("mobilebert tokenizer vocabulary ids are not 0..n-1")
    _check_vocabulary(vocab)
    added = _added_tokens(tokenizer.get("added_tokens", []), vocab)
    framing = {}
    for name, token in _SPECIAL_TOKENS.items():
        token_id = added.get(token, vocab.get(token))
        if token_id is None or not 0 <= int(token_id) < cfg["vocab_size"]:
            raise ValueError(f"mobilebert tokenizer does not define {token}")
        framing[name] = int(token_id)
    if framing["pad_token_id"] != cfg["pad_token_id"]:
        raise ValueError(f"mobilebert config says pad_token_id={cfg['pad_token_id']} but the "
                         f"tokenizer puts [PAD] at {framing['pad_token_id']}")
    _check_framing(tokenizer.get("post_processor"))
    return data, framing


def build_engine(weights: dict, cfg: dict, *, seq_len: int, verbose: bool = False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 would round every float32 product to a 10-bit mantissa; the
    # next-sentence logits are near 2.7e8 and their gap is near 1e4.
    config.clear_flag(trt.BuilderFlag.TF32)
    constants = graph.build_network(network, weights, cfg, seq_len)
    plan = builder.build_serialized_network(network, config)
    del constants
    if plan is None:
        raise RuntimeError("mobilebert engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one MobileBERT pretraining bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("mobilebert supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend != "trt":
        raise ValueError("mobilebert supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("mobilebert does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("mobilebert runs one sequence per call; max_batch_size must be 1")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("mobilebert does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("mobilebert does not support context parallelism")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("mobilebert does not take an image")
    if request.video_num_frames is not None:
        raise NotImplementedError("mobilebert does not take a video")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError(
            f"mobilebert builds the float32 checkpoint as stored; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("mobilebert builds float32 throughout; fp32_layers does not apply")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("mobilebert does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"mobilebert builds fp32 only, not {precision}: its last layer leaves the stream "
            "near 6e7, beyond float16, and the next-sentence logits differ by a relative 1e-4")

    if request.max_sequence_length is None:
        raise ValueError("mobilebert needs max_sequence_length: the engine is fixed-length")
    seq_len = int(request.max_sequence_length)
    if seq_len < 3:
        raise ValueError("mobilebert needs room for at least [CLS], one token and [SEP]")

    model_dir = Path(request.model_dir)
    weights = _open_torch_checkpoint(model_dir)
    cfg = config_module.resolve(model_dir, weights)
    _require_weights(weights, cfg)
    if seq_len > cfg["max_positions"]:
        raise NotImplementedError(
            f"mobilebert has {cfg['max_positions']} learned positions, not {seq_len}")
    tokenizer_json, framing = _tokenizer(model_dir, cfg)

    writer.set_header(family="mobilebert", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(weights, cfg, seq_len=seq_len,
                                                verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", tokenizer_json)
    writer.add_json(
        "runtime.json",
        {
            "max_sequence_length": seq_len,
            "vocab_size": cfg["vocab_size"],
            "type_vocab_size": cfg["type_vocab_size"],
            **framing,
            "vocabulary_id": "mobilebert",
            "relation_labels": list(_RELATION_LABELS),
            "relation_vocabulary_id": "mobilebert.next_sentence",
        },
    )
