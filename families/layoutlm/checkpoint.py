# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check a LayoutLMForQuestionAnswering checkpoint.

The runtime reproduces transformers' document-QA pipeline for LayoutLM, which
only works for one tokenizer shape: byte-level BPE over pre-split words with a
prefix space, framed as ``<s> A </s></s> B </s>`` (RoBERTa). Checkpoints that
pair LayoutLM with another tokenizer (WordPiece, SentencePiece) are refused
here rather than tokenized differently at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np

PREFIX = "layoutlm."
_EMBEDDINGS = (
    "word_embeddings.weight",
    "position_embeddings.weight",
    "x_position_embeddings.weight",
    "y_position_embeddings.weight",
    "h_position_embeddings.weight",
    "w_position_embeddings.weight",
    "token_type_embeddings.weight",
    "LayerNorm.weight",
    "LayerNorm.bias",
)
_LAYER = (
    "attention.self.query.weight",
    "attention.self.query.bias",
    "attention.self.key.weight",
    "attention.self.key.bias",
    "attention.self.value.weight",
    "attention.self.value.bias",
    "attention.output.dense.weight",
    "attention.output.dense.bias",
    "attention.output.LayerNorm.weight",
    "attention.output.LayerNorm.bias",
    "intermediate.dense.weight",
    "intermediate.dense.bias",
    "output.dense.weight",
    "output.dense.bias",
    "output.LayerNorm.weight",
    "output.LayerNorm.bias",
)
# Stored but not on the question-answering path: the pooler feeds only the
# classification heads, and transformers treats position_ids as a
# non-persistent buffer (it uses arange regardless of the stored values).
_UNUSED = ("layoutlm.pooler.dense.weight", "layoutlm.pooler.dense.bias",
           "layoutlm.embeddings.position_ids")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"layoutlm checkpoint lacks {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"layoutlm {path.name} is not a JSON object")
    return value


def read_config(model_dir: Path) -> dict:
    """The encoder dimensions, checked against what the graph implements."""
    raw = _read_json(Path(model_dir) / "config.json")
    if raw.get("model_type") != "layoutlm":
        raise ValueError(f"layoutlm cannot build model_type {raw.get('model_type')!r}")
    if "LayoutLMForQuestionAnswering" not in (raw.get("architectures") or []):
        raise ValueError("layoutlm builds only the LayoutLMForQuestionAnswering head")
    if raw.get("hidden_act", "gelu") != "gelu":
        raise ValueError("layoutlm implements the exact (erf) GELU only")
    if raw.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("layoutlm implements absolute position embeddings only")
    config = {
        "hidden_size": int(raw["hidden_size"]),
        "num_layers": int(raw["num_hidden_layers"]),
        "num_heads": int(raw["num_attention_heads"]),
        "intermediate_size": int(raw["intermediate_size"]),
        "vocab_size": int(raw["vocab_size"]),
        "max_position_embeddings": int(raw["max_position_embeddings"]),
        "max_2d_position_embeddings": int(raw.get("max_2d_position_embeddings", 1024)),
        "type_vocab_size": int(raw.get("type_vocab_size", 2)),
        "layer_norm_eps": float(raw.get("layer_norm_eps", 1e-12)),
    }
    if config["hidden_size"] % config["num_heads"]:
        raise ValueError("layoutlm hidden_size is not a multiple of num_attention_heads")
    if config["max_2d_position_embeddings"] <= 1000:
        raise ValueError("layoutlm boxes use a 0..1000 grid; the 2-D tables are too small")
    if config["type_vocab_size"] < 1:
        raise ValueError("layoutlm needs at least one token type embedding")
    return config


def read_tokenizer(model_dir: Path) -> dict:
    """The effective tokenizer of the reference, checked for the supported shape.

    transformers applies ``add_prefix_space`` from tokenizer_config.json over the
    value stored in tokenizer.json (``add_prefix_space`` appears twice in this
    checkpoint's config; the last one, ``true``, wins as in any JSON reader),
    and the document-QA pipeline refuses pre-split words without it.
    """
    directory = Path(model_dir)
    document = _read_json(directory / "tokenizer.json")
    settings = _read_json(directory / "tokenizer_config.json")
    model = document.get("model") or {}
    pre_tokenizer = document.get("pre_tokenizer") or {}
    processor = document.get("post_processor") or {}
    if model.get("type") != "BPE" or pre_tokenizer.get("type") != "ByteLevel":
        raise ValueError("layoutlm supports byte-level BPE tokenizers only")
    if not pre_tokenizer.get("use_regex", True) or document.get("normalizer") is not None:
        raise ValueError("layoutlm supports the GPT-2 regex pre-tokenizer without a normalizer")
    if model.get("dropout") is not None or model.get("byte_fallback") or model.get("ignore_merges"):
        raise ValueError("layoutlm does not support BPE dropout, byte fallback or ignore_merges")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise ValueError("layoutlm does not support BPE subword affixes")
    if processor.get("type") != "RobertaProcessing":
        raise ValueError("layoutlm supports the RoBERTa pair framing <s> A </s></s> B </s> only")
    for token in document.get("added_tokens") or []:
        if token.get("normalized") or token.get("single_word"):
            raise ValueError(f"layoutlm cannot match the added token {token.get('content')!r}")
    prefix_space = settings.get("add_prefix_space", pre_tokenizer.get("add_prefix_space", False))
    if prefix_space is not True:
        raise ValueError("layoutlm needs add_prefix_space=true to encode pre-split words")
    if settings.get("padding_side", "right") != "right":
        raise ValueError("the document-QA reference supports right padding only")
    # The windows are cut left to right, as the reference does with right truncation.
    if settings.get("truncation_side", "right") != "right":
        raise ValueError("layoutlm windows the document left to right; truncation_side must be right")
    # The reference keeps tokenizer.cls_token_id eligible as an answer start and
    # boxes tokenizer.sep_token_id with [1000] * 4; the runtime uses the ids of
    # the pair framing, so both must name the framing's tokens.
    special_map = directory / "special_tokens_map.json"
    special = _read_json(special_map) if special_map.is_file() else {}
    for name, framing, default in (("cls_token", processor["cls"][0], "<s>"),
                                   ("sep_token", processor["sep"][0], "</s>")):
        value = settings.get(name, special.get(name, default))
        if isinstance(value, dict):
            value = value.get("content")
        if value != framing:
            raise ValueError(f"layoutlm needs the tokenizer's {name} to be the framing's {framing}, "
                             f"not {value!r}")

    pre_tokenizer["add_prefix_space"] = True
    processor["add_prefix_space"] = True
    vocabulary_size = 1 + max(
        [int(value) for value in (model.get("vocab") or {}).values()]
        + [int(token["id"]) for token in document.get("added_tokens") or []]
    )
    return {
        "json": json.dumps(document, ensure_ascii=False).encode("utf-8"),
        "cls_token_id": int(processor["cls"][1]),
        "sep_token_id": int(processor["sep"][1]),
        "vocab_size": vocabulary_size,
        "model_max_length": settings.get("model_max_length"),
    }


def _open_safetensors(model_dir: Path) -> dict:
    """All tensors of model.safetensors as float32 NumPy arrays (BF16 decoded)."""
    from safetensors import safe_open

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"layoutlm needs model.safetensors in {model_dir}")
    weights = {}
    with safe_open(str(path), framework="numpy") as handle:
        for name in handle.keys():
            value = handle.get_tensor(name)
            if value.dtype == ml_dtypes.bfloat16 or value.dtype == np.float16:
                value = value.astype(np.float32)
            weights[name] = value
    return weights


def read_weights(model_dir: Path, config: dict) -> dict:
    """The question-answering tensors under their checkpoint names, shape-checked."""
    weights = _open_safetensors(model_dir)
    hidden = config["hidden_size"]
    inner = config["intermediate_size"]
    expected = {f"{PREFIX}embeddings.{name}" for name in _EMBEDDINGS}
    for layer in range(config["num_layers"]):
        expected.update(f"{PREFIX}encoder.layer.{layer}.{name}" for name in _LAYER)
    expected.update({"qa_outputs.weight", "qa_outputs.bias"})
    missing = sorted(expected - weights.keys())
    unknown = sorted(weights.keys() - expected - set(_UNUSED))
    if missing or unknown:
        raise ValueError(f"layoutlm checkpoint tensors do not match LayoutLMForQuestionAnswering: "
                         f"missing {missing[:5]}, unexpected {unknown[:5]}")
    shapes = {
        f"{PREFIX}embeddings.word_embeddings.weight": (config["vocab_size"], hidden),
        f"{PREFIX}embeddings.position_embeddings.weight": (config["max_position_embeddings"], hidden),
        f"{PREFIX}embeddings.token_type_embeddings.weight": (config["type_vocab_size"], hidden),
        "qa_outputs.weight": (2, hidden),
        "qa_outputs.bias": (2,),
    }
    for table in ("x", "y", "h", "w"):
        shapes[f"{PREFIX}embeddings.{table}_position_embeddings.weight"] = (
            config["max_2d_position_embeddings"], hidden)
    for layer in range(config["num_layers"]):
        base = f"{PREFIX}encoder.layer.{layer}."
        for name in ("attention.self.query", "attention.self.key", "attention.self.value",
                     "attention.output.dense"):
            shapes[base + name + ".weight"] = (hidden, hidden)
        shapes[base + "intermediate.dense.weight"] = (inner, hidden)
        shapes[base + "output.dense.weight"] = (hidden, inner)
    for name, shape in shapes.items():
        if tuple(weights[name].shape) != shape:
            raise ValueError(f"layoutlm tensor {name} has shape {tuple(weights[name].shape)}, "
                             f"expected {shape}")
    return {name: np.asarray(weights[name], dtype=np.float32) for name in expected}
