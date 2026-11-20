# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve wav2vec2 geometry while keeping each checkpoint's normalization layout."""

from __future__ import annotations

import json
from pathlib import Path

# The samples the engine's optimisation profile is built for. The feature
# encoder's normalisation is taken over the time axis, so a padded clip does not
# give the unpadded answer and the engine has to accept the true length.
MIN_SAMPLES = 4000
OPT_SAMPLES = 80000
MAX_SAMPLES = 480000


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required wav2vec2 {what} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"wav2vec2 {message}")


def resolve(model_dir: Path, shapes: dict) -> dict:
    """Return the geometry the builder needs, or raise naming what is wrong."""
    model_dir = Path(model_dir)
    cfg = _read_json(model_dir / "config.json", "config.json")
    pre = _read_json(model_dir / "preprocessor_config.json", "preprocessor_config.json")
    vocab = _read_json(model_dir / "vocab.json", "vocab.json")

    architectures = tuple(cfg.get("architectures") or ())
    _require("Wav2Vec2ForCTC" in architectures,
             f"builds Wav2Vec2ForCTC, not {architectures or '(unstated)'}")

    _require(isinstance(cfg.get("do_stable_layer_norm"), bool),
             "needs a boolean do_stable_layer_norm")
    _require(cfg.get("feat_extract_norm") in {"group", "layer"},
             "needs feat_extract_norm='group' or 'layer'")
    _require(isinstance(cfg.get("conv_bias"), bool), "needs a boolean conv_bias")
    _require(not cfg.get("add_adapter", False) and cfg.get("adapter_attn_dim") is None,
             "does not build adapters")
    _require(cfg.get("hidden_act") == "gelu",
             f"builds the exact GELU, not {cfg.get('hidden_act')!r}")
    _require(cfg.get("feat_extract_activation", "gelu") == "gelu",
             "builds the exact GELU feature activation")

    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    _require(hidden % heads == 0, f"needs hidden_size {hidden} divisible by {heads} heads")

    kernels = [int(v) for v in cfg["conv_kernel"]]
    strides = [int(v) for v in cfg["conv_stride"]]
    dims = [int(v) for v in cfg["conv_dim"]]
    _require(len(kernels) == len(strides) == len(dims),
             "needs one kernel, stride and width per feature-encoder layer")

    pos_kernel = int(cfg["num_conv_pos_embeddings"])
    # An even kernel convolved with half its width of padding leaves one extra
    # column, and the reference drops the last one. An odd kernel would not.
    _require(pos_kernel % 2 == 0,
             f"needs an even num_conv_pos_embeddings; {pos_kernel} would not need the same-pad trim")
    pos_groups = int(cfg["num_conv_pos_embedding_groups"])
    _require(hidden % pos_groups == 0,
             f"needs hidden_size {hidden} divisible by {pos_groups} positional-conv groups")

    _require(int(pre.get("feature_size", 1)) == 1, "reads single-channel audio")
    _require(float(pre.get("padding_value", 0.0)) == 0.0, "needs a zero padding value")
    _require(pre.get("padding_side", "right") == "right", "needs right padding")
    sample_rate = int(pre.get("sampling_rate", 16000))

    vocab_size = int(cfg["vocab_size"])
    _require(len(vocab) == vocab_size,
             f"vocab.json holds {len(vocab)} tokens but the config states {vocab_size}")
    head = shapes.get("lm_head.weight")
    if head is not None:
        _require(tuple(head) == (vocab_size, hidden),
                 f"lm_head {tuple(head)} does not match ({vocab_size}, {hidden})")

    tokens = [""] * vocab_size
    for token, index in vocab.items():
        index = int(index)
        _require(0 <= index < vocab_size, f"vocab.json maps {token!r} outside the vocabulary")
        _require(tokens[index] == "", f"vocab.json maps two tokens to id {index}")
        tokens[index] = token
    _require(all(tokens), "vocab.json leaves a token id unnamed")

    blank = vocab.get("<pad>")
    _require(blank is not None, "needs a <pad> token to use as the CTC blank")
    delimiter = vocab.get("|")
    _require(delimiter is not None, "needs a '|' token to use as the word delimiter")

    return {
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "eps": float(cfg["layer_norm_eps"]),
        "feat_extract_norm": cfg["feat_extract_norm"],
        "do_stable_layer_norm": cfg["do_stable_layer_norm"],
        "conv_bias": cfg["conv_bias"],
        "conv_kernel": kernels,
        "conv_stride": strides,
        "conv_dim": dims,
        "pos_kernel": pos_kernel,
        "pos_groups": pos_groups,
        "sample_rate": sample_rate,
        "do_normalize": bool(pre.get("do_normalize", True)),
        "tokens": tokens,
        "blank_token_id": int(blank),
        "word_delimiter_token": "|",
        "vocab_size": vocab_size,
    }
