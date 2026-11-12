# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve WavLM geometry and the CTC vocabulary while keeping each checkpoint's layout."""

from __future__ import annotations

import json
from pathlib import Path

# The samples the engine's optimisation profile is built for. A padded clip does
# not give the unpadded answer (the group-normalised feature encoder normalises
# over time and the attention has no padding mask), so the engine accepts the
# true length.
MIN_SAMPLES = 4000
OPT_SAMPLES = 80000
MAX_SAMPLES = 480000

# The only tokenizer whose decoding the runtime reproduces: collapse repeats,
# drop the pad token, turn the word delimiter into a space, strip the ends. The
# phoneme tokenizer joins its tokens with spaces instead, so it is refused by name.
_TOKENIZER = "Wav2Vec2CTCTokenizer"
# Transformers' clean_up_tokenization only rewrites spaces next to these
# characters, so for a vocabulary without them it changes nothing.
_CLEANUP_CHARACTERS = frozenset(".?!,'")


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required wavlm {what} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"wavlm {message}")


def _token_text(value, default: str) -> str:
    """A tokenizer_config token is a string or a serialised AddedToken."""
    if value is None:
        return default
    if isinstance(value, dict):
        value = value.get("content")
    _require(isinstance(value, str) and value != "",
             "needs string special tokens in tokenizer_config.json")
    return value


def resolve_encoder(cfg: dict) -> dict:
    """Return the encoder geometry of one WavLM config, or raise naming what is wrong.

    The pretrained encoder (``WavLMModel``) and its CTC fine-tunes share this
    layout, so the same rules apply to both.
    """
    _require(cfg.get("model_type") == "wavlm",
             f"builds model_type 'wavlm', not {cfg.get('model_type')!r}")
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
    _require(len(kernels) == len(strides) == len(dims) > 0,
             "needs one kernel, stride and width per feature-encoder layer")

    pos_kernel = int(cfg["num_conv_pos_embeddings"])
    # An even kernel convolved with half its width of padding leaves one extra
    # column, and the reference drops the last one. An odd kernel would not.
    _require(pos_kernel % 2 == 0,
             f"needs an even num_conv_pos_embeddings; {pos_kernel} would not need the same-pad trim")
    pos_groups = int(cfg["num_conv_pos_embedding_groups"])
    _require(hidden % pos_groups == 0,
             f"needs hidden_size {hidden} divisible by {pos_groups} positional-conv groups")

    # The reference gives each sign of the key-minus-query offset half of the
    # buckets, keeps the first half of those exact and spaces the rest
    # logarithmically up to max_bucket_distance. Without an exact range below
    # that distance the logarithmic spacing is undefined.
    buckets = int(cfg["num_buckets"])
    max_distance = int(cfg["max_bucket_distance"])
    max_exact = buckets // 2 // 2
    _require(max_exact >= 1 and max_distance > max_exact,
             f"needs num_buckets >= 4 and max_bucket_distance above {max_exact}, "
             f"not {buckets} and {max_distance}")

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
        "num_buckets": buckets,
        "max_distance": max_distance,
    }


def _tokens(vocab: dict, added: dict, vocab_size: int) -> tuple[list[str], dict[str, int]]:
    """Name every CTC id from vocab.json plus the tokenizer's added tokens."""
    ids: dict[str, int] = {}
    tokens = [""] * vocab_size
    for token, index in [*vocab.items(), *added.items()]:
        index = int(index)
        _require(0 <= index < vocab_size,
                 f"tokenizer maps {token!r} to id {index}, outside the {vocab_size} CTC classes")
        _require(ids.get(token, index) == index, f"tokenizer maps {token!r} to two ids")
        _require(tokens[index] in ("", token), f"tokenizer maps two tokens to id {index}")
        tokens[index] = token
        ids[token] = index
    _require(all(tokens), "tokenizer leaves a CTC id unnamed")
    return tokens, ids


def _tokenizer_settings(tokenizer: dict, special_map: dict) -> dict:
    """The tokenizer settings as Transformers' loader resolves them.

    A tokenizer_config.json without ``added_tokens_decoder`` is the legacy
    layout: every entry of special_tokens_map.json overrides the configuration.
    With ``added_tokens_decoder`` the map is not read at all.
    """
    if "added_tokens_decoder" in tokenizer:
        return dict(tokenizer)
    return {**tokenizer, **special_map}


def _added_tokens(tokenizer: dict, added_file: dict) -> dict:
    """Added tokens from ``added_tokens_decoder`` when the configuration has it,
    otherwise from added_tokens.json, as Transformers' loader takes them."""
    decoder = tokenizer.get("added_tokens_decoder")
    if decoder is None:
        return added_file
    added = {}
    for index, token in decoder.items():
        content = token.get("content") if isinstance(token, dict) else token
        _require(isinstance(content, str) and content != "",
                 "needs string added tokens in tokenizer_config.json")
        added[content] = int(index)
    return added


def require_head(cfg: dict, shapes: dict) -> None:
    """Refuse a CTC head whose shape is not the vocabulary over the encoder width."""
    head = shapes.get("lm_head.weight")
    if head is not None:
        expected = (cfg["vocab_size"], cfg["hidden_size"])
        _require(tuple(head) == expected, f"lm_head {tuple(head)} does not match {expected}")


def resolve(model_dir: Path, shapes: dict) -> dict:
    """Return everything the CTC builder needs, or raise naming what is wrong.

    Only the JSON files are read, so the builder calls this before loading any
    weight; ``shapes`` may be empty then and the head is checked once loaded.
    """
    model_dir = Path(model_dir)
    cfg = _read_json(model_dir / "config.json", "config.json")
    architectures = tuple(cfg.get("architectures") or ())
    _require("WavLMForCTC" in architectures,
             f"builds WavLMForCTC, not {architectures or '(unstated)'}: "
             "a checkpoint without a CTC head has no transcription to build")
    geometry = resolve_encoder(cfg)

    pre = _read_json(model_dir / "preprocessor_config.json", "preprocessor_config.json")
    vocab = _read_json(model_dir / "vocab.json", "vocab.json")
    tokenizer = _read_optional_json(model_dir / "tokenizer_config.json")
    added = _added_tokens(tokenizer, _read_optional_json(model_dir / "added_tokens.json"))
    settings = _tokenizer_settings(tokenizer,
                                   _read_optional_json(model_dir / "special_tokens_map.json"))

    _require(int(pre.get("feature_size", 1)) == 1, "reads single-channel audio")
    _require(float(pre.get("padding_value", 0.0)) == 0.0, "needs a zero padding value")
    _require(pre.get("padding_side", "right") == "right", "needs right padding")
    sample_rate = int(pre.get("sampling_rate", 16000))

    tokenizer_class = tokenizer.get("tokenizer_class") or cfg.get("tokenizer_class") or _TOKENIZER
    _require(tokenizer_class == _TOKENIZER,
             f"decodes {_TOKENIZER} vocabularies, not {tokenizer_class}")
    _require(settings.get("replace_word_delimiter_char", " ") == " ",
             "decodes the word delimiter to a space only")

    vocab_size = int(cfg["vocab_size"])
    tokens, ids = _tokens(vocab, added, vocab_size)
    # The runtime neither lowercases nor applies Transformers' space clean-up;
    # a tokenizer that would is refused unless the clean-up cannot apply.
    _require(not settings.get("do_lower_case", False), "decodes without do_lower_case")
    if settings.get("clean_up_tokenization_spaces", False):
        affected = sorted({char for token in tokens for char in token
                           if char in _CLEANUP_CHARACTERS})
        _require(not affected,
                 f"does not reproduce clean_up_tokenization_spaces around {affected}")

    # The tokenizer's pad token is the CTC blank its decoder drops; the head was
    # trained with config.pad_token_id as that blank, so the two must agree.
    pad = _token_text(settings.get("pad_token"), "<pad>")
    blank = ids.get(pad)
    _require(blank is not None, f"needs its pad token {pad!r} in the vocabulary as the CTC blank")
    pad_token_id = cfg.get("pad_token_id")
    _require(pad_token_id is None or int(pad_token_id) == blank,
             f"pad token {pad!r} is id {blank} but config.pad_token_id is {pad_token_id}")
    delimiter = _token_text(settings.get("word_delimiter_token"), "|")
    _require(delimiter in ids, f"needs its word delimiter {delimiter!r} in the vocabulary")

    resolved = {
        **geometry,
        "sample_rate": sample_rate,
        "do_normalize": bool(pre.get("do_normalize", True)),
        "tokens": tokens,
        "blank_token_id": int(blank),
        "word_delimiter_token": delimiter,
        "vocab_size": vocab_size,
    }
    require_head(resolved, shapes)
    return resolved
