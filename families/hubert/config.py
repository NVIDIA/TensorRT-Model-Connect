# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one HubertForCTC checkpoint into the geometry this family builds.

Two encoder layouts share the ``hubert`` model type: the base layout (group
norm on the first feature-encoder convolution, post-norm encoder blocks) and
the large layout (layer norm on every convolution, pre-norm "stable" blocks).
The config selects the layout, and each loads cleanly under the other's
weights, so the layout is read from the config and the checkpoint's tensor
inventory is then checked name by name and shape by shape against it.
"""

from __future__ import annotations

import json
from pathlib import Path

# The samples the engine's optimisation profile is built for. The base layout's
# first normalisation runs over the time axis, so a padded clip does not give
# the unpadded answer and the engine takes the true length instead.
MIN_SAMPLES = 4000
OPT_SAMPLES = 80000
MAX_SAMPLES = 480000

PREFIX = "hubert."

# Transformers' HubertConfig defaults. A config.json that omits a key gets
# these from the reference, so the family reads them the same way; the official
# large and xlarge CTC configs omit feat_proj_layer_norm and conv_pos_batch_norm.
_DEFAULTS = {
    "feat_extract_norm": "group",
    "feat_extract_activation": "gelu",
    "do_stable_layer_norm": False,
    "conv_bias": False,
    "feat_proj_layer_norm": True,
    "conv_pos_batch_norm": False,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "pad_token_id": 0,
}

# Present in checkpoints whose config enables SpecAugment; used only to mask
# features during training, never at inference.
TRAINING_ONLY = (PREFIX + "masked_spec_embed",)


def _read_json(path: Path, what: str, *, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required hubert {what} is missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise NotImplementedError(f"hubert {what} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"hubert {message}")


def _setting(cfg: dict, key: str):
    return cfg[key] if key in cfg else _DEFAULTS[key]


def _token_text(value, default: str) -> str:
    """A special token as tokenizer_config.json spells it: a string or an AddedToken object."""
    if value is None:
        return default
    if isinstance(value, dict):
        value = value.get("content")
    _require(isinstance(value, str) and value != "", f"cannot read special token {value!r}")
    return value


def _check_decoding(tokenizer: dict) -> None:
    """Refuse the tokenizer settings under which the reference decodes other text.

    The runtime decodes as Transformers' Wav2Vec2CTCTokenizer does by default:
    the delimiter becomes a space, case is kept and no tokenization spaces are
    cleaned up. A checkpoint that changes any of these, or names another
    tokenizer class (a phoneme tokenizer joins its tokens differently), would
    still build and run, but its transcript would not be the reference's.
    """
    tokenizer_class = tokenizer.get("tokenizer_class")
    _require(tokenizer_class in (None, "Wav2Vec2CTCTokenizer"),
             f"decodes as Wav2Vec2CTCTokenizer, not {tokenizer_class!r}")
    _require(tokenizer.get("do_lower_case", False) is False,
             "keeps the vocabulary's case; do_lower_case is not supported")
    _require(tokenizer.get("replace_word_delimiter_char", " ") == " ",
             "maps the word delimiter to a space; replace_word_delimiter_char "
             f"{tokenizer.get('replace_word_delimiter_char')!r} is not supported")
    _require(tokenizer.get("clean_up_tokenization_spaces", False) is False,
             "does not clean up tokenization spaces; clean_up_tokenization_spaces is not supported")


def expected_tensors(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Every tensor the resolved layout reads, with its exact shape."""
    hidden = cfg["hidden_size"]
    inner = cfg["intermediate_size"]
    dims = cfg["conv_dim"]
    tensors: dict[str, tuple[int, ...]] = {}
    for index, (width, kernel) in enumerate(zip(dims, cfg["conv_kernel"])):
        base = f"{PREFIX}feature_extractor.conv_layers.{index}."
        tensors[base + "conv.weight"] = (width, dims[index - 1] if index else 1, kernel)
        if cfg["conv_bias"]:
            tensors[base + "conv.bias"] = (width,)
        if cfg["feat_extract_norm"] == "layer" or index == 0:
            tensors[base + "layer_norm.weight"] = (width,)
            tensors[base + "layer_norm.bias"] = (width,)
    if cfg["feat_proj_layer_norm"]:
        tensors[PREFIX + "feature_projection.layer_norm.weight"] = (dims[-1],)
        tensors[PREFIX + "feature_projection.layer_norm.bias"] = (dims[-1],)
    tensors[PREFIX + "feature_projection.projection.weight"] = (hidden, dims[-1])
    tensors[PREFIX + "feature_projection.projection.bias"] = (hidden,)

    positional = PREFIX + "encoder.pos_conv_embed.conv."
    tensors[positional + "weight_g"] = (1, 1, cfg["pos_kernel"])
    tensors[positional + "weight_v"] = (hidden, hidden // cfg["pos_groups"], cfg["pos_kernel"])
    tensors[positional + "bias"] = (hidden,)
    tensors[PREFIX + "encoder.layer_norm.weight"] = (hidden,)
    tensors[PREFIX + "encoder.layer_norm.bias"] = (hidden,)
    for index in range(cfg["num_hidden_layers"]):
        base = f"{PREFIX}encoder.layers.{index}."
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors[f"{base}attention.{projection}.weight"] = (hidden, hidden)
            tensors[f"{base}attention.{projection}.bias"] = (hidden,)
        for norm in ("layer_norm", "final_layer_norm"):
            tensors[f"{base}{norm}.weight"] = (hidden,)
            tensors[f"{base}{norm}.bias"] = (hidden,)
        tensors[base + "feed_forward.intermediate_dense.weight"] = (inner, hidden)
        tensors[base + "feed_forward.intermediate_dense.bias"] = (inner,)
        tensors[base + "feed_forward.output_dense.weight"] = (hidden, inner)
        tensors[base + "feed_forward.output_dense.bias"] = (hidden,)
    tensors["lm_head.weight"] = (cfg["vocab_size"], hidden)
    tensors["lm_head.bias"] = (cfg["vocab_size"],)
    return tensors


def check_tensors(cfg: dict, shapes: dict[str, tuple[int, ...]]) -> None:
    """Refuse a checkpoint whose tensors are not exactly the resolved layout's.

    A missing tensor would be randomly initialised by the reference, and an
    unexpected one means the checkpoint carries a layer this family does not
    build (an adapter, a batch-normed positional convolution, another head).
    """
    expected = expected_tensors(cfg)
    missing = sorted(name for name in expected if name not in shapes)
    _require(not missing, "checkpoint is missing " + ", ".join(missing[:8])
             + (f" and {len(missing) - 8} more" if len(missing) > 8 else ""))
    unexpected = sorted(name for name in shapes
                        if name not in expected and name not in TRAINING_ONLY)
    _require(not unexpected, "checkpoint carries tensors this layout does not read: "
             + ", ".join(unexpected[:8]))
    wrong = [f"{name} {tuple(shapes[name])} != {shape}"
             for name, shape in expected.items() if tuple(shapes[name]) != shape]
    _require(not wrong, "checkpoint tensor shapes do not match the config: " + "; ".join(wrong[:4]))


def resolve(model_dir: Path, shapes: dict[str, tuple[int, ...]]) -> dict:
    """Return the geometry the builder needs, or raise naming what is wrong."""
    model_dir = Path(model_dir)
    cfg = _read_json(model_dir / "config.json", "config.json")
    pre = _read_json(model_dir / "preprocessor_config.json", "preprocessor_config.json")
    vocab = _read_json(model_dir / "vocab.json", "vocab.json")
    tokenizer = _read_json(model_dir / "tokenizer_config.json", "tokenizer_config.json",
                           required=False)

    _require(cfg.get("model_type") == "hubert",
             f"builds model_type 'hubert', not {cfg.get('model_type')!r}")
    architectures = tuple(cfg.get("architectures") or ())
    _require("HubertForCTC" in architectures,
             f"builds HubertForCTC, not {architectures or '(unstated)'}")

    norm = _setting(cfg, "feat_extract_norm")
    _require(norm in {"group", "layer"}, f"needs feat_extract_norm 'group' or 'layer', not {norm!r}")
    for key in ("do_stable_layer_norm", "conv_bias", "feat_proj_layer_norm", "conv_pos_batch_norm"):
        _require(isinstance(_setting(cfg, key), bool), f"needs a boolean {key}")
    _require(not _setting(cfg, "conv_pos_batch_norm"),
             "builds the weight-normed positional convolution, not conv_pos_batch_norm")
    _require(not cfg.get("add_adapter", False) and cfg.get("adapter_attn_dim") is None,
             "does not build adapters")
    _require(_setting(cfg, "hidden_act") == "gelu",
             f"builds the exact GELU, not hidden_act {_setting(cfg, 'hidden_act')!r}")
    _require(_setting(cfg, "feat_extract_activation") == "gelu",
             f"builds the exact GELU, not feat_extract_activation "
             f"{_setting(cfg, 'feat_extract_activation')!r}")

    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    _require(hidden % heads == 0, f"needs hidden_size {hidden} divisible by {heads} heads")

    kernels = [int(v) for v in cfg["conv_kernel"]]
    strides = [int(v) for v in cfg["conv_stride"]]
    dims = [int(v) for v in cfg["conv_dim"]]
    _require(len(kernels) == len(strides) == len(dims) > 0,
             "needs one kernel, stride and width per feature-encoder layer")

    pos_kernel = int(_setting(cfg, "num_conv_pos_embeddings"))
    # An even kernel over half its width of padding leaves one extra frame and
    # the reference drops the last one; an odd kernel would not need the trim.
    _require(pos_kernel > 0 and pos_kernel % 2 == 0,
             f"needs an even num_conv_pos_embeddings; {pos_kernel} would not need the same-pad trim")
    pos_groups = int(_setting(cfg, "num_conv_pos_embedding_groups"))
    _require(hidden % pos_groups == 0,
             f"needs hidden_size {hidden} divisible by {pos_groups} positional-conv groups")

    _require(int(pre.get("feature_size", 1)) == 1, "reads single-channel audio")
    _require(float(pre.get("padding_value", 0.0)) == 0.0, "needs a zero padding value")
    _require(pre.get("padding_side", "right") == "right", "needs right padding")
    sample_rate = int(pre.get("sampling_rate", 16000))
    _require(sample_rate > 0, "needs a positive sampling_rate")

    vocab_size = int(cfg["vocab_size"])
    _require(len(vocab) == vocab_size,
             f"vocab.json holds {len(vocab)} tokens but the config states {vocab_size}")
    tokens = [""] * vocab_size
    for token, index in vocab.items():
        index = int(index)
        _require(0 <= index < vocab_size, f"vocab.json maps {token!r} outside the vocabulary")
        _require(tokens[index] == "", f"vocab.json maps two tokens to id {index}")
        tokens[index] = token
    _require(all(tokens), "vocab.json leaves a token id unnamed")

    # The tokenizer's pad token is the CTC blank its decoder drops, and the
    # model's CTC loss uses config.pad_token_id; both must name the same id.
    pad = _token_text(tokenizer.get("pad_token"), "<pad>")
    _require(pad in vocab, f"needs the pad token {pad!r} in vocab.json to use as the CTC blank")
    blank = int(vocab[pad])
    _require(int(_setting(cfg, "pad_token_id")) == blank,
             f"config pad_token_id {_setting(cfg, 'pad_token_id')} is not the tokenizer's "
             f"CTC blank {pad!r} = {blank}")
    delimiter = _token_text(tokenizer.get("word_delimiter_token"), "|")
    _require(delimiter in vocab, f"needs the word delimiter {delimiter!r} in vocab.json")
    _check_decoding(tokenizer)

    resolved = {
        "hidden_size": hidden,
        "intermediate_size": int(cfg["intermediate_size"]),
        "num_attention_heads": heads,
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "eps": float(_setting(cfg, "layer_norm_eps")),
        "feat_extract_norm": norm,
        "do_stable_layer_norm": _setting(cfg, "do_stable_layer_norm"),
        "conv_bias": _setting(cfg, "conv_bias"),
        "feat_proj_layer_norm": _setting(cfg, "feat_proj_layer_norm"),
        "conv_kernel": kernels,
        "conv_stride": strides,
        "conv_dim": dims,
        "pos_kernel": pos_kernel,
        "pos_groups": pos_groups,
        "sample_rate": sample_rate,
        "do_normalize": bool(pre.get("do_normalize", True)),
        "tokens": tokens,
        "blank_token_id": blank,
        "word_delimiter_token": delimiter,
        "vocab_size": vocab_size,
    }
    check_tensors(resolved, shapes)
    return resolved
