# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the DeBERTa-v2/v3 geometry out of the tensors, checking the file against them.

The value worth the most attention here is ``position_biased_input``. It is
**false** on every v3 checkpoint, which means the absolute position table is
not added to the embeddings - and yet
``embeddings.position_embeddings.weight`` is still present in the file, 512
rows of it, fully trained. A reader who wires it up the way every other BERT
wires it up gets a model that loads every tensor, runs, and is wrong. It is
read from the config and refused when true, rather than inferred from the
tensor being there.

The rest of the refusals are for variants this family does not build:

* ``pos_att_type`` must be exactly ``c2p`` and ``p2c``. The term count sets
  the attention scale, so a checkpoint with one term is not a subset of this
  one - it divides by a different number.
* ``share_att_key`` must be true. When it is false the positional
  projections are their own tensors, which this checkpoint does not carry.
* ``norm_rel_ebd`` must ask for the layer norm.
* ``conv_kernel_size`` must be unset: DeBERTa-v2-xlarge puts a convolution
  beside the first layer, which is a different model.
* ``type_vocab_size`` must be 0 and the embedding width must equal the hidden
  width, because neither the token-type table nor ``embed_proj`` exists here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_LAYER = re.compile(r"encoder\.layer\.(\d+)\.")


def _load(model_dir: Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing deberta_v2 config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(config: dict, key: str, measured: int, what: str) -> None:
    if key in config and int(config[key]) != measured:
        raise ValueError(
            f"deberta_v2 config says {key}={config[key]} but the tensors say {what}={measured}")


def _layer_count(weights: dict, prefix: str) -> int:
    indices = sorted({int(m.group(1)) for key in weights
                      if key.startswith(prefix) and (m := _LAYER.search(key))})
    if not indices:
        raise ValueError("deberta_v2 checkpoint has no encoder.layer.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"deberta_v2 layer indices are not contiguous: {indices}")
    return len(indices)


def resolve(model_dir, weights: dict, prefix: str) -> dict:
    config = _load(model_dir)

    vocab_size, hidden_size = np.asarray(
        weights[f"{prefix}embeddings.word_embeddings.weight"]).shape
    intermediate = int(np.asarray(
        weights[f"{prefix}encoder.layer.0.intermediate.dense.weight"]).shape[0])
    num_layers = _layer_count(weights, prefix)
    _agree(config, "hidden_size", int(hidden_size), "hidden_size")
    _agree(config, "vocab_size", int(vocab_size), "vocab_size")
    _agree(config, "intermediate_size", intermediate, "intermediate_size")
    _agree(config, "num_hidden_layers", num_layers, "num_layers")

    embedding_size = int(config.get("embedding_size", hidden_size))
    if embedding_size != int(hidden_size):
        raise NotImplementedError(
            "deberta_v2 does not build the projected-embedding variant (embed_proj)")

    num_heads = int(config.get("num_attention_heads", 0))
    if num_heads <= 0:
        raise ValueError("deberta_v2 config does not state a head count")
    if int(hidden_size) % num_heads != 0:
        raise ValueError(
            f"deberta_v2 hidden_size {hidden_size} is not divisible by {num_heads} heads")
    head_dim = int(hidden_size) // num_heads

    if not config.get("relative_attention", False):
        raise NotImplementedError("deberta_v2 builds the relative-attention model only")
    if config.get("position_biased_input", True):
        # The table is in the file either way; only the config says whether
        # the reference adds it.
        raise NotImplementedError(
            "deberta_v2 does not build a checkpoint that adds absolute position embeddings")

    raw = config.get("pos_att_type") or []
    terms = tuple(sorted(raw.split("|") if isinstance(raw, str) else [str(v) for v in raw]))
    if terms != ("c2p", "p2c"):
        raise NotImplementedError(
            f"deberta_v2 builds the c2p and p2c terms together, not {list(terms)}")
    if not config.get("share_att_key", False):
        raise NotImplementedError(
            "deberta_v2 builds the shared-projection form only (share_att_key)")
    if "layer_norm" not in str(config.get("norm_rel_ebd", "")):
        raise NotImplementedError(
            "deberta_v2 expects the relative embeddings to be layer-normalised")
    if int(config.get("conv_kernel_size", 0) or 0) > 0:
        raise NotImplementedError("deberta_v2 does not build the convolution variant")
    if int(config.get("type_vocab_size", 0) or 0) > 0:
        raise NotImplementedError("deberta_v2 does not build token type embeddings")
    activation = str(config.get("hidden_act", "gelu")).lower()
    if activation != "gelu":
        raise NotImplementedError(f"deberta_v2 builds the exact gelu only, not {activation!r}")

    buckets = int(config.get("position_buckets", -1))
    max_positions = int(config.get("max_position_embeddings", 512))
    max_relative = int(config.get("max_relative_positions", -1))
    if max_relative < 1:
        max_relative = max_positions
    span = buckets if buckets > 0 else max_relative
    if span <= 0:
        raise ValueError("deberta_v2 cannot determine the relative position span")

    rel_rows = int(np.asarray(weights[f"{prefix}encoder.rel_embeddings.weight"]).shape[0])
    if rel_rows != 2 * span:
        raise ValueError(
            f"deberta_v2 relative table has {rel_rows} rows, expected {2 * span} "
            f"for a span of {span}")

    return {
        "prefix": prefix,
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "eps": float(config.get("layer_norm_eps", 1e-7)),
        "position_buckets": buckets,
        "max_relative_positions": max_relative,
        "span": span,
        "max_positions": max_positions,
        # One content term plus one per positional term. This is the number
        # the attention scale is built from, so it is derived from the terms
        # actually built rather than hard-coded.
        "scale_factor": 1 + len(terms),
        "pad_token_id": int(config.get("pad_token_id", 0)),
    }
