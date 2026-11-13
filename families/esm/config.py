# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the ESM-2 geometry out of the tensors, checking the file against them.

Three config values decide whether this family builds the checkpoint at all,
and all three are read rather than inferred from which tensors happen to be
present:

* ``position_embedding_type`` must be ``rotary``. ESM-1b and ESM-1v use
  learned absolute positions and ship a ``position_embeddings`` table; a
  reader that ignores this key builds a model that loads every tensor it
  looks for and places every residue wrongly.
* ``emb_layer_norm_before`` must be false. When true there is an extra
  normalisation between the word lookup and the first block, which is a
  different graph, not a different weight.
* ``token_dropout`` is carried through to the builder rather than assumed.
  It rescales the whole embedding by a factor that depends on how many
  ``<mask>`` positions the *input* has, so a build that drops it is correct
  only on inputs with no mask at all - which is exactly the input a
  masked-language test would not use.
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
        raise FileNotFoundError(f"missing esm config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(config: dict, key: str, measured: int, what: str) -> None:
    if key in config and int(config[key]) != measured:
        raise ValueError(f"esm config says {key}={config[key]} but the tensors say {what}={measured}")


def _layer_count(weights: dict, prefix: str) -> int:
    indices = sorted({int(m.group(1)) for key in weights
                      if key.startswith(prefix) and (m := _LAYER.search(key))})
    if not indices:
        raise ValueError("esm checkpoint has no encoder.layer.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"esm layer indices are not contiguous: {indices}")
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

    num_heads = int(config.get("num_attention_heads", 0))
    if num_heads <= 0:
        raise ValueError("esm config does not state a head count")
    if int(hidden_size) % num_heads != 0:
        raise ValueError(f"esm hidden_size {hidden_size} is not divisible by {num_heads} heads")
    head_dim = int(hidden_size) // num_heads
    if head_dim % 2 != 0:
        raise NotImplementedError(f"esm rotary needs an even head width, not {head_dim}")

    position = str(config.get("position_embedding_type", "absolute")).lower()
    if position != "rotary":
        raise NotImplementedError(
            f"esm builds the rotary variant only, not position_embedding_type={position!r}")
    if config.get("emb_layer_norm_before", False):
        raise NotImplementedError("esm does not build the emb_layer_norm_before variant")
    if config.get("is_folding_model", False) or config.get("esmfold_config") is not None:
        raise NotImplementedError("esm does not build the ESMFold variant")
    activation = str(config.get("hidden_act", "gelu")).lower()
    if activation != "gelu":
        raise NotImplementedError(f"esm builds the exact gelu only, not {activation!r}")

    mask_token_id = int(config.get("mask_token_id", -1))
    if not 0 <= mask_token_id < int(vocab_size):
        raise ValueError("esm config does not state a usable mask_token_id")

    return {
        "prefix": prefix,
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "eps": float(config.get("layer_norm_eps", 1e-5)),
        "max_positions": int(config.get("max_position_embeddings", 1026)),
        "token_dropout": bool(config.get("token_dropout", False)),
        "mask_token_id": mask_token_id,
        "pad_token_id": int(config.get("pad_token_id", 1)),
    }
