# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the GTE-v1.5 geometry out of the tensors, checking the file against them.

Unlike the GPT-2-shaped config next door in ``families/nomic_bert``, this one
states each number once. What it does instead is state several *switches*
that change the arithmetic without changing any shape, so a checkpoint that
flips one of them would load, run, and answer wrong:

* ``position_embedding_type`` selects ``rope`` or a learned ``absolute``
  table. Both read the same tensors; only one of them is this model.
* ``layer_norm_type`` selects ``layer_norm`` or ``rms_norm``. An RMSNorm
  checkpoint has no bias tensors, so that one the tensors can also settle,
  and both claims are checked.
* ``hidden_act`` selects the MLP activation. This family builds the exact
  erf gelu only.
* ``pack_qkv`` decides whether the projection is one matrix or three. The
  tensors settle it, and the flag is checked to agree.
* ``type_vocab_size`` is **0** here, so there is no token type embedding at
  all - a difference from every BERT-shaped encoder in this repo, and from
  ``families/nomic_bert``, which adds a constant row-0 shift.
* ``logn_attention_scale`` would multiply the query by a length-dependent
  scalar. The published remote code leaves it as a TODO and never applies
  it, so a checkpoint that set it would mean something this family does not
  build.
* ``rope_scaling`` is the one that actually fires on the shipped checkpoint:
  ``{"type": "ntk", "factor": 2.0}``. See ``rope.py`` for what it does to
  the frequencies. The ``mixed_b`` variant of the same NTK branch is a
  different formula and is refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_LAYER = re.compile(r"^encoder\.layer\.(\d+)\.")


def _load(model_dir) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing gte config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(config: dict, key: str, measured: int, what: str) -> None:
    """Refuse a stated value the tensors contradict."""
    if key in config and int(config[key]) != measured:
        raise ValueError(
            f"gte config says {key}={config[key]} but the tensors say {what}={measured}")


def _layer_count(weights: dict) -> int:
    indices = sorted({int(m.group(1)) for key in weights if (m := _LAYER.match(key))})
    if not indices:
        raise ValueError("gte checkpoint has no encoder.layer.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"gte layer indices are not contiguous: {indices}")
    return len(indices)


def _rope_scaling(config: dict) -> float:
    scaling = config.get("rope_scaling")
    if scaling is None:
        return 1.0
    if not isinstance(scaling, dict):
        raise ValueError(f"gte rope_scaling is not a mapping: {scaling!r}")
    kind = str(scaling.get("type", "")).lower()
    if kind != "ntk":
        # The published remote code implements the ntk branch and raises on
        # everything else; linear and dynamic are commented out there.
        raise NotImplementedError(f"gte builds the ntk rotary scaling only, not {kind!r}")
    if scaling.get("mixed_b") is not None:
        # Mixed NTK divides each frequency by its own lambda rather than by
        # one factor. Same shape, different table.
        raise NotImplementedError("gte does not build the mixed-b NTK rotary")
    factor = float(scaling.get("factor", 1.0))
    if factor < 1.0:
        raise ValueError(f"gte rope_scaling factor must be at least 1.0, got {factor}")
    return factor


def resolve(model_dir, weights: dict) -> dict:
    config = _load(model_dir)

    vocab_size, hidden_size = np.asarray(weights["embeddings.word_embeddings.weight"]).shape
    num_layers = _layer_count(weights)
    up_gate_rows = int(np.asarray(weights["encoder.layer.0.mlp.up_gate_proj.weight"]).shape[0])
    if up_gate_rows % 2 != 0:
        raise ValueError(
            f"gte up_gate_proj has {up_gate_rows} rows, which is not two stacked matrices")
    intermediate = up_gate_rows // 2
    down_columns = int(np.asarray(weights["encoder.layer.0.mlp.down_proj.weight"]).shape[1])
    if down_columns != intermediate:
        raise ValueError(
            f"gte up_gate_proj implies an intermediate size of {intermediate} but down_proj "
            f"takes {down_columns}")

    _agree(config, "hidden_size", int(hidden_size), "hidden_size")
    _agree(config, "vocab_size", int(vocab_size), "vocab_size")
    _agree(config, "intermediate_size", intermediate, "intermediate_size")
    _agree(config, "num_hidden_layers", num_layers, "num_layers")

    # The packed QKV is one matrix whatever the head count is, so this is the
    # number the tensors cannot settle. It is read, and then checked.
    num_heads = int(config.get("num_attention_heads") or 0)
    if num_heads <= 0:
        raise ValueError("gte config does not state a head count")
    if int(hidden_size) % num_heads != 0:
        raise ValueError(
            f"gte hidden_size {hidden_size} is not divisible by {num_heads} heads")
    head_dim = int(hidden_size) // num_heads
    if head_dim % 2 != 0:
        raise ValueError(f"gte needs an even head_dim for the rotary, got {head_dim}")

    qkv_rows = int(np.asarray(weights["encoder.layer.0.attention.qkv_proj.weight"]).shape[0])
    if qkv_rows != 3 * int(hidden_size):
        raise ValueError(
            f"gte expects a packed QKV of {3 * int(hidden_size)} rows, found {qkv_rows}")
    if not config.get("pack_qkv", True):
        raise ValueError(
            "gte config says pack_qkv is false but the checkpoint carries a packed qkv_proj")

    position_kind = str(config.get("position_embedding_type", "rope")).lower()
    if position_kind != "rope":
        raise NotImplementedError(
            f"gte builds the rotary variant only, not position_embedding_type={position_kind!r}")
    if "embeddings.position_embeddings.weight" in weights:
        raise NotImplementedError(
            "gte expects rotary positions, but this checkpoint carries a position table")

    norm_kind = str(config.get("layer_norm_type", "layer_norm")).lower()
    if norm_kind != "layer_norm":
        raise NotImplementedError(f"gte builds the layer_norm variant only, not {norm_kind!r}")

    activation = str(config.get("hidden_act", "gelu")).lower()
    if activation != "gelu":
        raise NotImplementedError(f"gte builds the exact gelu only, not {activation!r}")

    type_vocab = int(config.get("type_vocab_size", 0) or 0)
    if type_vocab > 0:
        raise NotImplementedError(
            "gte builds the token-type-free embedding; this checkpoint declares "
            f"type_vocab_size={type_vocab}")
    if "embeddings.token_type_embeddings.weight" in weights:
        raise ValueError("gte found a token_type_embeddings table the config denies")

    if config.get("logn_attention_scale", False):
        raise NotImplementedError(
            "gte does not build the logn attention scale, whose query multiplier depends on "
            "the input length")

    # Three biases exist and one does not. The flags are implicit in this
    # architecture, so the tensors are the only claim, and both directions
    # are checked: a missing bias is as wrong as an unexpected one.
    for name in ("attention.qkv_proj", "attention.o_proj", "mlp.down_proj"):
        if f"encoder.layer.0.{name}.bias" not in weights:
            raise KeyError(f"gte checkpoint is missing {name}.bias")
    if "encoder.layer.0.mlp.up_gate_proj.bias" in weights:
        raise ValueError("gte found an up_gate_proj.bias, which this architecture does not have")
    for suffix in ("weight", "bias"):
        for name in ("attn_ln", "mlp_ln"):
            if f"encoder.layer.0.{name}.{suffix}" not in weights:
                raise KeyError(f"gte checkpoint is missing {name}.{suffix}")

    max_positions = int(config.get("max_position_embeddings") or 0)
    if max_positions <= 0:
        raise ValueError("gte config does not state a position limit")

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "eps": float(config.get("layer_norm_eps", 1e-12)),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "rope_scaling_factor": _rope_scaling(config),
        "max_positions": max_positions,
        "pad_token_id": int(config.get("pad_token_id", 0)),
    }
