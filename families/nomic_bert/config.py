# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the nomic-embed geometry out of the tensors, checking the file against them.

This checkpoint's ``config.json`` is a GPT-2 config with a BERT inside it, so
it states most things twice - ``n_embd`` and ``hidden_size``, ``n_layer`` and
``num_hidden_layers``, ``n_inner`` and ``intermediate_size`` - and once
inconsistently: ``n_positions`` is 8192 while ``max_position_embeddings`` and
``max_trained_positions`` are both 2048. Nothing here trusts a stated number
that a tensor can settle, and the one number no tensor can settle - the head
count - is read and then checked for divisibility.

The values that decide correctness rather than shape are refused when they
are not what this family builds, rather than being approximated:

* ``activation_function`` must be ``swiglu``; a gelu MLP has two weights
  where this has three.
* ``prenorm`` must be false. The norms sit after the residual add here, and
  a pre-norm reading of the same weights runs without complaint.
* ``rotary_emb_fraction`` must be 1.0 and ``rotary_emb_interleaved`` false.
* ``rotary_scaling_factor`` must be unset: the dynamic-NTK variant changes
  the rotary base with the input length, which a fixed-length engine with a
  build-time table cannot reproduce.
* every projection bias flag must be false, and the tensors are checked to
  agree - a stated flag and a missing tensor are two different claims.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_LAYER = re.compile(r"^encoder\.layers\.(\d+)\.")


def _load(model_dir: Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing nomic_bert config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(config: dict, keys: tuple[str, ...], measured: int, what: str) -> None:
    """Refuse a stated value the tensors contradict."""
    for key in keys:
        if key in config and int(config[key]) != measured:
            raise ValueError(
                f"nomic_bert config says {key}={config[key]} but the tensors say "
                f"{what}={measured}")


def _layer_count(weights: dict) -> int:
    indices = sorted({int(m.group(1)) for key in weights if (m := _LAYER.match(key))})
    if not indices:
        raise ValueError("nomic_bert checkpoint has no encoder.layers.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"nomic_bert layer indices are not contiguous: {indices}")
    return len(indices)


def resolve(model_dir, weights: dict) -> dict:
    config = _load(model_dir)

    vocab_size, hidden_size = np.asarray(weights["embeddings.word_embeddings.weight"]).shape
    intermediate = int(np.asarray(weights["encoder.layers.0.mlp.fc11.weight"]).shape[0])
    num_layers = _layer_count(weights)
    _agree(config, ("hidden_size", "n_embd"), int(hidden_size), "hidden_size")
    _agree(config, ("vocab_size",), int(vocab_size), "vocab_size")
    _agree(config, ("intermediate_size", "n_inner"), intermediate, "intermediate_size")
    _agree(config, ("num_hidden_layers", "n_layer"), num_layers, "num_layers")

    # The only number the tensors cannot settle: the packed QKV is one matrix
    # whatever the head count is. So it is read, and then checked.
    num_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
    if num_heads <= 0:
        raise ValueError("nomic_bert config does not state a head count")
    if int(hidden_size) % num_heads != 0:
        raise ValueError(
            f"nomic_bert hidden_size {hidden_size} is not divisible by {num_heads} heads")
    head_dim = int(hidden_size) // num_heads
    if "head_dim" in config and int(config["head_dim"]) != head_dim:
        raise ValueError(
            f"nomic_bert config says head_dim={config['head_dim']} but "
            f"{hidden_size}/{num_heads} is {head_dim}")
    kv_heads = config.get("num_heads_kv")
    if kv_heads is not None and int(kv_heads) != num_heads:
        raise NotImplementedError("nomic_bert does not support grouped-query attention")

    qkv_rows = int(np.asarray(weights["encoder.layers.0.attn.Wqkv.weight"]).shape[0])
    if qkv_rows != 3 * int(hidden_size):
        raise ValueError(
            f"nomic_bert expects a packed QKV of {3 * int(hidden_size)} rows, found {qkv_rows}")

    activation = str(config.get("activation_function", "")).lower()
    if activation != "swiglu":
        raise NotImplementedError(
            f"nomic_bert builds the swiglu MLP only, not {activation!r}")
    if config.get("prenorm", False):
        raise NotImplementedError("nomic_bert builds the post-norm block only")
    if config.get("causal", False):
        raise NotImplementedError("nomic_bert builds the bidirectional encoder only")
    if int(config.get("moe_every_n_layers", 0) or 0) > 0:
        raise NotImplementedError("nomic_bert does not build the mixture-of-experts variant")

    fraction = float(config.get("rotary_emb_fraction", 1.0))
    if fraction != 1.0:
        raise NotImplementedError(
            f"nomic_bert rotates the whole head, not a fraction of {fraction}")
    if config.get("rotary_emb_interleaved", False):
        raise NotImplementedError("nomic_bert builds the non-interleaved rotary only")
    if config.get("rotary_scaling_factor") is not None:
        raise NotImplementedError(
            "nomic_bert does not build the dynamic-NTK rotary, whose base depends on the "
            "input length and so cannot be a build-time table")
    if config.get("rotary_emb_scale_base") is not None:
        raise NotImplementedError("nomic_bert does not build the xpos-scaled rotary")

    for flag in ("qkv_proj_bias", "mlp_fc1_bias", "mlp_fc2_bias"):
        if config.get(flag, False):
            raise NotImplementedError(f"nomic_bert builds biasless projections, but {flag} is set")
    for name in ("attn.Wqkv", "attn.out_proj", "mlp.fc11", "mlp.fc12", "mlp.fc2"):
        if f"encoder.layers.0.{name}.bias" in weights:
            raise ValueError(
                f"nomic_bert found a {name}.bias the config says should not exist")

    type_vocab = int(config.get("type_vocab_size", 0))
    if type_vocab <= 0:
        raise NotImplementedError("nomic_bert expects token type embeddings")
    if "embeddings.position_embeddings.weight" in weights:
        raise NotImplementedError(
            "nomic_bert expects rotary positions, but this checkpoint carries a position table")

    eps = float(config.get("layer_norm_epsilon", config.get("layer_norm_eps", 1e-12)))
    # max_trained_positions is the honest limit. n_positions is 8192 here and
    # the model was trained to 2048; building past what it was trained on
    # would be inventing a capability the checkpoint does not claim.
    max_positions = int(config.get("max_trained_positions")
                        or config.get("max_position_embeddings") or 0)
    if max_positions <= 0:
        raise ValueError("nomic_bert config does not state a trained position limit")

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "eps": eps,
        "rotary_base": float(config.get("rotary_emb_base", 10000.0)),
        "max_positions": max_positions,
        "pad_token_id": int(config.get("pad_token_id", 0)),
    }
