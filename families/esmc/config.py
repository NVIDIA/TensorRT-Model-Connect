# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the ESMC geometry out of the tensors, checking the file against them.

The checkpoint is a Llama-shaped stack with five differences, and each one is
a config bit that the builder cannot recover from the tensor names alone:

* ``attention_bias`` and ``mlp_bias`` must be false. Every ESMC release is
  bias-free, so a projection that carries a bias is a different graph; wiring
  it without the bias loads every tensor and runs, quietly wrong.
* ``hidden_act`` must be ``silu``. The MLP is SwiGLU - ``down(silu(gate(x)) *
  up(x))`` - and the gelu of a ``families/esm``-shaped MLP is a different
  function on the same three tensors.
* ``tie_word_embeddings`` must be false. ESMC stores ``lm_head.decoder.weight``
  separately, so a checkpoint that ties it is a different head, not a missing
  tensor to be filled from the embedding table.
* ``rope_parameters`` must be the default rotation at theta 10000. No
  checkpoint ships an ``inv_freq`` buffer - unlike ESM-2, whose stored
  schedule has to be read - so the schedule is recomputed here and any other
  schedule is refused rather than silently swapped.
* ``num_key_value_heads`` must equal ``num_attention_heads``. The reference
  asserts it; this reader refuses it, because a grouped-query checkpoint
  divides the same q/k/v tensors by a different head count.

The layer count, the hidden width, the intermediate width and the vocabulary
width are measured from the tensors and cross-checked against the file.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

_LAYER = re.compile(r"layers\.(\d+)\.")

# The residual branches are divided by this, and the reference computes it from
# the layer count rather than reading it: ``sqrt(num_hidden_layers / 36)`` is
# 1.0 for the 36-layer checkpoint and 0.912871 for the 30-layer one. Omitting it
# leaves the 36-layer model numerically unchanged and moves the 30-layer one,
# which is why it is a config value with a test rather than a literal.
_RESIDUAL_REFERENCE_LAYERS = 36

# Torch's ``nn.LayerNorm`` default. The reference constructs every ESMC norm
# without an explicit eps and ``config.json`` does not carry one, so there is
# nothing to read.
_LAYER_NORM_EPS = 1e-5


def _load(model_dir: Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing esmc config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agree(config: dict, key: str, measured: int, what: str) -> None:
    if key in config and int(config[key]) != measured:
        raise ValueError(
            f"esmc config says {key}={config[key]} but the tensors say {what}={measured}"
        )


def _layer_count(weights: dict, prefix: str) -> int:
    indices = sorted(
        {int(m.group(1)) for key in weights if key.startswith(prefix) and (m := _LAYER.search(key))}
    )
    if not indices:
        raise ValueError("esmc checkpoint has no layers.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"esmc layer indices are not contiguous: {indices}")
    return len(indices)


def _bool(config: dict, key: str) -> bool:
    return bool(config.get(key, False))


def resolve(model_dir, weights: dict, prefix: str) -> dict:
    config = _load(model_dir)

    model_type = config.get("model_type")
    if model_type not in (None, "esmc"):
        raise ValueError(f"esmc builds model_type='esmc', not {model_type!r}")

    # Geometry, measured rather than declared.
    vocab_size, hidden_size = np.asarray(weights[f"{prefix}embed_tokens.weight"]).shape
    intermediate_size = int(np.asarray(weights[f"{prefix}layers.0.mlp.gate_proj.weight"]).shape[0])
    num_layers = _layer_count(weights, prefix)
    _agree(config, "hidden_size", int(hidden_size), "hidden_size")
    _agree(config, "vocab_size", int(vocab_size), "vocab_size")
    _agree(config, "intermediate_size", intermediate_size, "intermediate_size")
    _agree(config, "num_hidden_layers", num_layers, "num_layers")

    num_heads = int(config.get("num_attention_heads", 0))
    if num_heads <= 0:
        raise ValueError("esmc config does not state a head count")
    if int(hidden_size) % num_heads != 0:
        raise ValueError(f"esmc hidden_size {hidden_size} is not divisible by {num_heads} heads")
    head_dim = int(hidden_size) // num_heads
    declared = config.get("head_dim")
    if declared is not None and int(declared) != head_dim:
        raise NotImplementedError(
            f"esmc builds head_dim={head_dim} (hidden_size / heads), not {declared}"
        )
    if head_dim % 2 != 0:
        raise NotImplementedError(f"esmc rotary needs an even head width, not {head_dim}")

    heads_kv = config.get("num_key_value_heads")
    if heads_kv is not None and int(heads_kv) != num_heads:
        raise NotImplementedError(
            f"esmc does not build grouped-query attention: num_key_value_heads={heads_kv} "
            f"with num_attention_heads={num_heads}"
        )

    activation = str(config.get("hidden_act", "silu")).lower()
    if activation != "silu":
        raise NotImplementedError(f"esmc builds the silu MLP only, not {activation!r}")
    if _bool(config, "attention_bias") or _bool(config, "mlp_bias"):
        raise NotImplementedError("esmc builds the bias-free projections only")
    if _bool(config, "tie_word_embeddings"):
        raise NotImplementedError(
            "esmc builds tie_word_embeddings=false: lm_head.decoder.weight is stored separately"
        )

    rope = config.get("rope_parameters") or {}
    if not isinstance(rope, dict):
        raise NotImplementedError("esmc reads rope_parameters as an object")
    rope_type = str(rope.get("rope_type", "default")).lower()
    if rope_type != "default":
        raise NotImplementedError(f"esmc builds the default rotation only, not {rope_type!r}")
    rope_theta = float(rope.get("rope_theta", 10000.0))
    if not math.isclose(rope_theta, 10000.0):
        raise NotImplementedError(f"esmc builds theta 10000 only, not {rope_theta}")

    if "layer_norm_eps" in config and not math.isclose(
        float(config["layer_norm_eps"]), _LAYER_NORM_EPS
    ):
        raise NotImplementedError(
            f"esmc builds torch's default LayerNorm eps {_LAYER_NORM_EPS}, not "
            f"{config['layer_norm_eps']}"
        )

    mask_token_id = int(config.get("mask_token_id", -1))
    if not 0 <= mask_token_id < int(vocab_size):
        raise ValueError("esmc config does not state a usable mask_token_id")

    return {
        "prefix": prefix,
        "vocab_size": int(vocab_size),
        "hidden_size": int(hidden_size),
        "intermediate_size": intermediate_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "eps": _LAYER_NORM_EPS,
        "rope_theta": rope_theta,
        "residual_scale": math.sqrt(num_layers / _RESIDUAL_REFERENCE_LAYERS),
        "max_positions": int(config.get("max_position_embeddings", 2048)),
        "mask_token_id": mask_token_id,
        "pad_token_id": int(config.get("pad_token_id", 1)),
    }
