# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the MobileBERT geometry out of the tensors and check the config against it.

MobileBERT is configured by switches that change the graph, not the weights,
and several of them leave every tensor loadable when they are ignored:

* ``normalization_type`` is ``no_norm``: every "LayerNorm" of the encoder is
  an elementwise ``x * weight + bias`` without statistics. The tensors are
  named ``LayerNorm`` and have the LayerNorm shapes, so a reader that wires a
  real layer normalisation loads them all and computes a different model. Only
  the masked-LM head transform keeps a real layer normalisation.
* ``trigram_input``: the 128-wide token embedding of each row is concatenated
  with those of its right and left neighbours before the projection to the
  hidden width.
* ``use_bottleneck`` with ``key_query_shared_bottleneck`` and without
  ``use_bottleneck_attention``: query and key read one shared 128-wide
  bottleneck of the 512-wide stream, the value reads the 512-wide stream
  itself, and the attention residual is a second bottleneck.
* ``classifier_activation`` false: the pooled row is the first row as it is,
  with no dense layer and no tanh.

The family builds exactly that layout, which is the one the published
checkpoint uses, and refuses the others by name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_PREFIX = "mobilebert."
_LAYER = re.compile(r"^mobilebert\.encoder\.layer\.(\d+)\.")
_FFN = re.compile(r"^mobilebert\.encoder\.layer\.0\.ffn\.(\d+)\.")

# The switches of the layout this family builds, with the value it requires.
_LAYOUT = {
    "normalization_type": "no_norm",
    "trigram_input": True,
    "use_bottleneck": True,
    "key_query_shared_bottleneck": True,
    "use_bottleneck_attention": False,
    "classifier_activation": False,
    "hidden_act": "relu",
}

# The defaults of transformers' MobileBertConfig for the switches above, used
# when config.json leaves a key out.
_DEFAULTS = {
    "normalization_type": "no_norm",
    "trigram_input": True,
    "use_bottleneck": True,
    "key_query_shared_bottleneck": True,
    "use_bottleneck_attention": False,
    "classifier_activation": True,
    "hidden_act": "relu",
}

_ARCHITECTURE = "MobileBertForPreTraining"


def _load(model_dir: Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing mobilebert config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("mobilebert config.json must be a JSON object")
    return config


def _shape(weights: dict, key: str) -> tuple[int, ...]:
    if key not in weights:
        raise KeyError(f"mobilebert checkpoint is missing {key}")
    return tuple(int(value) for value in np.asarray(weights[key]).shape)


def _agree(config: dict, key: str, measured: int, what: str) -> None:
    if key in config and int(config[key]) != measured:
        raise ValueError(
            f"mobilebert config says {key}={config[key]} but the tensors say {what}={measured}")


def _layer_count(weights: dict) -> int:
    indices = sorted({int(match.group(1)) for key in weights if (match := _LAYER.match(key))})
    if not indices:
        raise ValueError("mobilebert checkpoint has no mobilebert.encoder.layer.* tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"mobilebert layer indices are not contiguous: {indices}")
    return len(indices)


def _ffn_count(weights: dict) -> int:
    """Extra feed-forward blocks per layer: ``num_feedforward_networks - 1``."""
    indices = sorted({int(match.group(1)) for key in weights if (match := _FFN.match(key))})
    if indices != list(range(len(indices))):
        raise ValueError(f"mobilebert layer 0 feed-forward indices are not contiguous: {indices}")
    return len(indices)


def _check_layout(config: dict) -> None:
    for key, required in _LAYOUT.items():
        value = config.get(key, _DEFAULTS[key])
        if isinstance(required, bool):
            if not isinstance(value, bool):
                raise ValueError(f"mobilebert config {key} must be a boolean, not {value!r}")
        elif isinstance(value, str):
            value = value.lower()
        if value != required:
            raise NotImplementedError(
                f"mobilebert builds the {key}={required!r} layout only, not {key}={value!r}")


def resolve(model_dir, weights: dict) -> dict:
    """The geometry and ids the builder needs, checked against config.json."""
    config = _load(model_dir)
    if config.get("model_type") != "mobilebert":
        raise ValueError(
            f"mobilebert config has model_type={config.get('model_type')!r}, not 'mobilebert'")
    architectures = config.get("architectures") or []
    if _ARCHITECTURE not in architectures:
        raise NotImplementedError(
            "mobilebert builds the pretraining heads of a MobileBertForPreTraining checkpoint "
            f"(masked LM and next sentence); this checkpoint declares {architectures}")
    _check_layout(config)

    vocab_size, embedding_size = _shape(weights, f"{_PREFIX}embeddings.word_embeddings.weight")
    max_positions, hidden_size = _shape(weights, f"{_PREFIX}embeddings.position_embeddings.weight")
    type_vocab_size, type_width = _shape(weights,
                                         f"{_PREFIX}embeddings.token_type_embeddings.weight")
    projected, trigram = _shape(weights, f"{_PREFIX}embeddings.embedding_transformation.weight")
    if type_width != hidden_size or projected != hidden_size:
        raise ValueError("mobilebert embedding tables disagree about the hidden width")
    if trigram != 3 * embedding_size:
        raise ValueError(
            f"mobilebert embedding_transformation reads {trigram} values, not the "
            f"3 x {embedding_size} of a trigram input")
    all_heads, bottleneck = _shape(weights, f"{_PREFIX}encoder.layer.0.attention.self.query.weight")
    if all_heads != bottleneck:
        raise ValueError("mobilebert query projection is not square in the bottleneck width")
    intermediate, _ = _shape(weights, f"{_PREFIX}encoder.layer.0.intermediate.dense.weight")
    num_layers = _layer_count(weights)
    extra_ffns = _ffn_count(weights)

    _agree(config, "vocab_size", vocab_size, "vocab_size")
    _agree(config, "embedding_size", embedding_size, "embedding_size")
    _agree(config, "hidden_size", hidden_size, "hidden_size")
    _agree(config, "max_position_embeddings", max_positions, "max_positions")
    _agree(config, "type_vocab_size", type_vocab_size, "type_vocab_size")
    _agree(config, "intra_bottleneck_size", bottleneck, "bottleneck width")
    _agree(config, "true_hidden_size", bottleneck, "bottleneck width")
    _agree(config, "intermediate_size", intermediate, "intermediate_size")
    _agree(config, "num_hidden_layers", num_layers, "num_layers")
    _agree(config, "num_feedforward_networks", extra_ffns + 1, "feed-forward blocks per layer")

    num_heads = int(config.get("num_attention_heads", 0))
    if num_heads <= 0 or bottleneck % num_heads:
        raise ValueError(
            f"mobilebert bottleneck width {bottleneck} does not split into {num_heads} heads")
    if hidden_size <= embedding_size:
        raise ValueError("mobilebert hidden width must exceed the embedding width")
    head_rows, head_vocab = _shape(weights, "cls.predictions.dense.weight")
    if (head_rows, head_vocab) != (hidden_size - embedding_size, vocab_size):
        raise ValueError(
            f"mobilebert cls.predictions.dense.weight is {(head_rows, head_vocab)}, not "
            f"{(hidden_size - embedding_size, vocab_size)}")
    pad_token_id = int(config.get("pad_token_id", 0))
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"mobilebert pad_token_id {pad_token_id} is outside the vocabulary")

    return {
        "vocab_size": vocab_size,
        "embedding_size": embedding_size,
        "hidden_size": hidden_size,
        "bottleneck_size": bottleneck,
        "intermediate_size": intermediate,
        "num_layers": num_layers,
        "extra_ffns": extra_ffns,
        "num_heads": num_heads,
        "head_dim": bottleneck // num_heads,
        "type_vocab_size": type_vocab_size,
        "max_positions": max_positions,
        "head_eps": float(config.get("layer_norm_eps", 1e-12)),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", True)),
        "pad_token_id": pad_token_id,
    }
