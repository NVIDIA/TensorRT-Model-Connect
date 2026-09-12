# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map the language model's checkpoint tensors onto the decoder's WeightDict.

The published tensors use the ordinary Hugging Face Qwen3 layout --
``model.layers.{i}.self_attn.q_proj.weight`` and friends. The standard decoder
builder wants the flat convention ``layer.{i}.w_q`` documented on
:class:`.checkpoint_mapper.WeightDict`, and it wants the projections
**transposed**: a checkpoint stores ``[out, in]`` because ``torch.nn.Linear``
computes ``x @ W.T``, while the builder multiplies ``x @ W`` and so needs
``[in, out]``.

Two details are specific to this model rather than to the convention:

* ``q_norm`` and ``k_norm`` are per-head vectors of length ``head_dim``. The
  builder's fields are sized by the whole projection, so each is tiled across
  its heads -- 32 for the query, 8 for the key. Tiling is correct only because
  Qwen3 shares one norm across heads; a per-head norm would need the builder to
  grow a head axis.
* The head is untied. ``lm_head.weight`` is a separate tensor, so ``w_out``
  comes from it rather than from the embedding.

The dtype follows the requested precision rather than widening everything to
float32. That is not a micro-optimisation: this stack is 36 layers of roughly
772 MB each at float32, plus 3.3 GB of embedding and as much again for the
head. Widening costs about 35 GB before TensorRT copies the constants into the
network, and the build container is capped at 125 GB -- a float32 build is
killed by the OOM reaper partway through.
"""

from __future__ import annotations

from typing import Any

from .checkpoint_mapper import WeightDict, _target_np_dtype
from .language_model import (
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    NUM_KEY_VALUE_HEADS,
)

#: Prefix the published checkpoint stores the decoder under.
CHECKPOINT_PREFIX = "model"

#: What the two per-head norms are tiled up to.
QUERY_WIDTH = NUM_ATTENTION_HEADS * HEAD_DIM
KEY_VALUE_WIDTH = NUM_KEY_VALUE_HEADS * HEAD_DIM


def _t(tensors: dict, key: str, precision: str = "fp32"):
    """Return ``tensors[key]`` transposed from [out, in] to [in, out]."""

    import numpy as np

    array = np.asarray(tensors[key])
    if array.ndim != 2:
        raise ValueError(f"{key} has rank {array.ndim}, expected 2")
    return np.ascontiguousarray(array.T, dtype=_target_np_dtype(precision))


def _tiled_norm(tensors: dict, key: str, width: int, precision: str = "fp32"):
    """Return a per-head norm repeated across the heads of a projection."""

    import numpy as np

    array = np.asarray(tensors[key], dtype=np.float32).reshape(-1)
    if array.size != HEAD_DIM:
        raise ValueError(
            f"{key} has {array.size} elements, expected the head dim {HEAD_DIM}"
        )
    if width % HEAD_DIM:
        raise ValueError(f"width {width} is not a multiple of the head dim {HEAD_DIM}")
    return np.ascontiguousarray(np.tile(array, width // HEAD_DIM),
                                dtype=_target_np_dtype(precision))


def build_weight_dict(tensors: dict, *, num_layers: int = NUM_HIDDEN_LAYERS,
                      precision: str = "fp32") -> Any:
    """Return the decoder WeightDict for one language-model checkpoint."""

    import numpy as np

    dtype = _target_np_dtype(precision)

    weights = WeightDict()
    prefix = CHECKPOINT_PREFIX

    embedding = np.asarray(tensors[f"{prefix}.embed_tokens.weight"])
    if embedding.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"embedding is {embedding.shape}, expected a width of {HIDDEN_SIZE}"
        )
    weights["embedding"] = np.ascontiguousarray(embedding, dtype=dtype)

    for layer in range(num_layers):
        block = f"{prefix}.layers.{layer}"
        out = f"layer.{layer}"
        weights[f"{out}.input_norm"] = np.asarray(
            tensors[f"{block}.input_layernorm.weight"], dtype=dtype
        )
        weights[f"{out}.w_q"] = _t(tensors, f"{block}.self_attn.q_proj.weight", precision)
        weights[f"{out}.w_k"] = _t(tensors, f"{block}.self_attn.k_proj.weight", precision)
        weights[f"{out}.w_v"] = _t(tensors, f"{block}.self_attn.v_proj.weight", precision)
        weights[f"{out}.q_norm"] = _tiled_norm(
            tensors, f"{block}.self_attn.q_norm.weight", QUERY_WIDTH, precision
        )
        weights[f"{out}.k_norm"] = _tiled_norm(
            tensors, f"{block}.self_attn.k_norm.weight", KEY_VALUE_WIDTH, precision
        )
        weights[f"{out}.w_o"] = _t(tensors, f"{block}.self_attn.o_proj.weight", precision)
        weights[f"{out}.post_attn_norm"] = np.asarray(
            tensors[f"{block}.post_attention_layernorm.weight"], dtype=dtype
        )
        weights[f"{out}.w_gate"] = _t(tensors, f"{block}.mlp.gate_proj.weight", precision)
        weights[f"{out}.w_up"] = _t(tensors, f"{block}.mlp.up_proj.weight", precision)
        weights[f"{out}.w_down"] = _t(tensors, f"{block}.mlp.down_proj.weight", precision)

    weights["final_norm"] = np.asarray(tensors[f"{prefix}.norm.weight"], dtype=dtype)
    weights["w_out"] = _t(tensors, "lm_head.weight", precision)
    return weights
