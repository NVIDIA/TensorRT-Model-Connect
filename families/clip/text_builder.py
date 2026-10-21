# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP text tower: token ids to one projected embedding.

**Where the sequence is pooled is the whole correctness question here, and the
checkpoint's own config states it wrongly.** ``text_config.eos_token_id`` is
``2``; the tokenizer's end-of-text id is ``49407``. The pooled row is the first
``49407`` in the sequence, which for a prompt of ``n`` real tokens is row
``n - 1``. Pooling the last row instead - the obvious reading of a padded
sequence, and what a text encoder written for a diffusion model does - reads a
padding row and returns a plausible vector for the wrong thing.

The runtime passes that row index as an engine input rather than deriving it
here, because the runtime is what tokenised the prompt and so is what knows it.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "text_model"


def build_text_tower(network, input_ids, eos_index, weights, cfg, projection, *,
                     dtype=np.float32):
    """``[seq]`` token ids to an unnormalised ``[1, projection_dim]`` embedding."""
    width = cfg["hidden_size"]
    seq = int(input_ids.shape[0])

    table = g.add_constant(network, (cfg["vocab_size"], width),
                           weights[f"{PREFIX}.embeddings.token_embedding.weight"], dtype=dtype)
    hidden = network.add_gather(table, input_ids, axis=0).get_output(0)
    positions = g.add_constant(network, (seq, width),
                               weights[f"{PREFIX}.embeddings.position_embedding.weight"][:seq],
                               dtype=dtype)
    hidden = g.add_sum(network, hidden, positions)

    mask = encoder_builder.causal_mask(network, seq, dtype=dtype)
    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, mask=mask,
                                          dtype=dtype)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.final_layer_norm.weight"],
                              weights[f"{PREFIX}.final_layer_norm.bias"], cfg["eps"])

    pooled = network.add_gather(hidden, eos_index, axis=0).get_output(0)
    reshape = network.add_shuffle(pooled)
    reshape.reshape_dims = (1, width)
    return g.add_linear(network, reshape.get_output(0), projection, dtype=dtype)
