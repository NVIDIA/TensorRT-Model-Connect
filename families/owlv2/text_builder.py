# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OWLv2 text tower: one query phrase to one query embedding.

The pooled row is the end-of-text token, located by the index the runtime
passes in. The reference finds it with ``argmax(input_ids)``, which works only
because the pad id is 0 and the end marker is the largest id in the vocabulary;
passing the index as an engine input says the same thing without relying on
that coincidence holding for every checkpoint.

Measured: supplying the padding attention mask as well changes the pooled row
by 0.0. Attention is causal, so no row at or before the end marker can read a
pad token. The causal mask itself is not optional.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder
from . import graph as g

PREFIX = "owlv2.text_model"


def build_query(network, input_ids, eos_index, weights, cfg, *, dtype=np.float32):
    """``input_ids [seq]`` and ``eos_index [1]`` to one query embedding ``[1, dim]``."""
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
    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg,
                                          mask=mask, dtype=dtype)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.final_layer_norm.weight"],
                              weights[f"{PREFIX}.final_layer_norm.bias"], cfg["eps"])

    pooled = network.add_gather(hidden, eos_index, axis=0).get_output(0)
    reshape = network.add_shuffle(pooled)
    reshape.reshape_dims = (1, width)
    projected = g.add_linear(network, reshape.get_output(0),
                             weights["owlv2.text_projection.weight"], dtype=dtype)
    # The class head compares normalised vectors, and its epsilon sits outside
    # the square root. Doing it here keeps the runtime a dot product.
    return g.add_l2_normalise_eps(network, projected)
