# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP 2 text tower: a full window of token ids to one projected embedding.

The pooled row is the **last row of the padded window**, which is padding for any
prompt shorter than the window - the reference says so next to the slice. Two rules
follow, and both produce a plausible vector when broken:

* the window length is a correctness rule, not a capacity: the engine is built at
  exactly the checkpoint's window and the runtime always pads to it;
* there is no mask. The reference's tokenizer emits only ``input_ids`` for this
  checkpoint, so the padding rows take part in attention.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "text_model"


def build_text_tower(network, input_ids, weights, cfg, *, dtype=np.float32):
    """``[window]`` int32 token ids to an unnormalised ``[1, projection_size]`` embedding."""
    width = cfg["hidden_size"]
    window = int(input_ids.shape[0])

    table = g.add_constant(network, (cfg["vocab_size"], width),
                           weights[f"{PREFIX}.embeddings.token_embedding.weight"], dtype=dtype)
    hidden = network.add_gather(table, input_ids, axis=0).get_output(0)
    positions = g.add_constant(network, (window, width),
                               weights[f"{PREFIX}.embeddings.position_embedding.weight"][:window],
                               dtype=dtype)
    hidden = g.add_sum(network, hidden, positions)

    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, dtype=dtype)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.final_layer_norm.weight"],
                              weights[f"{PREFIX}.final_layer_norm.bias"], cfg["eps"])

    last = network.add_slice(hidden, start=(window - 1, 0), shape=(1, width), stride=(1, 1))
    return g.add_linear(network, last.get_output(0), weights[f"{PREFIX}.head.weight"],
                        weights[f"{PREFIX}.head.bias"], dtype=dtype)
