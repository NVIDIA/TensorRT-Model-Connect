# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP text tower: token ids to one projected embedding.

**The pooled row is the last row of the padded window, and that row is
padding.** The reference says so in its own comment - "the model uses the last
token's hidden state, which may be padding" - and it is the opposite of CLIP,
which pools the first end-of-text token. Two consequences follow, and both are
easy to get wrong while producing a plausible vector:

* the window length is a correctness rule, not a capacity. Measured on the
  base checkpoint, running the same prompt in a 32-row window instead of the
  trained 64 moves the pooled embedding by 1.98 (cosine 0.863), because a
  different row gets pooled. The engine is therefore built at exactly the
  checkpoint's window and the runtime always pads to it.
* there is no mask of any kind. The reference's processor emits no attention
  mask at all, so the padding rows take part in attention; measured, feeding
  an all-ones mask changes the pooled embedding by exactly 0.
"""

from __future__ import annotations

import numpy as np

from . import encoder_builder, graph as g

PREFIX = "text_model"


def build_text_tower(network, input_ids, weights, cfg, *, dtype=np.float32, quant=None):
    """``[seq]`` token ids to an unnormalised ``[1, projection_size]`` embedding."""
    width = cfg["hidden_size"]
    seq = int(input_ids.shape[0])

    table = g.add_constant(network, (cfg["vocab_size"], width),
                           weights[f"{PREFIX}.embeddings.token_embedding.weight"], dtype=dtype)
    hidden = network.add_gather(table, input_ids, axis=0).get_output(0)
    positions = g.add_constant(network, (seq, width),
                               weights[f"{PREFIX}.embeddings.position_embedding.weight"][:seq],
                               dtype=dtype)
    hidden = g.add_sum(network, hidden, positions)

    hidden = encoder_builder.build_layers(network, hidden, weights, PREFIX, cfg, dtype=dtype,
                                          quant=quant)
    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.final_layer_norm.weight"],
                              weights[f"{PREFIX}.final_layer_norm.bias"], cfg["eps"])

    last = network.add_slice(hidden, start=(seq - 1, 0), shape=(1, width), stride=(1, 1))
    return g.add_linear(network, last.get_output(0), weights[f"{PREFIX}.head.weight"],
                        weights[f"{PREFIX}.head.bias"], dtype=dtype)
