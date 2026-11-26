# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TIPSv2 text tower: padded token ids to one pooled embedding.

The engine takes **only** the token ids and derives the padding mask from them, exactly as the
reference's tokenizer does (``is_padding = (tokens == 0)``). The padding id is a SentencePiece
control piece, so no prompt can produce it and there is no second input that could disagree with
the first.

The mask is used in three places:

* the attention adds it to the scores, so a padding row is never attended **as a key**;
* the pooling averages over the non-padding rows only - they are dropped from the sum and from
  the count;
* the MLP masks both its hidden and its output rows with it.

**The first two carry the property; the third does not.** Because of them the embedding of a
prompt that fits does not depend on how much padding follows it, and `tests/test_model.py` pins
exactly that: the builder runs over two window lengths and is compared, then compared again with
each of those two disabled, where the embedding moves by 0.26 and 0.59 instead of 2.7e-07.

The MLP masking is kept for a different reason. Nothing downstream reads a padded row - the
attention will not attend it and the pooling discards it - so zeroing it cannot move the
embedding, and the test says so. It is here because it is what the reference computes
(`x * mlp_mask`), and because a selection is the one form of it that could not turn a non-finite
padded row into a NaN if a later change ever made those rows observable.

**Precision.** The pooled average is the one reduction that cannot be taken in the network's
dtype: the reference divides by ``count + 1e-8``, and 1e-8 is below FP16's smallest normal, so in
FP16 the epsilon would vanish and an all-padding prompt would divide zero by zero. The sum, the
count, the epsilon and the division are FP32 here whatever the engine's dtype.

The two MLP maskings **select** their rows where the reference multiplies by a 0/1 factor. The
two agree for every finite value; they differ when an FP16 projection overflows, where the
product is a NaN that spreads through the residual stream and the selection is a zero.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

PREFIX = "text_encoder"


def _packed_qkv(weights, layer: str, width: int):
    """Split ``nn.MultiheadAttention``'s ``in_proj`` into q, k and v, in that order."""
    packed_weight = np.asarray(weights[f"{layer}.attn.in_proj_weight"])
    packed_bias = np.asarray(weights[f"{layer}.attn.in_proj_bias"])
    if packed_weight.shape != (3 * width, width) or packed_bias.shape != (3 * width,):
        raise ValueError(f"tipsv2 {layer}.attn.in_proj is not a packed [3 * width, width] tensor")
    return (np.split(packed_weight, 3, axis=0), np.split(packed_bias, 3, axis=0))


def _masks(network, input_ids, cfg):
    """From the ids alone: the three masks the tower needs.

    Returns ``(keep_rows[seq, 1] as BOOL, additive_mask[1, 1, 1, seq] in FP32,
    valid[seq] in FP32)``.

    ``keep_rows`` is a boolean rather than a 0/1 factor on purpose. The reference multiplies the
    MLP's rows by its mask, and a product is the one form that cannot be trusted in FP16: a
    projection that overflows to an infinity on a padding row would become a NaN against the zero
    and then spread through the residual stream into every later attention. Selecting the row
    gives the same answer for every finite value and a zero for the rest.
    """
    seq = cfg["max_text_len"]
    pad = network.add_constant((1,), np.asarray([cfg["pad_token_id"]], dtype=np.int32))
    is_padding = network.add_elementwise(input_ids, pad.get_output(0),
                                         trt.ElementWiseOperation.EQUAL).get_output(0)
    padding = network.add_cast(is_padding, trt.float32).get_output(0)
    one = g.add_constant(network, (1,), np.ones((1,), dtype=np.float32))
    valid = network.add_elementwise(one, padding, trt.ElementWiseOperation.SUB).get_output(0)

    keep = network.add_unary(is_padding, trt.UnaryOperation.NOT).get_output(0)
    rows = network.add_shuffle(keep)
    rows.reshape_dims = (seq, 1)
    keep_rows = rows.get_output(0)

    # (valid - 1) * |mask value| is 0 on a real token and the mask value on a padding one. A
    # finite number rather than -inf: a masked score multiplied by a zero weight would otherwise
    # be a NaN instead of a zero.
    shifted = network.add_elementwise(
        valid, one, trt.ElementWiseOperation.SUB).get_output(0)
    magnitude = g.add_constant(network, (1,),
                               np.full((1,), -g.mask_additive_value(np.float32), np.float32))
    additive = g.add_prod(network, shifted, magnitude)
    reshaped = network.add_shuffle(additive)
    reshaped.reshape_dims = (1, 1, 1, seq)
    return keep_rows, reshaped.get_output(0), valid


def _masked_rows(network, tensor, keep_rows, zero):
    """The reference's `x * mlp_mask`, as a selection so a non-finite row cannot spread.

    A named function rather than a closure because it is one of the three uses of the padding
    mask that `tests/test_model.py` disables one at a time: `keep_rows` also drives the pooled
    average, so a test that replaced the tensor could not tell the two apart.
    """
    return network.add_select(keep_rows, tensor, zero).get_output(0)


def build_text_tower(network, input_ids, weights, cfg, *, dtype=np.float32):
    """``input_ids[seq]`` (int32) to the unnormalised pooled embedding ``[width]``."""
    width = cfg["hidden_size"]
    seq = cfg["max_text_len"]
    keep_rows, additive_mask, valid = _masks(network, input_ids, cfg)
    zero = g.add_constant(network, (1, 1), np.zeros((1,), dtype=np.float32), dtype=dtype)

    def masked(tensor):
        return _masked_rows(network, tensor, keep_rows, zero)

    table = g.add_constant(network, (cfg["vocab_size"], width),
                           weights[f"{PREFIX}.token_embedding.weight"], dtype=dtype)
    hidden = network.add_gather(table, input_ids, 0).get_output(0)
    # scale_sqrt_depth: the reference multiplies the embedding by sqrt(width) before the position
    # signal, so the signal is small next to it rather than comparable.
    hidden = g.add_prod(network, hidden,
                        g.add_constant(network, (1, 1), np.full((1,), width ** 0.5), dtype=dtype))
    hidden = g.add_residual(network, hidden,
                            g.add_constant(network, (seq, width),
                                           g.sinusoidal_position_signal(seq, width), dtype=dtype),
                            dtype)

    for index in range(cfg["num_layers"]):
        layer = f"{PREFIX}.transformer.resblocks.{index}"
        (q_w, k_w, v_w), (q_b, k_b, v_b) = _packed_qkv(weights, layer, width)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.ln_1.weight"],
                                  weights[f"{layer}.ln_1.bias"], cfg["eps"])
        attention = g.add_attention(
            network,
            g.add_linear(network, normed, q_w, q_b, dtype=dtype),
            g.add_linear(network, normed, k_w, k_b, dtype=dtype),
            g.add_linear(network, normed, v_w, v_b, dtype=dtype),
            num_heads=cfg["num_heads"], head_dim=cfg["head_dim"],
            additive_mask=additive_mask, dtype=dtype)
        attention = g.add_linear(network, attention, weights[f"{layer}.attn.out_proj.weight"],
                                 weights[f"{layer}.attn.out_proj.bias"], dtype=dtype)
        hidden = g.add_residual(network, hidden, attention, dtype)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.ln_2.weight"],
                                  weights[f"{layer}.ln_2.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.mlp.c_fc.weight"],
                             weights[f"{layer}.mlp.c_fc.bias"], dtype=dtype)
        inner = g.add_relu(network, inner)
        # ReLU passes an overflowed projection straight through on a row that is **not** padding,
        # where the masking below does not touch it; `c_proj` would then meet it with a zero
        # weight and produce a NaN. The clamp is at this dtype's own maximum, so no value it can
        # carry is changed.
        inner = g.clamp_to_finite(network, inner, dtype)
        inner = masked(inner)  # the reference's first masking
        inner = g.add_linear(network, inner, weights[f"{layer}.mlp.c_proj.weight"],
                             weights[f"{layer}.mlp.c_proj.bias"], dtype=dtype)
        inner = masked(inner)  # and its second
        hidden = g.add_residual(network, hidden, inner, dtype)
        if int(hidden.shape[-1]) != width:
            raise ValueError("tipsv2 text block changed the sequence width")

    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.ln_final.weight"],
                              weights[f"{PREFIX}.ln_final.bias"], cfg["eps"])

    # GlobalAvgPooling over the sequence, in FP32. The reference selects rather than multiplies
    # here (`torch.where(paddings > 0, 0, inputs)`), which is also the only form that cannot turn
    # a non-finite padded row into a NaN.
    pooled, _ = g.as_float32(network, hidden)
    zeros = g.add_constant(network, (1, 1), np.zeros((1,), dtype=np.float32))
    selected = network.add_select(keep_rows, pooled, zeros).get_output(0)

    total = network.add_reduce(selected, trt.ReduceOperation.SUM, 1 << 0, True).get_output(0)
    count = network.add_reduce(valid, trt.ReduceOperation.SUM, 1 << 0, True).get_output(0)
    count = g.add_sum(network, count,
                      g.add_constant(network, (1,), np.full((1,), cfg["pooling_eps"]),
                                     dtype=np.float32))
    count_row = network.add_shuffle(count)
    count_row.reshape_dims = (1, 1)
    averaged = network.add_elementwise(total, count_row.get_output(0),
                                       trt.ElementWiseOperation.DIV).get_output(0)
    flat = network.add_shuffle(averaged)
    flat.reshape_dims = (width,)
    return flat.get_output(0)
