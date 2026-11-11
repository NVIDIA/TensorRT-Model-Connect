# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conditioning engine: T5Gemma prompt encoder plus the ``seconds_total`` embedder.

It reproduces ``MultiConditioner`` and ``ConditionedDiffusionModelWrapper.get_conditioning_inputs``
of the reference for one prompt:

* the T5Gemma encoder runs in BF16, the dtype the checkpoint stores and the one
  the reference loads it in (``from_pretrained`` defaults to the config dtype);
  its attention is plain scaled dot-product attention, as with the reference's
  default ``sdpa`` implementation, which does not apply ``attn_logit_softcapping``;
* padded prompt positions are replaced by the learned ``padding_embedding``;
* ``seconds_total`` is clamped, normalized, expanded into exponential Fourier
  features and projected, in the model dtype.

Outputs are the DiT's cross-attention context ``[prompt | seconds]`` and its
global conditioning ``seconds``, both FP32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph import MASKED, Graph, expo_fourier_features, expo_fourier_frequencies

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
SECONDS = "seconds_total"
CONTEXT = "cross_attn_cond"
GLOBAL = "global_cond"


def text_encoder_names(text: dict) -> list[str]:
    names = ["embed_tokens.weight", "norm.weight"]
    for layer in range(text["layers"]):
        prefix = f"layers.{layer}."
        names += [prefix + name for name in (
            "pre_self_attn_layernorm.weight", "post_self_attn_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
    return names


CONDITIONER_NAMES = [
    "prompt.padding_embedding",
    "seconds_total.embedder.embedding.1.weight",
    "seconds_total.embedder.embedding.1.bias",
]


def _rotate(g: Graph, x, cos, sin, head_dim: int):
    """``x * cos + rotate_half(x) * sin`` in ``x``'s dtype, as transformers' T5Gemma does."""
    half = head_dim // 2
    lead = [int(v) for v in x.shape[:-1]]
    first = g.slice(x, (0, 0, 0, 0), (*lead, half))
    second = g.slice(x, (0, 0, 0, half), (*lead, half))
    rotated = g.concat([g.mul_scalar(second, -1.0), first], 3)
    return g.add(g.mul(x, cos), g.mul(rotated, sin))


def _encoder_layer(g: Graph, h, weights: dict, text: dict, layer: int, cos, sin, mask):
    prefix = f"layers.{layer}."
    tokens, hidden, heads, head_dim = text["max_length"], text["hidden"], text["heads"], text["head_dim"]

    def w(name):
        return weights[prefix + name]

    def split(x):
        return g.shuffle(x, reshape=(1, tokens, heads, head_dim), second=(0, 2, 1, 3))

    residual = h
    x = g.rms_norm(h, w("pre_self_attn_layernorm.weight"), text["eps"], offset=1.0)
    q = _rotate(g, split(g.linear(x, w("self_attn.q_proj.weight"))), cos, sin, head_dim)
    k = _rotate(g, split(g.linear(x, w("self_attn.k_proj.weight"))), cos, sin, head_dim)
    v = split(g.linear(x, w("self_attn.v_proj.weight")))
    context = g.attention(q, k, v, text["query_scale"], mask)
    context = g.shuffle(context, first=(0, 2, 1, 3), reshape=(1, tokens, hidden))
    x = g.linear(context, w("self_attn.o_proj.weight"))
    x = g.rms_norm(x, w("post_self_attn_layernorm.weight"), text["eps"], offset=1.0)
    h = g.add(residual, x)

    residual = h
    x = g.rms_norm(h, w("pre_feedforward_layernorm.weight"), text["eps"], offset=1.0)
    gate = g.act(g.linear(x, w("mlp.gate_proj.weight")), trt.ActivationType.GELU_TANH)
    x = g.linear(g.mul(gate, g.linear(x, w("mlp.up_proj.weight"))), w("mlp.down_proj.weight"))
    x = g.rms_norm(x, w("post_feedforward_layernorm.weight"), text["eps"], offset=1.0)
    return g.add(residual, x)


def _prompt_embeddings(g: Graph, ids, mask_int, weights: dict, text: dict):
    tokens, hidden, head_dim = text["max_length"], text["hidden"], text["head_dim"]
    table = g.const(weights["embed_tokens.weight"])
    h = g.n.add_gather(table, ids, 0).get_output(0)  # [1, S, hidden]
    h = g.mul(h, g.const(np.full((1, 1, 1), np.sqrt(hidden), dtype=np.float32)))

    inv_freq = 1.0 / (text["rope_theta"] ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32)
                                              / head_dim))
    angles = np.outer(np.arange(tokens, dtype=np.float32), inv_freq.astype(np.float32)).astype(np.float32)
    angles = np.concatenate([angles, angles], axis=-1).astype(np.float64)
    cos = g.const(np.cos(angles).astype(np.float32), shape=(1, 1, tokens, head_dim))
    sin = g.const(np.sin(angles).astype(np.float32), shape=(1, 1, tokens, head_dim))

    valid = g.cast(g.shuffle(mask_int, reshape=(1, 1, 1, tokens)), trt.float32)
    mask = g.mul_scalar(g.add_scalar(g.mul_scalar(valid, -1.0), 1.0), MASKED)  # broadcast over queries

    for layer in range(text["layers"]):
        h = _encoder_layer(g, h, weights, text, layer, cos, sin, mask)
    return g.rms_norm(h, weights["norm.weight"], text["eps"], offset=1.0)


def _apply_learned_padding(g: Graph, h, mask_int, padding: np.ndarray, model_dtype, tokens: int):
    """``where(mask, h, padding_embedding)`` in FP32, with the padding rounded to the model dtype."""
    h32 = g.cast(h, trt.float32)
    keep = g.cast(g.shuffle(mask_int, reshape=(1, tokens, 1)), trt.float32)
    pad = g.cast(g.const(padding, model_dtype, shape=(1, 1, padding.size)), trt.float32)
    drop = g.add_scalar(g.mul_scalar(keep, -1.0), 1.0)
    return g.add(g.mul(h32, keep), g.mul(pad, drop))


def _seconds_embedding(g: Graph, seconds, weights: dict, spec: dict, model_dtype):
    lo, hi = spec["min"], spec["max"]
    clamped = g.n.add_elementwise(seconds, g.scalar(lo, seconds), trt.ElementWiseOperation.MAX).get_output(0)
    clamped = g.n.add_elementwise(clamped, g.scalar(hi, seconds), trt.ElementWiseOperation.MIN).get_output(0)
    normalized = g.div(g.add_scalar(clamped, -lo), g.scalar(hi - lo, clamped))
    # NumberConditioner casts the normalized value to the embedder dtype before
    # ExpoFourierFeatures upcasts it again.
    normalized = g.cast(g.cast(normalized, model_dtype), trt.float32)
    freqs = expo_fourier_frequencies(spec["features"], spec["min_freq"], spec["max_freq"])
    features = g.cast(expo_fourier_features(g, normalized, freqs), model_dtype)
    embedding = g.linear(features, weights["seconds_total.embedder.embedding.1.weight"],
                         weights["seconds_total.embedder.embedding.1.bias"])
    return g.cast(embedding, trt.float32)  # [1, cond]


def populate(network: trt.INetworkDefinition, text_weights: dict, cond_weights: dict, cfg: dict,
             model_dtype: trt.DataType, text_dtype: trt.DataType = trt.bfloat16) -> Graph:
    """Define the conditioning network.

    ``model_dtype`` is the DiT trunk dtype; ``text_dtype`` the T5Gemma dtype (BF16 as stored;
    tests prove the graph in FP32).
    """
    text, seconds_spec = cfg["text"], cfg["seconds"]
    tokens, hidden = text["max_length"], text["hidden"]
    g = Graph(network, text_dtype)
    ids = network.add_input(INPUT_IDS, trt.int32, (1, tokens))
    mask_int = network.add_input(ATTENTION_MASK, trt.int32, (1, tokens))
    seconds = network.add_input(SECONDS, trt.float32, (1, 1))

    prompt = _prompt_embeddings(g, ids, mask_int, text_weights, text)
    prompt = _apply_learned_padding(g, prompt, mask_int, cond_weights["prompt.padding_embedding"],
                                    model_dtype, tokens)
    seconds_embedding = _seconds_embedding(g, seconds, cond_weights, seconds_spec, model_dtype)
    context = g.concat([prompt, g.shuffle(seconds_embedding, reshape=(1, 1, hidden))], 1)
    context.name = CONTEXT
    network.mark_output(context)
    global_cond = g.shuffle(seconds_embedding, reshape=(1, hidden))
    global_cond.name = GLOBAL
    network.mark_output(global_cond)
    return g
