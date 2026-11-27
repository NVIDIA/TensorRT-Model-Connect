# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ESMFold2 building blocks shared by the family's plans.

Each block follows the released reference graph (transformers ``models/esmfold2``); see NOTICE. Token axes are
exact (no token padding), so the reference's token and pair masks are all ones and are not built. Atom axes are
padded to a multiple of the 32-atom query block, so the atom stacks carry an atom mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .graph import Net

# The atom stacks attend within |query - key| <= radius over the atom index, in blocks of this many queries.
ATOM_QUERY_BLOCK = 32
# A masked attention score: finite, so a query row whose keys are all masked stays finite (it is a padded atom and
# never feeds a valid one), and far below every real score so a masked key gets probability exactly 0 in float32.
MASKED_SCORE = -1.0e30


def transition(net: Net, x: Any, prefix: str) -> Any:
    """``x + SwiGLU(LayerNorm(x))`` (``EsmFold2Transition``, unchunked)."""

    normed = net.cast(net.layer_norm(x, f"{prefix}.norm"), x.dtype)
    return net.add(x, swiglu(net, normed, f"{prefix}.mlp"))


def swiglu(net: Net, x: Any, prefix: str) -> Any:
    """``down(up * silu(gate))`` with the fused ``gate_up_proj`` (gate first)."""

    bundled = net.linear(x, f"{prefix}.gate_up_proj")
    width = int(bundled.shape[-1]) // 2
    gate, up = net.split_last(bundled, (width, width))
    return net.linear(net.mul(up, net.silu(gate)), f"{prefix}.down_proj")


def triangle_multiplication(net: Net, z: Any, prefix: str, *, outgoing: bool) -> Any:
    """Gated triangle multiplicative update of a ``[B, N, N, C]`` pair grid (no token padding)."""

    width = int(z.shape[-1])
    normalized = net.cast(net.layer_norm(z, f"{prefix}.norm_start"), z.dtype)
    bundled = net.linear(normalized, f"{prefix}.proj_bundle")
    signal, gate_logits = net.split_last(bundled, (2 * width, 2 * width))
    routed = net.mul(signal, net.sigmoid(gate_logits))
    left, right = net.split_last(routed, (width, width))
    # outgoing: out[i, j, d] = sum_k left[i, k, d] * right[j, k, d]
    # incoming: out[i, j, d] = sum_k left[k, i, d] * right[k, j, d]
    order = (0, 3, 1, 2) if outgoing else (0, 3, 2, 1)
    contracted = net.matmul(net.transpose(left, order), net.transpose(right, order), transpose_b=True)
    contracted = net.transpose(contracted, (0, 2, 3, 1))
    mixed = net.linear(net.cast(net.layer_norm(contracted, f"{prefix}.norm_mix"), z.dtype), f"{prefix}.proj_emit")
    return net.mul(mixed, net.sigmoid(net.linear(normalized, f"{prefix}.proj_gate")))


def pair_update(net: Net, z: Any, prefix: str) -> Any:
    """``EsmFold2PairUpdateLayer``: outgoing and incoming triangle updates, then the pair transition."""

    z = net.add(z, triangle_multiplication(net, z, f"{prefix}.tri_mul_out", outgoing=True))
    z = net.add(z, triangle_multiplication(net, z, f"{prefix}.tri_mul_in", outgoing=False))
    return transition(net, z, f"{prefix}.pair_transition")


def pair_update_stack(net: Net, z: Any, prefix: str, count: int) -> Any:
    for layer in range(count):
        z = pair_update(net, z, f"{prefix}.layers.{layer}")
    return z


# ----------------------------------------------------------------------------------------------- atom stacks


@dataclass(frozen=True)
class AtomRope:
    cos: Any
    sin: Any


def atom_rope_inverse_frequencies(
    head_dim: int, spatial_pairs: int, uid_pairs: int, spatial_base: float, uid_base: float
) -> np.ndarray:
    """``EsmFold2RotaryEmbedding`` frequencies, correctly rounded to float32 (zeros for inactive pairs)."""

    spatial = 1.0 / spatial_base ** (np.arange(spatial_pairs, dtype=np.float64) / spatial_pairs)
    uid = 1.0 / uid_base ** (np.arange(uid_pairs, dtype=np.float64) / uid_pairs)
    inactive = head_dim // 2 - 3 * spatial_pairs - uid_pairs
    return np.concatenate([np.tile(spatial, 3), uid, np.zeros(inactive)]).astype(np.float32)


def atom_rope(net: Net, ref_pos: Any, ref_space_uid: Any, stack: Any) -> AtomRope:
    """3D RoPE over reference-conformer positions plus a per-token id, at the full head width (``compute``)."""

    pairs = stack.head_dim // 2
    spatial = stack.spatial_pairs_per_axis
    inv_freq = atom_rope_inverse_frequencies(
        stack.head_dim, spatial, stack.uid_pairs, stack.spatial_rope_base, stack.uid_rope_base
    )
    # positions: each axis repeated `spatial` times, then the uid repeated for the uid and inactive pairs.
    axis_columns = [net.slice_last(ref_pos, axis, 1) for axis in range(3) for _ in range(spatial)]
    uid = net.reshape(net.cast(ref_space_uid, net.f32), (0, 0, 1))
    uid_columns = [uid] * (pairs - 3 * spatial)
    positions = net.concat([*axis_columns, *uid_columns], axis=2)
    freqs = net.mul(positions, net.constant(inv_freq, shape=(1, 1, pairs)))
    emb = net.concat([freqs, freqs], axis=2)
    cos = net.cast(net.unary(emb, net.trt.UnaryOperation.COS), net.compute)
    sin = net.cast(net.unary(emb, net.trt.UnaryOperation.SIN), net.compute)
    return AtomRope(cos=cos, sin=sin)


def rotate_half(net: Net, x: Any) -> Any:
    half = int(x.shape[-1]) // 2
    first, second = net.split_last(x, (half, half))
    return net.concat([net.mul(second, net.scalar(-1.0, len(x.shape), x.dtype)), first], axis=len(x.shape) - 1)


def apply_rope(net: Net, x: Any, cos: Any, sin: Any) -> Any:
    """``x * cos + rotate_half(x) * sin`` in the dtype of ``x`` (cos / sin broadcast over heads)."""

    return net.add(net.mul(x, cos), net.mul(rotate_half(net, x), sin))


def atom_window_keys(net: Net, tensor: Any, radius: int) -> Any:
    """Gather the keys every 32-atom query block sees: ``[S, A, ...]`` -> ``[S, A / 32, 32 + 2 * radius, ...]``.

    Block ``b`` covers atoms ``[32 b - radius, 32 b + 32 + radius)``; atoms outside ``[0, A)`` read as zero.
    """

    rank = len(tensor.shape)
    int64 = net.trt.int64
    padded_atoms = net.add(net.dim(tensor, 1), net.constant([2 * radius], int64))
    size = net.shape_of([net.dim(tensor, 0), padded_atoms, *[net.dim(tensor, axis) for axis in range(2, rank)]])
    pad = net.network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
    pad.mode = net.trt.SampleMode.FILL
    pad.set_input(1, net.constant([0, -radius] + [0] * (rank - 2), int64))
    pad.set_input(2, size)
    padded = pad.get_output(0)
    window = ATOM_QUERY_BLOCK + 2 * radius
    blocks = net.network.add_elementwise(
        net.dim(tensor, 1), net.constant([ATOM_QUERY_BLOCK], int64), net.trt.ElementWiseOperation.FLOOR_DIV
    ).get_output(0)
    index = net.network.add_fill((1,), net.trt.FillOperation.LINSPACE, net.trt.int32)
    index.set_input(0, net.shape_of([blocks, window]))
    index.set_input(1, net.constant(np.asarray(0), net.trt.int32, shape=()))
    index.set_input(2, net.constant([ATOM_QUERY_BLOCK, 1], net.trt.int32))
    return net.gather(padded, index.get_output(0), 1)


def atom_window_bias(net: Net, atom_mask: Any, radius: int) -> Any:
    """Additive float32 bias ``[S, 1, A / 32, 32, 32 + 2 * radius]``: 0 where the key is a real atom within the radius.

    ``atom_mask`` is float32 ``[S, A]`` (1 = real atom).
    """

    window = ATOM_QUERY_BLOCK + 2 * radius
    query = np.arange(ATOM_QUERY_BLOCK)[:, None]
    key = np.arange(window)[None, :]
    # query atom 32 b + q, key atom 32 b - radius + w: |q + radius - w| <= radius
    band = ((key >= query) & (key <= query + 2 * radius)).astype(np.float32)
    band_tensor = net.constant(band, shape=(1, 1, 1, ATOM_QUERY_BLOCK, window))
    keys = atom_window_keys(net, net.reshape(atom_mask, (0, 0, 1)), radius)  # [S, nb, window, 1]
    valid = net.reshape(net.transpose(keys, (0, 3, 1, 2)), (0, 0, 0, 1, -1))  # [S, 1, nb, 1, window]
    allowed = net.mul(valid, band_tensor)
    return net.mul(net.sub(net.scalar(1.0, 5), allowed), net.scalar(MASKED_SCORE, 5))


def atom_attention(net: Net, x: Any, bias: Any, rope: AtomRope, prefix: str, stack: Any, radius: int) -> Any:
    """Sliding-window self-attention with 3D RoPE and a gated output (``EsmFold2AtomAttention``)."""

    heads, head_dim = stack.num_heads, stack.head_dim
    eps = 1.0e-6

    def project(name: str) -> Any:
        return net.reshape(net.linear(x, f"{prefix}.{name}"), (0, 0, heads, head_dim))  # [S, A, H, d]

    cos = net.reshape(rope.cos, (0, 0, 1, head_dim))
    sin = net.reshape(rope.sin, (0, 0, 1, head_dim))
    query = apply_rope(net, net.cast(net.rms_norm(project("q_proj"), eps=eps), x.dtype), cos, sin)
    key = apply_rope(net, net.cast(net.rms_norm(project("k_proj"), eps=eps), x.dtype), cos, sin)
    value = project("v_proj")
    # queries [S, H, nb, 32, d]; keys and values [S, H, nb, window, d]
    query = net.transpose(net.reshape(query, (0, -1, ATOM_QUERY_BLOCK, heads, head_dim)), (0, 3, 1, 2, 4))
    key = net.transpose(atom_window_keys(net, key, radius), (0, 3, 1, 2, 4))
    value = net.transpose(atom_window_keys(net, value, radius), (0, 3, 1, 2, 4))
    # Scores stay float32, as in the reference's fused attention.
    scores = net.matmul(net.cast(query, net.f32), net.cast(key, net.f32), transpose_b=True)
    scores = net.add(net.mul(scores, net.scalar(head_dim**-0.5, 5)), bias)
    probabilities = net.cast(net.softmax(scores, -1), x.dtype)
    context = net.matmul(probabilities, value)  # [S, H, nb, 32, d]
    context = net.reshape(net.transpose(context, (0, 2, 3, 1, 4)), (0, -1, heads * head_dim))
    gated = net.mul(context, net.sigmoid(net.linear(x, f"{prefix}.gate_proj")))
    return net.linear(gated, f"{prefix}.o_proj")


def atom_layer(
    net: Net, hidden: Any, conditioning: Any, bias: Any, rope: AtomRope, prefix: str, stack: Any, radius: int
) -> Any:
    """adaLN-Zero modulated window attention and SwiGLU (``EsmFold2AtomLayer``)."""

    width = stack.hidden_size
    modulation = net.linear(net.silu(conditioning), f"{prefix}.adaln_linear")
    attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = net.split_last(modulation, (width,) * 6)
    one = net.scalar(1.0, 3, hidden.dtype)
    eps = 1.0e-6

    def modulate(state: Any, scale: Any, shift: Any) -> Any:
        normed = net.cast(net.rms_norm(state, eps=eps), state.dtype)
        return net.add(net.mul(normed, net.add(one, scale)), shift)

    attended = atom_attention(net, modulate(hidden, attn_scale, attn_shift), bias, rope, f"{prefix}.self_attn",
                              stack, radius)
    hidden = net.add(hidden, net.mul(attn_gate, attended))
    mlp = swiglu(net, modulate(hidden, mlp_scale, mlp_shift), f"{prefix}.mlp")
    return net.add(hidden, net.mul(mlp_gate, mlp))


def atom_features(
    net: Net, ref_pos: Any, ref_charge: Any, atom_mask: Any, ref_element: Any, ref_name_chars: Any,
    max_atomic_number: int, char_vocab_size: int, max_chars: int,
) -> Any:
    """``[xyz, charge, mask, element one-hot, atom-name-char one-hots]`` in float32, zero at padded atoms."""

    mask = net.reshape(atom_mask, (0, 0, 1))
    element = net.mul(net.one_hot(ref_element, max_atomic_number, net.f32), mask)
    chars = net.one_hot(ref_name_chars, char_vocab_size, net.f32)  # [S, A, chars, vocab]
    chars = net.mul(net.reshape(chars, (0, 0, max_chars * char_vocab_size)), mask)
    return net.concat([ref_pos, net.reshape(ref_charge, (0, 0, 1)), mask, element, chars], axis=2)


def embed_atoms(net: Net, features: Any, prefix: str) -> Any:
    """``atom_norm(atom_linear(features))`` in ``compute``."""

    projected = net.linear(features, f"{prefix}.atom_linear")
    return net.cast(net.layer_norm(projected, f"{prefix}.atom_norm"), net.compute)


def atom_to_token_mean(net: Net, atom_values: Any, atom_to_token: Any, atom_mask: Any, tokens: Any) -> Any:
    """Mean of each token's valid atoms: ``[S, A, W]`` -> ``[S, N, W]`` (float32 accumulation, ``compute`` out).

    ``tokens`` is any tensor whose axis 1 has length N.
    """

    token_ids = net.cast(net.iota(net.dim(tokens, 1)), net.trt.int32)  # [N]
    owner = net.reshape(atom_to_token, (0, 0, 1))  # [S, A, 1]
    member = net.equal(owner, net.reshape(token_ids, (1, 1, -1)))  # [S, A, N]
    weights = net.mul(net.cast(member, net.f32), net.reshape(atom_mask, (0, 0, 1)))
    counts = net.reduce_sum(weights, 1, keep_dims=True)  # [S, 1, N]
    weights = net.div(weights, net.maximum(counts, net.scalar(1.0, 3)))
    pooled = net.matmul(net.transpose(weights, (0, 2, 1)), net.cast(atom_values, net.f32))  # [S, N, W]
    return net.cast(pooled, net.compute)


def token_to_atom(net: Net, token_values: Any, atom_to_token: Any) -> Any:
    """Each atom takes its token's row: ``[S, N, W]`` with a ``[1, A]`` map -> ``[S, A, W]``."""

    return net.gather(token_values, net.reshape(atom_to_token, (-1,)), 1)


def lm_layer_mix_weights(layer_weights: np.ndarray, compute_is_bf16: bool) -> np.ndarray:
    """``softmax(layer_weights)`` as the reference computes it (bf16 parameter and output in bf16 models)."""

    import ml_dtypes

    values = layer_weights.astype(np.float64)
    if compute_is_bf16:
        values = values.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float64)
    shifted = np.exp(values - values.max())
    weights = (shifted / shifted.sum()).astype(np.float32)
    if compute_is_bf16:
        weights = weights.astype(ml_dtypes.bfloat16).astype(np.float32)
    return weights


def esmc_inverse_frequencies(head_dim: int, theta: float) -> np.ndarray:
    exponents = np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    return (1.0 / theta**exponents).astype(np.float32)


def esmc_residual_scale(num_layers: int) -> float:
    return math.sqrt(num_layers / 36)
