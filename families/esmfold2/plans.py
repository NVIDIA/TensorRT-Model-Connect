# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TensorRT networks of one ESMFold2 bundle.

Tokens are the protein's residues (``N``, exact, no padding); atoms are its heavy atoms padded to a multiple of 32
(``A``); the diffusion runs ``S`` samples at once. The runtime drives the plans in this order: ``esmc`` and
``embed`` once, ``msa`` once (ESMFold2 only), ``loop`` once per trunk iteration, ``decode`` once, ``conditioning``
once, ``denoise`` once per sampling step, ``confidence`` once per sample.
"""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np

from . import layers
from .config import EsmFold2Config
from .graph import Net


def _inputs(net: Net, specs: list[tuple[str, Any, tuple[int, ...]]]) -> dict[str, Any]:
    return {name: net.network.add_input(name, dtype, shape) for name, dtype, shape in specs}


def _atom_specs(trt: Any) -> list[tuple[str, Any, tuple[int, ...]]]:
    return [
        ("ref_pos", trt.float32, (1, -1, 3)),
        ("ref_charge", trt.float32, (1, -1)),
        ("atom_mask", trt.float32, (1, -1)),
        ("ref_element", trt.int32, (1, -1)),
        ("ref_name_chars", trt.int32, (1, -1, 4)),
        ("ref_space_uid", trt.int32, (1, -1)),
        ("atom_to_token", trt.int32, (1, -1)),
    ]


def _div_scalar(net: Net, x: Any, divisor: float) -> Any:
    """``x / divisor`` evaluated in float32 and rounded once (torch's scalar arithmetic on bf16)."""

    return net.cast(net.div(net.cast(x, net.f32), net.scalar(divisor, len(x.shape))), x.dtype)


def _mul_scalar(net: Net, x: Any, factor: float) -> Any:
    return net.cast(net.mul(net.cast(x, net.f32), net.scalar(factor, len(x.shape))), x.dtype)


# --------------------------------------------------------------------------------------------- ESMC


def define_esmc(net: Net, config: EsmFold2Config, layer_mix: np.ndarray) -> None:
    """ESMC-6B over ``[BOS] sequence [EOS]`` and the LM encoder's softmax layer mix of every hidden state.

    Output ``lm_single`` ``[1, T, pair]``: ``sum_i w_i * pair_proj(LayerNorm(h_i))`` over the embeddings, the 79
    inner layer outputs and the final-norm output (the reference's ``output_hidden_states``).
    """

    trt = net.trt
    e = config.esmc
    ids = _inputs(net, [("input_ids", trt.int32, (1, -1))])["input_ids"]
    table = net.constant(net.weight("esmc.embed_tokens.weight"), net.compute)
    hidden = net.gather(table, ids, 0)  # [1, T, H]

    positions = net.cast(net.iota(net.dim(ids, 1)), net.f32)  # [T]
    inv_freq = layers.esmc_inverse_frequencies(e.head_dim, e.rope_theta)
    freqs = net.mul(net.reshape(positions, (-1, 1)), net.constant(inv_freq, shape=(1, e.head_dim // 2)))
    emb = net.concat([freqs, freqs], axis=1)  # [T, d]
    cos = net.reshape(net.cast(net.unary(emb, trt.UnaryOperation.COS), net.compute), (1, -1, 1, e.head_dim))
    sin = net.reshape(net.cast(net.unary(emb, trt.UnaryOperation.SIN), net.compute), (1, -1, 1, e.head_dim))
    scale = layers.esmc_residual_scale(e.num_layers)
    mix = None

    def mix_in(state: Any, weight: float, accumulated: Any) -> Any:
        normed = net.cast(net.layer_norm(state, "language_model.pair_input_norm"), net.compute)
        projected = net.cast(net.linear(normed, "language_model.pair_proj"), net.f32)
        term = net.mul(projected, net.scalar(float(weight), 3))
        return term if accumulated is None else net.add(accumulated, term)

    for layer in range(e.num_layers):
        mix = mix_in(hidden, layer_mix[layer], mix)
        prefix = f"esmc.layers.{layer}"
        attn_in = net.cast(net.layer_norm(hidden, f"{prefix}.input_layernorm"), net.compute)

        def head_projection(name: str, norm: str | None) -> Any:
            projected = net.linear(attn_in, f"{prefix}.self_attn.{name}")
            if norm is not None:
                projected = net.cast(net.layer_norm(projected, f"{prefix}.self_attn.{norm}", bias=False), net.compute)
            return net.reshape(projected, (1, 0, e.num_heads, e.head_dim))  # [1, T, heads, d]

        query = layers.apply_rope(net, head_projection("q_proj", "q_norm"), cos, sin)
        key = layers.apply_rope(net, head_projection("k_proj", "k_norm"), cos, sin)
        value = head_projection("v_proj", None)
        query, key, value = (net.transpose(t, (0, 2, 1, 3)) for t in (query, key, value))  # [1, heads, T, d]
        scores = net.matmul(net.cast(query, net.f32), net.cast(key, net.f32), transpose_b=True)
        scores = net.mul(scores, net.scalar(e.head_dim**-0.5, 4))
        context = net.matmul(net.cast(net.softmax(scores, -1), net.compute), value)
        context = net.reshape(net.transpose(context, (0, 2, 1, 3)), (1, 0, e.hidden_size))
        attended = net.linear(context, f"{prefix}.self_attn.o_proj")
        hidden = net.add(hidden, _div_scalar(net, attended, scale))

        mlp_in = net.cast(net.layer_norm(hidden, f"{prefix}.post_attention_layernorm"), net.compute)
        gate = net.silu(net.linear(mlp_in, f"{prefix}.mlp.gate_proj"))
        up = net.linear(mlp_in, f"{prefix}.mlp.up_proj")
        mlp = net.linear(net.mul(gate, up), f"{prefix}.mlp.down_proj")
        hidden = net.add(hidden, _div_scalar(net, mlp, scale))

    final = net.cast(net.layer_norm(hidden, "esmc.norm", bias=False), net.compute)
    mix = mix_in(final, layer_mix[e.num_layers], mix)
    net.mark_output(net.cast(mix, net.compute), "lm_single")


# --------------------------------------------------------------------------------------------- input embedding


def _relative_position_one_hot(
    net: Net, diff: Any, bins: int, keep: Any, dtype: Any
) -> Any:
    """Clipped offset in ``2 * bins + 1`` classes, plus an out-of-context class where ``keep`` is false."""

    rank = len(diff.shape)
    shifted = net.add(diff, net.int_scalar(bins, rank))
    clipped = net.maximum(net.minimum(shifted, net.int_scalar(2 * bins, rank)), net.int_scalar(0, rank))
    binned = net.select(keep, clipped, net.int_scalar(2 * bins + 1, rank))
    return net.one_hot(binned, 2 * bins + 2, dtype)


def relative_position_encoding(net: Net, tokens: dict[str, Any], config: EsmFold2Config) -> Any:
    """``EsmFold2RelativePositionEncoding`` for ``[1, N]`` int32 token ids -> ``[1, N, N, pair]``."""

    def rows(name: str) -> Any:
        return net.reshape(tokens[name], (1, -1, 1))

    def cols(name: str) -> Any:
        return net.reshape(tokens[name], (1, 1, -1))

    same_chain = net.equal(rows("asym_id"), cols("asym_id"))
    same_residue = net.equal(rows("residue_index"), cols("residue_index"))
    same_entity = net.equal(rows("entity_id"), cols("entity_id"))
    dtype = net.compute
    residue = _relative_position_one_hot(
        net, net.sub(rows("residue_index"), cols("residue_index")), config.residue_bins, same_chain, dtype
    )
    token = _relative_position_one_hot(
        net, net.sub(rows("token_index"), cols("token_index")), config.residue_bins,
        net.logical_and(same_chain, same_residue), dtype,
    )
    chain = _relative_position_one_hot(
        net, net.sub(rows("sym_id"), cols("sym_id")), config.chain_bins, net.logical_not(same_chain), dtype
    )
    entity = net.cast(net.cast(net.reshape(same_entity, (1, 0, 0, 1)), net.f32), dtype)
    features = net.concat([residue, token, entity, chain], axis=3)
    return net.linear(features, "input_embedder.rel_pos.embed")


def _single_to_pair(net: Net, single: Any, prefix: str) -> Any:
    """Outer product and difference of the down-projected singles, then a two-layer GELU MLP."""

    x = net.linear(single, f"{prefix}.downproject", bias=True)
    rows = net.reshape(x, (1, -1, 1, int(x.shape[-1])))
    cols = net.reshape(x, (1, 1, -1, int(x.shape[-1])))
    pair = net.concat([net.mul(rows, cols), net.sub(rows, cols)], axis=3)
    hidden = net.gelu(net.linear(pair, f"{prefix}.output_fc1", bias=True))
    return net.linear(hidden, f"{prefix}.output_fc2", bias=True)


def define_embed(net: Net, config: EsmFold2Config) -> None:
    """Atom encoder, single inputs, the initial pair, the relative-position encoding and the LM pair."""

    trt = net.trt
    tokens = _inputs(
        net,
        [
            ("lm_single", net.compute, (1, -1, config.pair_size)),
            ("res_type", trt.int32, (1, -1)),
            ("residue_index", trt.int32, (1, -1)),
            ("token_index", trt.int32, (1, -1)),
            ("asym_id", trt.int32, (1, -1)),
            ("sym_id", trt.int32, (1, -1)),
            ("entity_id", trt.int32, (1, -1)),
        ],
    )
    atoms = _inputs(net, _atom_specs(trt))
    stack, radius = config.trunk_atoms, config.atom_window_radius
    features = layers.atom_features(
        net, atoms["ref_pos"], atoms["ref_charge"], atoms["atom_mask"], atoms["ref_element"],
        atoms["ref_name_chars"], config.max_atomic_number, config.char_vocab_size, config.max_chars,
    )
    prefix = "input_embedder.atom_encoder"
    atom_embeds = layers.embed_atoms(net, features, prefix)
    rope = layers.atom_rope(net, atoms["ref_pos"], atoms["ref_space_uid"], stack)
    bias = layers.atom_window_bias(net, atoms["atom_mask"], radius)
    hidden = atom_embeds
    for layer in range(stack.num_layers):
        hidden = layers.atom_layer(net, hidden, atom_embeds, bias, rope, f"{prefix}.layers.{layer}", stack, radius)
    token_features = net.relu(net.linear(hidden, f"{prefix}.atom_to_token_linear"))
    atom_encoding = layers.atom_to_token_mean(
        net, token_features, atoms["atom_to_token"], atoms["atom_mask"], tokens["res_type"]
    )

    res_one_hot = net.one_hot(tokens["res_type"], config.num_res_types, net.compute)
    deletion_mean = net.mul(net.slice_last(res_one_hot, 0, 1), net.scalar(0.0, 3, net.compute))
    # Query-only MSA: the profile is the residue one-hot and the deletion mean is zero.
    single_inputs = net.concat([atom_encoding, res_one_hot, res_one_hot, deletion_mean], axis=2)
    net.mark_output(single_inputs, "single_inputs")

    width = config.pair_size
    first = net.reshape(net.linear(single_inputs, "input_embedder.pair_init_1"), (1, -1, 1, width))
    second = net.reshape(net.linear(single_inputs, "input_embedder.pair_init_2"), (1, 1, -1, width))
    relpos = relative_position_encoding(net, tokens, config)
    # The token-bond encoding is a bias-free projection of an all-zero bond map for unbonded proteins.
    initial = net.add(net.add(first, second), relpos)
    net.mark_output(initial, "initial_pair")
    net.mark_output(relpos, "relpos")

    lm_pair = _single_to_pair(net, tokens["lm_single"], "language_model.single_to_pair")
    net.mark_output(net.cast(net.layer_norm(lm_pair, "language_model.pair_output_norm"), net.compute), "lm_pair")


# --------------------------------------------------------------------------------------------- MSA encoder


def _outer_product_mean_single_row(net: Net, msa: Any, prefix: str, config: EsmFold2Config) -> Any:
    """Outer-product mean over a one-row MSA: ``left[i] (x) right[j]`` projected to the pair width."""

    normed = net.cast(net.layer_norm(msa, f"{prefix}.norm"), msa.dtype)
    projected = net.linear(normed, f"{prefix}.input_proj")  # [1, N, 1, 2d]
    d = config.msa_outer_size
    left, right = net.split_last(projected, (d, d))
    left = net.reshape(left, (1, -1, 1, d, 1))
    right = net.reshape(right, (1, 1, -1, 1, d))
    outer = net.reshape(net.mul(left, right), (1, 0, 0, d * d))
    # One valid row: the mean divides by 1.
    return net.linear(outer, f"{prefix}.output_proj", bias=True)


def _pair_weighted_averaging_single_row(net: Net, msa: Any, pair: Any, prefix: str, config: EsmFold2Config) -> Any:
    heads, head_dim = config.msa_heads, config.msa_head_dim
    normed = net.cast(net.layer_norm(msa, f"{prefix}.norm_single"), msa.dtype)  # [1, N, 1, m]
    bias = net.linear(net.cast(net.layer_norm(pair, f"{prefix}.bias_norm"), pair.dtype), f"{prefix}.bias_proj")
    weights = net.cast(net.softmax(bias, 2), pair.dtype)  # over j: [1, N, N, heads]
    values = net.reshape(net.linear(normed, f"{prefix}.v_proj"), (1, -1, heads, head_dim))  # [1, N(j), h, d]
    gate = net.sigmoid(net.reshape(net.linear(normed, f"{prefix}.gate_proj"), (1, -1, heads, head_dim)))
    attended = net.matmul(net.transpose(weights, (0, 3, 1, 2)), net.transpose(values, (0, 2, 1, 3)))  # [1,h,N,d]
    attended = net.transpose(attended, (0, 2, 1, 3))  # [1, N, h, d]
    out = net.reshape(net.mul(gate, attended), (1, -1, 1, heads * head_dim))
    return net.linear(out, f"{prefix}.o_proj")


def define_msa(net: Net, config: EsmFold2Config) -> None:
    """The MSA encoder over the query-only MSA; loop-invariant, so the runtime runs it once."""

    trt = net.trt
    inputs = _inputs(
        net,
        [
            ("initial_pair", net.compute, (1, -1, -1, config.pair_size)),
            ("single_inputs", net.compute, (1, -1, config.single_inputs_size)),
            ("res_type", trt.int32, (1, -1)),
        ],
    )
    one_hot = net.one_hot(net.reshape(inputs["res_type"], (1, -1, 1)), config.num_res_types, net.compute)
    deletions = net.mul(net.slice_last(one_hot, 0, 2), net.scalar(0.0, 4, net.compute))
    features = net.concat([one_hot, deletions], axis=3)  # [1, N, 1, res + 2]
    msa = net.add(
        net.linear(features, "msa_encoder.embed"),
        net.reshape(net.linear(inputs["single_inputs"], "msa_encoder.project_inputs"), (1, -1, 1, config.msa_hidden_size)),
    )
    pair = inputs["initial_pair"]
    for layer in range(config.msa_layers):
        prefix = f"msa_encoder.layers.{layer}"
        pair = net.add(pair, _outer_product_mean_single_row(net, msa, f"{prefix}.outer_product_mean", config))
        if layer != config.msa_layers - 1:
            update = _pair_weighted_averaging_single_row(
                net, msa, pair, f"{prefix}.msa_pair_weighted_averaging", config
            )
            msa = layers.transition(net, net.add(msa, update), f"{prefix}.msa_transition")
        pair = layers.pair_update(net, pair, prefix)
    net.mark_output(pair, "msa_pair")


# --------------------------------------------------------------------------------------------- trunk


def parcae_discretization(weights: dict[str, np.ndarray], compute_is_bf16: bool) -> tuple[np.ndarray, np.ndarray]:
    """``EsmFold2Parcae.discretize`` with the reference's per-operation rounding (bf16 parameters in bf16 models)."""

    def rounded(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if compute_is_bf16:
            return values.astype(ml_dtypes.bfloat16).astype(np.float32)
        return values

    log_delta = rounded(weights["parcae.log_delta"])
    log_decay = rounded(weights["parcae.log_state_decay"])
    continuous = rounded(weights["parcae.input_matrix_continuous"])
    # softplus(x) = log1p(exp(x)), and x itself above torch's threshold of 20.
    softplus = np.where(log_delta > 20.0, log_delta, np.log1p(np.exp(np.minimum(log_delta, 20.0))))
    delta = rounded(softplus)
    decay = rounded(np.exp(rounded(-delta * rounded(np.exp(log_decay)))))
    matrix = rounded(delta[:, None] * continuous)
    return decay, matrix


def define_loop(net: Net, config: EsmFold2Config, decay: np.ndarray, matrix: np.ndarray) -> None:
    """One trunk iteration: LM pair dropout and refinement, the parcae recurrence, the folding trunk."""

    trt = net.trt
    pair_shape = (1, -1, -1, config.pair_size)
    inputs = _inputs(
        net,
        [
            ("state", net.compute, pair_shape),
            ("base_pair", net.compute, pair_shape),
            ("lm_pair", net.compute, pair_shape),
            ("lm_keep", trt.bool, pair_shape),
        ],
    )
    keep = net.cast(inputs["lm_keep"], net.f32)
    scale = net.scalar(1.0 / (1.0 - config.lm_dropout), 4)
    dropped = net.mul(net.mul(net.cast(inputs["lm_pair"], net.f32), keep), scale)
    refined = layers.pair_update_stack(net, net.cast(dropped, net.compute), "lm_encoder", config.lm_layers)
    injected = net.add(inputs["base_pair"], refined)
    normed = net.cast(net.layer_norm(injected, "parcae.input_norm"), net.compute)
    width = config.pair_size
    decayed = net.mul(net.constant(decay, net.compute, shape=(1, 1, 1, width)), inputs["state"])
    state = net.add(decayed, net.linear(normed, weight=matrix))
    state = layers.pair_update_stack(net, state, "folding_trunk", config.trunk_layers)
    net.mark_output(state, "next_state")


def define_decode(net: Net, config: EsmFold2Config) -> None:
    """The parcae coda: ``out_proj`` and the output stack; the trunk pair leaves in float32."""

    state = _inputs(net, [("state", net.compute, (1, -1, -1, config.pair_size))])["state"]
    pair = layers.pair_update_stack(net, net.linear(state, "parcae.out_proj"), "parcae.output_stack", config.coda_layers)
    net.mark_output(pair, "pair_trunk", net.f32)


# --------------------------------------------------------------------------------------------- diffusion


def define_conditioning(net: Net, config: EsmFold2Config) -> None:
    """Noise-independent denoiser conditioning, built once per fold."""

    trt = net.trt
    inputs = _inputs(
        net,
        [
            ("pair_trunk", trt.float32, (1, -1, -1, config.pair_size)),
            ("relpos", net.compute, (1, -1, -1, config.pair_size)),
            ("single_inputs", net.compute, (1, -1, config.single_inputs_size)),
        ],
    )
    atoms = _inputs(net, [spec for spec in _atom_specs(trt) if spec[0] != "atom_to_token"])
    cond = "structure_head.conditioning"
    pair = net.concat([inputs["pair_trunk"], net.cast(inputs["relpos"], net.f32)], axis=3)
    pair = net.linear(net.cast(net.layer_norm(pair, f"{cond}.pair_input_norm"), net.compute), f"{cond}.pair_proj")
    pair = layers.transition(net, pair, f"{cond}.pair_transition_0")
    pair = layers.transition(net, pair, f"{cond}.pair_transition_1")
    biases = []
    for layer in range(config.diffusion_layers):
        prefix = f"structure_head.token_transformer.layers.{layer}"
        normed = net.cast(net.layer_norm(pair, f"{prefix}.pair_norm"), net.compute)
        biases.append(net.transpose(net.linear(normed, f"{prefix}.pair_bias_proj"), (0, 3, 1, 2)))
    net.mark_output(net.concat(biases, axis=0), "token_bias")  # [layers, heads, N, N]

    single = net.cast(net.layer_norm(inputs["single_inputs"], f"{cond}.single_input_norm"), net.compute)
    net.mark_output(net.linear(single, f"{cond}.single_proj"), "cond_single")

    stack = config.diffusion_atoms
    features = layers.atom_features(
        net, atoms["ref_pos"], atoms["ref_charge"], atoms["atom_mask"], atoms["ref_element"],
        atoms["ref_name_chars"], config.max_atomic_number, config.char_vocab_size, config.max_chars,
    )
    net.mark_output(layers.embed_atoms(net, features, "structure_head.atom_encoder"), "atom_embeds")
    rope = layers.atom_rope(net, atoms["ref_pos"], atoms["ref_space_uid"], stack)
    net.mark_output(rope.cos, "rope_cos")
    net.mark_output(rope.sin, "rope_sin")


def _adaptive_layer_norm(net: Net, hidden: Any, single: Any, prefix: str) -> Any:
    """adaLN-Zero: ``sigmoid(gate(c)) * LayerNorm(h) + shift(c)`` in float32, rounded once to ``compute``."""

    normed_hidden = net.layer_norm(hidden, None)
    normed_single = net.cast(net.layer_norm(single, f"{prefix}.cond_norm", bias=False), net.compute)
    gate = net.cast(net.sigmoid(net.linear(normed_single, f"{prefix}.gate_proj", bias=True)), net.f32)
    shift = net.cast(net.linear(normed_single, f"{prefix}.shift_proj"), net.f32)
    return net.cast(net.add(net.mul(gate, normed_hidden), shift), hidden.dtype)


def _token_attention(net: Net, x: Any, bias: Any, prefix: str, config: EsmFold2Config) -> Any:
    heads, head_dim = config.diffusion_heads, config.diffusion_head_dim

    def project(name: str, with_bias: bool = False) -> Any:
        projected = net.linear(x, f"{prefix}.{name}", bias=with_bias)
        return net.transpose(net.reshape(projected, (0, 0, heads, head_dim)), (0, 2, 1, 3))  # [S, h, N, d]

    query, key, value = project("q_proj", True), project("k_proj"), project("v_proj")
    scores = net.matmul(net.cast(query, net.f32), net.cast(key, net.f32), transpose_b=True)
    scores = net.add(net.mul(scores, net.scalar(head_dim**-0.5, 4)), net.cast(bias, net.f32))
    context = net.matmul(net.cast(net.softmax(scores, -1), x.dtype), value)
    context = net.reshape(net.transpose(context, (0, 2, 1, 3)), (0, 0, heads * head_dim))
    gated = net.mul(context, net.sigmoid(net.linear(x, f"{prefix}.gate_proj")))
    return net.linear(gated, f"{prefix}.o_proj")


def define_denoise(net: Net, config: EsmFold2Config) -> None:
    """One denoising call of ``S`` samples at the same noise level (``EsmFold2DiffusionModule.forward``)."""

    trt = net.trt
    stack, radius = config.diffusion_atoms, config.atom_window_radius
    inputs = _inputs(
        net,
        [
            ("noisy_coords", trt.float32, (-1, -1, 3)),
            ("noise_level", trt.float32, (-1, 1)),
            ("token_bias", net.compute, (config.diffusion_layers, config.diffusion_heads, -1, -1)),
            ("cond_single", net.compute, (1, -1, config.diffusion_hidden_size)),
            ("atom_embeds", net.compute, (1, -1, stack.hidden_size)),
            ("rope_cos", net.compute, (1, -1, stack.head_dim)),
            ("rope_sin", net.compute, (1, -1, stack.head_dim)),
            ("atom_mask", trt.float32, (1, -1)),
            ("atom_to_token", trt.int32, (1, -1)),
        ],
    )
    sigma = inputs["noise_level"]  # [S, 1] float32
    sigma_data = config.sigma_data
    cond = "structure_head.conditioning"

    # Step 1: noise-dependent single conditioning (Fourier embedding in float32).
    ratio = net.maximum(net.div(sigma, net.scalar(sigma_data, 2)), net.scalar(1.0e-20, 2))
    log_noise = net.mul(net.unary(ratio, trt.UnaryOperation.LOG), net.scalar(0.25, 2))
    frequencies = net.constant(net.weight(f"{cond}.fourier.frequencies"), shape=(1, config.fourier_dim))
    phases = net.constant(net.weight(f"{cond}.fourier.phases"), shape=(1, config.fourier_dim))
    angles = net.mul(net.add(net.mul(log_noise, frequencies), phases), net.scalar(2.0 * np.pi, 2))
    fourier = net.unary(angles, trt.UnaryOperation.COS)
    noise_embeds = net.linear(net.cast(net.layer_norm(fourier, f"{cond}.noise_norm"), net.compute), f"{cond}.noise_proj")
    single = net.add(inputs["cond_single"], net.reshape(noise_embeds, (0, 1, config.diffusion_hidden_size)))
    single = layers.transition(net, single, f"{cond}.single_transition_0")
    single = layers.transition(net, single, f"{cond}.single_transition_1")  # [S, N, h]

    # Steps 2-3: normalized coordinates into the atom encoder.
    sigma_rows = net.reshape(sigma, (0, 1, 1))
    variance = net.add(net.mul(sigma_rows, sigma_rows), net.scalar(sigma_data * sigma_data, 3))
    root = net.unary(variance, trt.UnaryOperation.SQRT)
    normalized = net.div(inputs["noisy_coords"], root)
    coord_input = net.concat([normalized, net.mul(normalized, net.scalar(0.0, 3))], axis=2)
    coord_embeds = net.linear(net.cast(coord_input, net.compute), "structure_head.coords_linear")
    atom_embeds = inputs["atom_embeds"]
    rope = layers.AtomRope(cos=inputs["rope_cos"], sin=inputs["rope_sin"])
    window_bias = layers.atom_window_bias(net, inputs["atom_mask"], radius)
    hidden = net.add(atom_embeds, coord_embeds)
    encoder = "structure_head.atom_encoder"
    for layer in range(stack.num_layers):
        hidden = layers.atom_layer(net, hidden, atom_embeds, window_bias, rope, f"{encoder}.layers.{layer}", stack, radius)
    token_features = net.relu(net.linear(hidden, f"{encoder}.atom_to_token_linear"))
    token_hidden = layers.atom_to_token_mean(
        net, token_features, inputs["atom_to_token"], inputs["atom_mask"], inputs["cond_single"]
    )

    # Step 4: conditioned single representation.
    step_single = net.cast(net.layer_norm(single, "structure_head.single_step_norm"), net.compute)
    token_hidden = net.add(token_hidden, net.linear(step_single, "structure_head.single_to_token"))

    # Step 5: token transformer with one pair bias per layer.
    for layer in range(config.diffusion_layers):
        prefix = f"structure_head.token_transformer.layers.{layer}"
        layer_bias = _layer_slice(net, inputs["token_bias"], layer)
        attn_in = _adaptive_layer_norm(net, token_hidden, single, f"{prefix}.input_layernorm")
        attended = _token_attention(net, attn_in, layer_bias, f"{prefix}.self_attn", config)
        token_hidden = net.add(
            token_hidden, net.mul(net.sigmoid(net.linear(single, f"{prefix}.attn_gate", bias=True)), attended)
        )
        mlp_in = _adaptive_layer_norm(net, token_hidden, single, f"{prefix}.post_attention_layernorm")
        mlp = layers.swiglu(net, mlp_in, f"{prefix}.mlp")
        token_hidden = net.add(
            token_hidden, net.mul(net.sigmoid(net.linear(single, f"{prefix}.mlp_gate", bias=True)), mlp)
        )

    # Steps 6-7: token norm and the atom decoder.
    token_hidden = net.cast(net.layer_norm(token_hidden, "structure_head.token_norm"), net.compute)
    decoder = "structure_head.atom_decoder"
    to_atoms = layers.token_to_atom(
        net, net.linear(token_hidden, f"{decoder}.token_to_atom_linear"), inputs["atom_to_token"]
    )
    hidden = net.add(hidden, to_atoms)
    for layer in range(stack.num_layers):
        hidden = layers.atom_layer(net, hidden, atom_embeds, window_bias, rope, f"{decoder}.layers.{layer}", stack, radius)
    update = net.linear(net.cast(net.layer_norm(hidden, f"{decoder}.norm"), net.compute), f"{decoder}.output_linear")

    # Step 8: EDM preconditioning, float32.
    data_variance = net.scalar(sigma_data * sigma_data, 3)
    skip = net.div(data_variance, variance)
    out = net.div(net.mul(net.scalar(sigma_data, 3), sigma_rows), root)
    denoised = net.add(net.mul(skip, inputs["noisy_coords"]), net.mul(out, net.cast(update, net.f32)))
    net.mark_output(denoised, "denoised")


def _layer_slice(net: Net, stacked: Any, layer: int) -> Any:
    """``stacked[layer : layer + 1]`` of a ``[layers, heads, N, N]`` tensor."""

    rank = len(stacked.shape)
    slicer = net.network.add_slice(stacked, (0,) * rank, (1,) * rank, (1,) * rank)
    slicer.set_input(1, net.constant([layer, 0, 0, 0], net.trt.int64))
    slicer.set_input(2, net.shape_of([1, net.dim(stacked, 1), net.dim(stacked, 2), net.dim(stacked, 3)]))
    return slicer.get_output(0)


# --------------------------------------------------------------------------------------------- confidence


def define_confidence(net: Net, config: EsmFold2Config) -> None:
    """Confidence head of one sample: per-atom expected pLDDT and the per-row pTM averages."""

    trt = net.trt
    c = config.pair_size
    inputs = _inputs(
        net,
        [
            ("single_inputs", net.compute, (1, -1, config.single_inputs_size)),
            ("pair_trunk", trt.float32, (1, -1, -1, c)),
            ("relpos", net.compute, (1, -1, -1, c)),
            ("coords", trt.float32, (1, -1, 3)),
            ("rep_atom", trt.int32, (1, -1)),
            ("atom_to_token", trt.int32, (1, -1)),
            ("intra_index", trt.int32, (1, -1)),
            ("tm_per_bin", trt.float32, (config.pae_bins,)),
        ],
    )
    head = "confidence_head"
    emb = f"{head}.input_embedder"
    single = net.cast(net.layer_norm(inputs["single_inputs"], f"{emb}.single_inputs_norm"), net.compute)
    pair = net.add(net.layer_norm(inputs["pair_trunk"], f"{emb}.pair_norm"), net.cast(inputs["relpos"], net.f32))
    first = net.reshape(net.linear(single, f"{emb}.single_to_pair"), (1, -1, 1, c))
    second = net.reshape(net.linear(single, f"{emb}.single_to_pair_transpose"), (1, 1, -1, c))
    pair = net.add(net.add(pair, net.cast(first, net.f32)), net.cast(second, net.f32))
    left = net.reshape(net.linear(single, f"{emb}.single_to_pair_prod_in1"), (1, -1, 1, c))
    right = net.reshape(net.linear(single, f"{emb}.single_to_pair_prod_in2"), (1, 1, -1, c))
    pair = net.add(pair, net.cast(net.linear(net.mul(left, right), f"{emb}.single_to_pair_prod_out"), net.f32))

    # Distance-bin embedding of the representative atoms of the predicted structure.
    rep = net.gather(inputs["coords"], net.reshape(inputs["rep_atom"], (-1,)), 1)  # [1, N, 3]
    delta = net.sub(net.reshape(rep, (1, -1, 1, 3)), net.reshape(rep, (1, 1, -1, 3)))
    distance = net.unary(net.reduce_sum(net.mul(delta, delta), 3), trt.UnaryOperation.SQRT)  # [1, N, N]
    boundaries = net.constant(net.weight(f"{head}.boundaries"), shape=(1, 1, 1, config.confidence_distance_bins - 1))
    above = net.cast(net.greater(net.reshape(distance, (1, 0, 0, 1)), boundaries), trt.int32)
    bins = net.reduce_sum(above, 3)  # [1, N, N] int32
    table = net.constant(net.weight(f"{head}.dist_bin_pairwise_embed.weight"), net.compute)
    pair = net.add(pair, net.cast(net.gather(table, bins, 0), net.f32))

    # The stack's output is added to its float32 input (the reference's in-place add).
    trunk = layers.pair_update_stack(net, net.cast(pair, net.compute), f"{head}.folding_trunk", config.confidence_layers)
    pair = net.cast(net.add(pair, net.cast(trunk, net.f32)), net.compute)

    # Row attention pooling to the single representation.
    pooling = f"{head}.row_attention_pooling"
    scores = net.reshape(net.linear(pair, f"{pooling}.attn_proj"), (1, 0, 0))  # [1, N, N]
    weights = net.cast(net.softmax(scores, -1), net.compute)
    pooled = net.matmul(net.reshape(weights, (1, 0, 1, -1)), pair)  # [1, N, 1, c]
    single_states = net.linear(net.reshape(pooled, (1, -1, c)), f"{pooling}.out_proj")  # [1, N, hidden]

    # Per-atom pLDDT logits from the token single of each atom and its slot within the token.
    at_atoms = net.gather(single_states, net.reshape(inputs["atom_to_token"], (-1,)), 1)  # [1, A, hidden]
    normed = net.cast(net.layer_norm(at_atoms, f"{head}.plddt_layernorm"), net.compute)
    slots, hidden, bins_plddt = config.max_atoms_per_token, config.hidden_size, config.plddt_bins
    weight = net.weight(f"{head}.plddt_weight")  # [slots, hidden, bins]
    all_slots = net.linear(normed, weight=weight.transpose(0, 2, 1).reshape(slots * bins_plddt, hidden))
    all_slots = net.reshape(all_slots, (1, 0, slots, bins_plddt))
    slot = net.minimum(inputs["intra_index"], net.int_scalar(slots - 1, 2))
    pick = net.reshape(net.one_hot(slot, slots, net.f32), (1, 0, slots, 1))
    logits = net.reduce_sum(net.mul(net.cast(all_slots, net.f32), pick), 2)  # [1, A, bins]
    centers = (np.arange(bins_plddt, dtype=np.float64) + 0.5) / bins_plddt
    probabilities = net.softmax(logits, -1)
    plddt = net.reduce_sum(net.mul(probabilities, net.constant(centers.astype(np.float32), shape=(1, 1, bins_plddt))), 2)
    net.mark_output(plddt, "plddt_atom")

    # pTM rows: expected TM term of every pair under the PAE distribution, averaged over j.
    pae_logits = net.linear(net.cast(net.layer_norm(pair, f"{head}.pae_layernorm"), net.compute), f"{head}.pae_head")
    pae_probabilities = net.softmax(pae_logits, -1)  # [1, N, N, bins] float32
    tm = net.reduce_sum(net.mul(pae_probabilities, net.reshape(inputs["tm_per_bin"], (1, 1, 1, config.pae_bins))), 3)
    count = net.cast(net.dim(tm, 2), net.f32)
    total = net.reduce_sum(tm, 2)  # [1, N]
    denominator = net.add(net.reshape(count, (1, 1)), net.scalar(config.confidence_eps, 2))
    net.mark_output(net.div(total, denominator), "ptm_rows")
