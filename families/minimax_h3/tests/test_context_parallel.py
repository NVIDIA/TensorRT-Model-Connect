# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel (Ulysses) tests for the MiniMax-H3 DiT.

``test_ulysses_layout_matches_full_attention`` emulates the exact reshape /
transpose / collective sequence of ``cp_ops`` in NumPy on N simulated ranks and
compares against single-device attention, so the permutation algebra is checked
without a GPU. The graph tests construct (not build) the real network.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from families.minimax_h3.config import MiniMaxH3Config

H, D = 8, 16


def _softmax_attention(q, k, v):
    scores = np.einsum("hqd,hkd->hqk", q, k) / np.sqrt(q.shape[-1])
    scores -= scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(-1, keepdims=True)
    return np.einsum("hqk,hkd->hqd", probs, v)


def _all_to_all(per_rank):
    """Rank r receives chunk r (dim 0) of every source rank, stacked by source."""
    n = len(per_rank)
    return [np.stack([per_rank[src][dst] for src in range(n)]) for dst in range(n)]


def _all_gather(per_rank):
    full = np.concatenate(per_rank, axis=0)
    return [full for _ in per_rank]


@pytest.mark.parametrize("n", [1, 2, 4, 8])
@pytest.mark.parametrize("head_rows", [1, 5, 23])
def test_ulysses_layout_matches_full_attention(n, head_rows):
    video = 8 * 6  # divisible by every CP size
    local = video // n
    h = H // n
    rng = np.random.default_rng(3)
    total = head_rows + video
    # q/k/v rows as [rows, H*D], replicated head segment then the video rows.
    full = {name: rng.normal(size=(total, H * D)).astype(np.float32) for name in "qkv"}

    # Reference: ordinary multi-head attention over the whole packed sequence.
    def heads(x):
        return x.reshape(total, H, D).transpose(1, 0, 2)

    reference = _softmax_attention(*(heads(full[c]) for c in "qkv"))
    reference = reference.transpose(1, 0, 2).reshape(total, H * D)

    def rank_rows(name, r):
        rows = full[name]
        return rows[:head_rows], rows[head_rows + r * local : head_rows + (r + 1) * local]

    # --- per-rank inputs after the exchanges (mirrors cp_ops) ---
    merged = {c: [] for c in "qkv"}
    for c in "qkv":
        # head-block select: [T, N, h*D] -> [N, T, h*D]; all-to-all of replicated data
        sent = []
        for r in range(n):
            head, _ = rank_rows(c, r)
            sent.append(head.reshape(-1, n, h * D).transpose(1, 0, 2))
        picked = [chunks[0].reshape(-1, h, D).transpose(1, 0, 2)[None] for chunks in _all_to_all(sent)]
        # video seq -> heads
        sent = []
        for r in range(n):
            _, vid = rank_rows(c, r)
            sent.append(vid.reshape(local, n, h, D).transpose(1, 0, 2, 3))
        full_seq = [
            ex.transpose(2, 0, 1, 3).reshape(1, h, video, D) for ex in _all_to_all(sent)
        ]
        merged[c] = [np.concatenate((picked[r], full_seq[r]), axis=2) for r in range(n)]

    context = [
        _softmax_attention(*(merged[c][r][0] for c in "qkv"))[None] for r in range(n)
    ]

    # --- exchanges back ---
    sent = []
    for r in range(n):
        vid_ctx = context[r][:, :, head_rows:, :]
        sent.append(vid_ctx.reshape(h, n, local, D).transpose(1, 0, 2, 3))
    video_out = [
        ex.transpose(2, 0, 1, 3).reshape(local, H * D) for ex in _all_to_all(sent)
    ]
    gathered_in = [context[r][:, :, :head_rows, :].reshape(h, -1, D) for r in range(n)]
    head_out = [
        g.transpose(1, 0, 2).reshape(-1, H * D) for g in _all_gather(gathered_in)
    ]

    for r in range(n):
        rows = np.concatenate((head_out[r], video_out[r]), axis=0)
        expected = np.concatenate(
            (reference[:head_rows], reference[head_rows + r * local : head_rows + (r + 1) * local])
        )
        np.testing.assert_allclose(rows, expected, rtol=1e-5, atol=1e-5)


def test_config_validates_cp_sizes():
    base = MiniMaxH3Config()
    for n in (1, 2, 4, 8):
        replace(base, context_parallel_size=n).validate()
    for bad in (3, 5, 16):
        with pytest.raises(ValueError, match="context_parallel_size"):
            replace(base, context_parallel_size=bad).validate()
    with pytest.raises(ValueError, match="first_block_cache"):
        replace(base, context_parallel_size=2, first_block_cache=True).validate()


def test_cp_row_geometry():
    cfg = replace(MiniMaxH3Config(), context_parallel_size=8)
    assert cfg.local_video_rows == 4662
    assert cfg.local_heads == 7
    assert cfg.local_video_rows * 8 == cfg.video_rows


# ---- real TensorRT graph construction (no GPU required) ----

trt_mod = pytest.importorskip("families.minimax_h3._trt")
trt = trt_mod.trt


def _small_profile(n: int) -> MiniMaxH3Config:
    video, audio, text = 64, 8, 16
    return MiniMaxH3Config(
        hidden_size=256,
        num_layers=2,
        num_refiner_layers=1,
        num_heads=8,
        head_dim=128,
        ffn_dim=128,
        text_dim=64,
        timestep_input_dim=32,
        timestep_hidden_size=64,
        timestep_embed_dim=32,
        video_rows=video,
        audio_rows=audio,
        text_rows=text,
        min_text_rows=1,
        opt_text_rows=4,
        padded_sequence_length=video + audio + text,
        context_parallel_size=n,
    )


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_cp_dit_network_structure(n):
    if not hasattr(trt.INetworkDefinition, "add_dist_collective"):
        pytest.skip("TensorRT build has no distributed collectives")
    from families.minimax_h3 import cp_ops
    from families.minimax_h3.dit_builder import checkpoint_keys, construct_dit_network
    from families.minimax_h3.tests.test_trt_builders import _weights

    profile = _small_profile(n)
    profile.validate()
    weights = _weights(profile)
    keep = set(checkpoint_keys(profile))
    weights = {k: v for k, v in weights.items() if k in keep}
    _, _, network, _ = construct_dit_network(weights, profile, workspace_bytes=1 << 30)

    kinds = [network.get_layer(i).type for i in range(network.num_layers)]
    collectives = kinds.count(trt.LayerType.DIST_COLLECTIVE)
    assert collectives == cp_ops.expected_collectives(profile)
    assert (collectives == 0) == (n == 1)
    # Same runtime ABI at every CP size: gathered video velocity is the full size.
    outputs = {network.get_output(i).name: tuple(network.get_output(i).shape) for i in range(network.num_outputs)}
    assert outputs["video_velocity"][-1] == profile.video_patch_dim
    assert outputs["audio_velocity"] == (profile.audio_rows, profile.audio_in_channels)
    if n > 1:
        assert outputs["video_velocity"][0] == profile.video_rows
        ranks = {
            network.get_layer(i).num_ranks
            for i in range(network.num_layers)
            if network.get_layer(i).type == trt.LayerType.DIST_COLLECTIVE
        }
        assert ranks == {n}


def _vae_weights(vae, layers: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)

    def w(*shape):
        return rng.normal(0.0, 0.02, shape).astype(np.float32)

    d, ffn = vae.DIM, vae.FFN_DIM
    state = {
        "post_quant_conv.weight": w(vae.CHANNELS, vae.CHANNELS),
        "post_quant_conv.bias": w(vae.CHANNELS),
        "decoder.proj_in.weight": w(d, vae.CHANNELS),
        "decoder.proj_in.bias": w(d),
        "decoder.register_tokens": w(1, vae.REGISTER_TOKENS, d),
        "decoder.norm_out.weight": np.ones(d, np.float32),
        "decoder.norm_out.bias": w(d),
        "decoder.proj_out.weight": w(vae.OUT_CHANNELS * vae.PATCH_T * vae.PATCH**2, d),
        "decoder.proj_out.bias": w(vae.OUT_CHANNELS * vae.PATCH_T * vae.PATCH**2),
    }
    for i in range(layers):
        p = f"decoder.transformer_blocks.{i}"
        state[f"{p}.norm1.weight"] = np.ones(d, np.float32)
        state[f"{p}.norm2.weight"] = np.ones(d, np.float32)
        state[f"{p}.scale1"] = np.ones(d, np.float32)
        state[f"{p}.scale2"] = np.ones(d, np.float32)
        for name in "qkv":
            state[f"{p}.attn.to_{name}.weight"] = w(d, d)
            state[f"{p}.attn.to_{name}.bias"] = w(d)
        state[f"{p}.attn.to_out.0.weight"] = w(d, d)
        state[f"{p}.attn.to_out.0.bias"] = w(d)
        state[f"{p}.ff.net.0.proj.weight"] = w(2 * ffn, d)
        state[f"{p}.ff.net.0.proj.bias"] = w(2 * ffn)
        state[f"{p}.ff.net.2.weight"] = w(d, ffn)
        state[f"{p}.ff.net.2.bias"] = w(d)
    return state


@pytest.mark.parametrize("n", [1, 2, 4, 7, 8])
def test_cp_vae_tile_decoder_structure(n, monkeypatch):
    if not hasattr(trt.INetworkDefinition, "add_dist_collective"):
        pytest.skip("TensorRT build has no distributed collectives")
    from families.minimax_h3 import vae_builder as vae

    if n not in (1, 2, 4, 8):
        with pytest.raises(ValueError, match="cp_size"):
            vae.construct_vae_tile_decoder_network({}, cp_size=n)
        return
    monkeypatch.setattr(vae, "LAYERS", 1)
    _, _, network, _ = vae.construct_vae_tile_decoder_network(
        _vae_weights(vae, 1), workspace_bytes=1 << 30, cp_size=n
    )
    kinds = [network.get_layer(i).type for i in range(network.num_layers)]
    assert kinds.count(trt.LayerType.DIST_COLLECTIVE) == (0 if n == 1 else 2)
    # ABI is identical at every CP size: all BATCH tiles in, all BATCH tiles out.
    assert tuple(network.get_input(0).shape) == (vae.BATCH, vae.CHANNELS, vae.FRAMES, vae.HEIGHT, vae.WIDTH)
    assert tuple(network.get_output(0).shape) == (
        vae.BATCH,
        vae.OUT_CHANNELS,
        vae.FRAMES * vae.PATCH_T,
        vae.HEIGHT * vae.PATCH,
        vae.WIDTH * vae.PATCH,
    )
