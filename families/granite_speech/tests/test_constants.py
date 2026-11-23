# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host constants and index algebra of the graphs against the reference (CPU).

The front end is evaluated here in NumPy exactly as the audio graph evaluates it (strided DFT
over the reflect-padded clip, power, mel projection, log10, clamp, stacking) and compared with the
official GraniteSpeechFeatureExtractor. The relative-shift reshape and the grouped-query layout
are checked against direct indexing; the rotary tables against Granite's rotary module.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from families.granite_speech import config as config_module, constants

_FE = config_module.FrontEnd(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)


def graph_features(samples: np.ndarray, fe=_FE) -> np.ndarray:
    """The audio graph's front end in NumPy (FP64 accumulation of FP32 constants)."""
    padded = constants.reflect_pad(samples, fe).astype(np.float64)
    frames = 1 + (padded.size - fe.n_fft) // fe.hop_length
    index = np.arange(fe.n_fft)[None, :] + fe.hop_length * np.arange(frames)[:, None]
    spectrum = padded[index] @ constants.dft_basis(fe).astype(np.float64).T  # [F, 2 * bins]
    bins = fe.n_fft // 2 + 1
    power = spectrum[:, :bins] ** 2 + spectrum[:, bins:] ** 2
    mel = power @ constants.mel_filterbank(fe).astype(np.float64)
    log10 = np.log(np.maximum(mel, 1e-10)) / math.log(10.0)
    scaled = np.maximum(log10, log10.max() - 8.0) * 0.25 + 1.0
    pairs = frames // 2
    return scaled[: 2 * pairs].reshape(pairs, 2 * fe.n_mels)


def test_dft_basis_is_the_windowed_one_sided_transform():
    rng = np.random.default_rng(0)
    frame = rng.standard_normal(_FE.n_fft)
    expected = np.fft.rfft(frame * constants.padded_window(_FE))
    product = constants.dft_basis(_FE).astype(np.float64) @ frame
    bins = _FE.n_fft // 2 + 1
    np.testing.assert_allclose(product[:bins], expected.real, atol=1e-4)
    # The sine rows carry +sin; the power is unchanged by the sign of the imaginary part.
    np.testing.assert_allclose(product[bins:], -expected.imag, atol=1e-4)


def test_constants_match_torchaudio():
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    # torch evaluates the window in FP32 (a few ulp from the exact values); the constants are
    # computed in FP64. The feature comparison below bounds the effect on the model's input.
    window = torch.hann_window(_FE.win_length).double().numpy()
    np.testing.assert_allclose(constants.hann_window(_FE.win_length), window, atol=5e-7)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
                                                     hop_length=160, n_mels=80)
    # torchaudio builds the filters in FP32 (linspace and pow), these in FP64: up to 5.3e-6 apart on
    # weights of order 1, i.e. under 1e-6 on the clamped log-mel features.
    np.testing.assert_allclose(constants.mel_filterbank(_FE), transform.mel_scale.fb.numpy(), atol=1e-5)
    signal = torch.randn(1000)
    padded = torch.nn.functional.pad(signal[None, None], (256, 256), mode="reflect")[0, 0].numpy()
    np.testing.assert_array_equal(constants.reflect_pad(signal.numpy(), _FE), padded)


@pytest.mark.parametrize("samples", [257, 4799, 4800, 16000 * 3 + 37])
def test_front_end_matches_the_official_feature_extractor(samples):
    pytest.importorskip("torchaudio")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    extractor = transformers.GraniteSpeechFeatureExtractor()
    rng = np.random.default_rng(samples)
    # Speech-like level and a quiet stretch, so the max-8 clamp is exercised.
    audio = (0.1 * rng.standard_normal(samples)).astype(np.float32)
    audio[: samples // 3] *= 1e-4
    official = extractor(torch.from_numpy(audio))["input_features"][0].numpy()
    ours = graph_features(audio)
    assert ours.shape == official.shape
    np.testing.assert_allclose(ours, official, atol=2e-5)
    assert extractor._get_num_audio_features([samples])[0] == math.ceil(ours.shape[0] / 15) * 3


def shift_numpy(offsets: np.ndarray, context: int) -> np.ndarray:
    """The audio graph's relative shift, in NumPy: pad, flatten, drop C - 1, regrid, crop."""
    batch, heads = offsets.shape[:2]
    padded = np.concatenate([offsets, np.zeros((batch, heads, context, 1), offsets.dtype)], axis=3)
    flat = padded.reshape(batch, heads, 2 * context * context)
    trimmed = flat[:, :, context - 1: context - 1 + context * (2 * context - 1)]
    return trimmed.reshape(batch, heads, context, 2 * context - 1)[..., :context]


def test_relative_shift_turns_offsets_into_positions():
    context = 7
    rng = np.random.default_rng(1)
    offsets = rng.standard_normal((3, 2, context, 2 * context - 1))
    shifted = shift_numpy(offsets, context)
    rows, cols = np.meshgrid(np.arange(context), np.arange(context), indexing="ij")
    np.testing.assert_array_equal(shifted, offsets[:, :, rows, cols - rows + context - 1])


def test_relative_table_reproduces_shaw_positions():
    # The reference: E[clamp(c - r, -C, C) + max_pos], contracted with the query of row c.
    context, max_pos, width = 6, 9, 4
    rng = np.random.default_rng(2)
    embedding = rng.standard_normal((2 * max_pos + 1, width))
    query = rng.standard_normal((2, 3, context, width))
    seq = np.arange(context)
    dists = np.clip(seq[:, None] - seq[None, :], -context, context) + max_pos
    reference = np.einsum("bhcd,crd->bhcr", query, embedding[dists])
    ours = shift_numpy(query @ constants.relative_table(embedding, context, max_pos), context)
    np.testing.assert_allclose(ours, reference, atol=1e-12)


def test_grouped_query_layout_matches_repeated_keys():
    # Query head h reads key head h // group (Transformers' repeat_kv). The decoder groups the
    # query heads of one key head along rows: [kv, group * S, D].
    heads, kv_heads, rows, width, keys = 8, 2, 3, 4, 5
    group = heads // kv_heads
    rng = np.random.default_rng(3)
    query = rng.standard_normal((rows, heads, width))
    key = rng.standard_normal((keys, kv_heads, width))
    repeated = np.repeat(key.transpose(1, 0, 2), group, axis=0)  # [heads, keys, D]
    reference = query.transpose(1, 0, 2) @ repeated.transpose(0, 2, 1)  # [heads, rows, keys]
    grouped = query.transpose(1, 0, 2).reshape(kv_heads, group * rows, width)
    ours = (grouped @ key.transpose(1, 2, 0)).reshape(heads, rows, keys)
    np.testing.assert_allclose(ours, reference, atol=1e-12)


def test_rope_tables_match_granite():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.granite.modeling_granite import GraniteRotaryEmbedding

    text_config = transformers.GraniteConfig(hidden_size=256, num_attention_heads=2, num_key_value_heads=1,
                                             max_position_embeddings=64, rope_theta=10000.0)
    rotary = GraniteRotaryEmbedding(text_config)
    positions = torch.arange(64)[None]
    cos, sin = rotary(torch.zeros(1, 64, 128), positions)
    text = config_module.Text(hidden_size=256, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=128,
                              intermediate_size=1, vocab_size=1, rms_norm_eps=1e-5, rope_theta=10000.0,
                              max_position_embeddings=64, embedding_multiplier=1.0, attention_multiplier=1.0,
                              residual_multiplier=1.0, logits_scaling=1.0, eos_token_id=0)
    ours_cos, ours_sin = constants.rope_tables(text, 64)
    np.testing.assert_allclose(ours_cos, cos[0].numpy(), atol=2e-6)
    np.testing.assert_allclose(ours_sin, sin[0].numpy(), atol=2e-6)
