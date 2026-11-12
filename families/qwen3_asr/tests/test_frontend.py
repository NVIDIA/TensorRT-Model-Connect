# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Qwen3-ASR geometry, filterbank, positions, configuration and token tables."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from families.qwen3_asr import config as config_module
from families.qwen3_asr import frontend, tokens


def _reference_token_length(frames: int) -> int:
    """``Qwen3ASRProcessor._get_audio_token_length`` (Transformers 5.22.2), transcribed."""
    remainder = frames % 100
    feat = (remainder - 1) // 2 + 1
    per_chunk = (feat - 1) // 2 + 1
    return (per_chunk - 1) // 2 + 1 + (frames // 100) * 13


def test_token_count_matches_the_processor_formula() -> None:
    for frames in range(1, 2501):
        assert frontend.audio_token_count(frames, chunk_frames=100) == _reference_token_length(frames)
        assert sum(frontend.chunk_token_counts(frames, chunk_frames=100)) == _reference_token_length(frames)
    # The LibriSpeech fixture: 373040 samples -> 2331 frames -> 303 placeholders.
    assert frontend.mel_frame_count(373040, hop_length=160, min_samples=8000) == 2331
    assert frontend.audio_token_count(2331, chunk_frames=100) == 303
    # Clips under min_length are padded first and the padding counts as audio.
    assert frontend.mel_frame_count(5000, hop_length=160, min_samples=8000) == 50


def test_mel_filters_match_transformers() -> None:
    audio_utils = pytest.importorskip("transformers.audio_utils")
    expected = audio_utils.mel_filter_bank(
        num_frequency_bins=201,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    ).astype(np.float32)
    actual = frontend.slaney_mel_filters(
        num_frequency_bins=201, num_mel_filters=128, max_frequency=8000.0, sampling_rate=16000
    )
    assert actual.dtype == np.float32 and actual.shape == (201, 128)
    np.testing.assert_array_equal(actual, expected)


def test_sinusoid_positions_match_the_reference_formula() -> None:
    torch = pytest.importorskip("torch")
    length, channels = 13, 896
    increment = np.log(10000) / (channels // 2 - 1)
    inverse = torch.exp(-increment * torch.arange(channels // 2).float())
    scaled = torch.arange(length)[:, np.newaxis] * inverse[np.newaxis, :]
    expected = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).numpy()
    # Both sides evaluate float32 exp / sin / cos; their libraries may differ by one float32 ulp of
    # the angle, about 1e-6 at the largest angle (12 rad). A structural error is orders larger.
    np.testing.assert_allclose(frontend.sinusoid_positions(length, channels), expected, rtol=0, atol=2e-6)


def _write_checkpoint(directory, **overrides) -> None:
    text = {
        "model_type": "qwen3", "hidden_act": "silu", "attention_bias": False,
        "tie_word_embeddings": True, "use_sliding_window": False, "sliding_window": None,
        "layer_types": ["full_attention"] * 2, "rope_parameters": {"rope_theta": 1.0e6, "rope_type": "default"},
        "vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 48, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
    }
    audio = {
        "model_type": "qwen3_asr_encoder", "activation_function": "gelu", "scale_embedding": False,
        "num_mel_bins": 128, "d_model": 24, "encoder_layers": 2, "encoder_attention_heads": 4,
        "num_key_value_heads": 4, "encoder_ffn_dim": 40, "downsample_hidden_size": 8, "n_window": 50,
        "n_window_infer": 800, "max_position_embeddings": 13, "output_dim": 32,
    }
    text.update(overrides.pop("text", {}))
    audio.update(overrides.pop("audio", {}))
    raw = {
        "model_type": "qwen3_asr", "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": audio, "text_config": text, "audio_token_id": 60, "eos_token_id": [61, 62],
    }
    raw.update(overrides)
    (directory / "config.json").write_text(json.dumps(raw))
    processor = {"feature_extractor": {
        "feature_extractor_type": "Qwen3ASRFeatureExtractor", "dither": 0.0, "padding_value": 0.0,
        "sampling_rate": 16000, "n_fft": 400, "hop_length": 160, "feature_size": 128,
        "min_length": 8000, "n_window": 50,
    }}
    (directory / "processor_config.json").write_text(json.dumps(processor))


def test_configuration_is_read_and_checked(tmp_path) -> None:
    _write_checkpoint(tmp_path)
    loaded = config_module.load_config(tmp_path)
    assert loaded.encoder.window_chunks == 8 and loaded.encoder.window_tokens == 104
    assert loaded.encoder.tokens_per_chunk == 13 and loaded.encoder.freq_bins_after_cnn == 16
    assert loaded.decoder.heads == 4 and loaded.decoder.kv_heads == 2
    assert loaded.eos_token_ids == (61, 62)
    assert loaded.features.min_samples == 8000


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": {"tie_word_embeddings": False}},
        {"text": {"use_sliding_window": True}},
        {"text": {"rope_parameters": {"rope_theta": 1.0e6, "rope_type": "yarn"}}},
        # Transformers applies a legacy ``rope_scaling`` (``type`` alias) in place of ``rope_parameters``.
        {"text": {"rope_scaling": {"type": "linear", "factor": 2.0}}},
        {"text": {"rope_parameters": None, "rope_scaling": {"rope_type": "dynamic", "factor": 2.0},
                  "rope_theta": 1.0e6}},
        {"text": {"rope_parameters": {"full_attention": {"rope_theta": 1.0e6, "rope_type": "linear",
                                                         "factor": 2.0}}}},
        {"text": {"rope_parameters": {"rope_theta": 1.0e6, "rope_type": "default",
                                      "partial_rotary_factor": 0.5}}},
        {"audio": {"n_window_infer": 750}},
        {"audio": {"output_dim": 16}},
        {"architectures": ["Qwen3ASRForTokenClassification"]},
        {"thinker_config": {}},
    ],
)
def test_unsupported_layouts_are_rejected(tmp_path, overrides) -> None:
    _write_checkpoint(tmp_path, **overrides)
    with pytest.raises(ValueError):
        config_module.load_config(tmp_path)


@pytest.mark.parametrize(
    ("text", "theta"),
    [
        ({"rope_parameters": None, "rope_scaling": {"type": "default"}, "rope_theta": 5.0e5}, 5.0e5),
        ({"rope_parameters": None, "rope_theta": 2.0e5}, 2.0e5),
        ({"rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 3.0e5}}}, 3.0e5),
    ],
)
def test_equivalent_rope_layouts_are_read(tmp_path, text, theta) -> None:
    _write_checkpoint(tmp_path, text=text)
    assert config_module.load_config(tmp_path).decoder.rope_theta == theta


def test_feature_extractor_defaults_follow_the_reference(tmp_path) -> None:
    # Qwen3ASRFeatureExtractor pads clips shorter than min_length, 8000 samples by default.
    _write_checkpoint(tmp_path)
    processor = json.loads((tmp_path / "processor_config.json").read_text())
    del processor["feature_extractor"]["min_length"]
    (tmp_path / "processor_config.json").write_text(json.dumps(processor))
    assert config_module.load_config(tmp_path).features.min_samples == 8000


def test_byte_decoder_inverts_the_byte_level_alphabet() -> None:
    decoder = tokens._byte_decoder()
    assert len(decoder) == 256 and sorted(decoder.values()) == list(range(256))
    assert decoder["Ġ"] == ord(" ") and decoder["Ċ"] == ord("\n") and decoder["A"] == ord("A")


def test_vocabulary_table_decoding_skips_special_and_unknown_ids() -> None:
    entries = [(0, b"Hel"), (0, b"lo"), (tokens.FLAG_SPECIAL, b"<|im_end|>"), (tokens.FLAG_UNKNOWN, b""),
               (0, b"\xe2\x82"), (0, b"\xac")]
    table = bytearray(tokens.VOCAB_MAGIC) + struct.pack("<II", tokens.VOCAB_VERSION, len(entries))
    for flag, data in entries:
        table += struct.pack("<BH", flag, len(data)) + data
    assert tokens.decode_with_table(bytes(table), [0, 1, 2, 3]) == "Hello"
    assert tokens.decode_with_table(bytes(table), [4, 5]) == "€"
    assert tokens.decode_with_table(bytes(table), [4]) == "�"
