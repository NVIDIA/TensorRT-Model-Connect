# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration reading: the reference's types and defaults, every refused value named, frame counts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from families.granite_speech5 import config as config_module

# The pinned checkpoint's files (ibm-granite/granite-speech-5.0-470m-turboctc @ b8e0718e).
CONFIG = {
    "architectures": ["GraniteSpeech5ForCTC"], "ctc_loss_reduction": "mean", "ctc_zero_infinity": True,
    "dtype": "bfloat16",
    "encoder_config": {
        "activation_dropout": 0.0, "attention_bias": True, "attention_dropout": 0.0, "context_size": 128,
        "conv_expansion_factor": 2, "conv_kernel_size": 7, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 4096, "max_position_embeddings": 512,
        "model_type": "granite_speech5_encoder", "num_attention_heads": 8, "num_hidden_layers": 16,
        "num_key_value_heads": 8, "num_mel_bins": 80, "subsample_layers": [0, 1], "vocab_size": 16384,
    },
    "initializer_range": 0.02, "model_type": "granite_speech5_ctc", "pad_token_id": 0, "tie_word_embeddings": True,
    "transformers_version": "5.16.1", "vocab_size": 16384,
}
PROCESSOR = {
    "feature_extractor": {
        "delta_win_length": 3, "feature_extractor_type": "GraniteSpeech5FeatureExtractor", "feature_size": 320,
        "hop_length": 160, "logmel_floor_db": 8.0, "n_fft": 512, "num_mel_bins": 80, "padding_side": "right",
        "padding_value": 0.0, "return_attention_mask": True, "sampling_rate": 16000, "win_length": 400,
    },
    "processor_class": "GraniteSpeech5Processor",
}
TOKENIZER_CONFIG = {
    "backend": "tokenizers", "pad_token": "<unk>", "processor_class": "GraniteSpeech5Processor",
    "tokenizer_class": "ParakeetTokenizer", "unk_token": "<unk>",
}
PIECES = {"<unk>": 0, "<|nospeech|>": 1, "<pad>": 2}


def _write(directory: Path, config=None, processor=None, tokenizer_config=None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("config.json", config or CONFIG), ("processor_config.json", processor or PROCESSOR),
                        ("tokenizer_config.json", tokenizer_config or TOKENIZER_CONFIG)):
        (directory / name).write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
    return directory


def _resolve(tmp_path: Path, **files):
    return config_module.resolve(_write(tmp_path / "checkpoint", **files), PIECES)


def _with_encoder(**values) -> dict:
    config = copy.deepcopy(CONFIG)
    for key, value in values.items():
        if value is _DELETE:
            del config["encoder_config"][key]
        else:
            config["encoder_config"][key] = value
    return config


def _with_front_end(**values) -> dict:
    processor = copy.deepcopy(PROCESSOR)
    processor["feature_extractor"].update(values)
    return processor


_DELETE = object()


def test_the_pinned_configuration_resolves(tmp_path):
    cfg = _resolve(tmp_path)
    e, fe = cfg.encoder, cfg.front_end
    assert (e.hidden_size, e.intermediate_size, e.num_layers, e.num_heads, e.head_dim) == (1024, 4096, 16, 8, 128)
    assert (e.context_size, e.max_position_embeddings, e.conv_kernel_size, e.conv_channels) == (128, 512, 7, 2048)
    assert e.subsample_layers == (0, 1) and e.feed_forward_bias and e.input_width == 320
    assert [index for index in range(16) if e.self_conditioning_after(index)] == [7]
    assert (cfg.vocab_size, cfg.blank_id) == (16384, 0)
    assert (fe.sample_rate, fe.n_fft, fe.win_length, fe.hop_length, fe.n_mels) == (16000, 512, 400, 160, 80)
    assert (fe.delta_win_length, fe.floor_db, fe.bins) == (3, 8.0, 257)


def test_absent_keys_and_optional_nulls_take_the_reference_defaults(tmp_path):
    minimal = copy.deepcopy(CONFIG)
    minimal["encoder_config"] = {"model_type": "granite_speech5_encoder", "head_dim": None,
                                 "num_key_value_heads": None, "subsample_layers": None}
    cfg = _resolve(tmp_path, config=minimal, processor={"feature_extractor": {
        "feature_extractor_type": "GraniteSpeech5FeatureExtractor"}})
    reference = _resolve(tmp_path / "pinned")
    assert cfg == reference
    for index, encoder in enumerate((None, _DELETE)):
        default = copy.deepcopy(CONFIG)
        if encoder is _DELETE:
            del default["encoder_config"]
        else:
            default["encoder_config"] = encoder
        assert _resolve(tmp_path / f"default-{index}", config=default) == reference
    broken = copy.deepcopy(CONFIG)
    broken["encoder_config"] = []
    with pytest.raises(ValueError, match="encoder_config must be a JSON object"):
        _resolve(tmp_path / "list", config=broken)


@pytest.mark.parametrize("value", ["1024", 1024.0, True, None])
def test_integers_are_not_coerced(tmp_path, value):
    with pytest.raises(ValueError, match="hidden_size"):
        _resolve(tmp_path, config=_with_encoder(hidden_size=value))


@pytest.mark.parametrize("value", ["8", True, "NaN", "Infinity", "1e39"])
def test_numbers_must_be_finite_float32(tmp_path, value):
    text = json.dumps(PROCESSOR).replace('"logmel_floor_db": 8.0', f'"logmel_floor_db": {json.dumps(value)}')
    if value in ("NaN", "Infinity", "1e39"):
        text = text.replace(f'"{value}"', value)
    with pytest.raises(ValueError, match="logmel_floor_db"):
        _resolve(tmp_path, processor=text)


def test_booleans_are_not_coerced(tmp_path):
    with pytest.raises(ValueError, match="attention_bias"):
        _resolve(tmp_path, config=_with_encoder(attention_bias=1))
    config = copy.deepcopy(CONFIG)
    config["tie_word_embeddings"] = "true"
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        _resolve(tmp_path, config=config)


def test_a_repeated_key_is_refused(tmp_path):
    text = json.dumps(CONFIG).replace('"pad_token_id": 0', '"pad_token_id": 0, "pad_token_id": 2')
    with pytest.raises(ValueError, match="repeats the key 'pad_token_id'"):
        _resolve(tmp_path, config=text)


@pytest.mark.parametrize(("change", "message"), [
    ({"model_type": "granite_speech"}, "model_type"),
    ({"architectures": ["GraniteSpeech5ForCTC", "Other"]}, "architectures"),
    ({"tie_word_embeddings": False}, "untied CTC head"),
    ({"vocab_size": 16000}, "vocab_size 16000 differs"),
    ({"pad_token_id": None}, "pad_token_id null"),
    ({"pad_token_id": 2}, "tokenizer pad token id 0 differs from pad_token_id 2"),
    ({"pad_token_id": 16384}, "outside the vocabulary"),
])
def test_unbuilt_model_values_are_named(tmp_path, change, message):
    config = copy.deepcopy(CONFIG)
    config.update(change)
    with pytest.raises(NotImplementedError, match=message):
        _resolve(tmp_path, config=config)


@pytest.mark.parametrize(("change", "message"), [
    ({"model_type": "granite_speech_encoder"}, "encoder model_type"),
    ({"num_key_value_heads": 4}, "num_key_value_heads 4"),
    ({"hidden_act": "gelu"}, "hidden_act"),
    ({"conv_kernel_size": 6}, "even conv_kernel_size"),
    ({"context_size": 513}, "context_size 513 above"),
    ({"subsample_layers": [0, 0]}, "subsample_layers"),
    ({"subsample_layers": [0, 16]}, "subsample_layers"),
    ({"num_mel_bins": 64}, "64"),
])
def test_unbuilt_encoder_values_are_named(tmp_path, change, message):
    with pytest.raises(NotImplementedError, match=message):
        _resolve(tmp_path, config=_with_encoder(**change))


@pytest.mark.parametrize("value", [-0.5, 1.5])
def test_an_activation_dropout_the_reference_refuses_is_refused(tmp_path, value):
    """``functional.dropout`` refuses it in eval mode too; an attention dropout the reference never applies loads."""
    with pytest.raises(ValueError, match="activation_dropout"):
        _resolve(tmp_path, config=_with_encoder(activation_dropout=value))
    accepted = _resolve(tmp_path / "accepted", config=_with_encoder(activation_dropout=1.0, attention_dropout=2.0))
    assert accepted.encoder.num_layers == 16


def test_negative_subsample_layers_are_refused(tmp_path):
    with pytest.raises(ValueError, match="subsample_layers"):
        _resolve(tmp_path, config=_with_encoder(subsample_layers=[-1]))


@pytest.mark.parametrize(("change", "message"), [
    ({"feature_extractor_type": "WhisperFeatureExtractor"}, "feature extractor"),
    ({"n_fft": 511}, "odd n_fft"),
    ({"win_length": 513}, "win_length 513"),
    ({"hop_length": 1}, "hop_length 1"),
    ({"num_mel_bins": 300}, "300 mel bins"),
])
def test_unbuilt_front_ends_are_named(tmp_path, change, message):
    with pytest.raises(NotImplementedError, match=message):
        _resolve(tmp_path, processor=_with_front_end(**change))


def test_delta_windows_below_three_are_refused(tmp_path):
    with pytest.raises(ValueError, match="delta_win_length"):
        _resolve(tmp_path, processor=_with_front_end(delta_win_length=2))


@pytest.mark.parametrize(("section", "key", "limit"), [
    ("feature_extractor", "n_fft", 8192), ("feature_extractor", "delta_win_length", 31),
    ("feature_extractor", "sampling_rate", 192000), ("encoder_config", "num_hidden_layers", 256),
    ("both", "vocab_size", 1 << 20),
])
def test_values_that_size_the_graph_are_bounded(tmp_path, section, key, limit):
    """At the limit the value is read; one above it is refused before any table or graph is sized."""
    for value, accepted in ((limit, True), (limit + 1, False)):
        config, processor = copy.deepcopy(CONFIG), copy.deepcopy(PROCESSOR)
        if section == "feature_extractor":
            processor["feature_extractor"][key] = value
        elif section == "encoder_config":
            config["encoder_config"][key] = value
        else:
            config[key] = config["encoder_config"][key] = value
        directory = tmp_path / f"{key}-{value}"
        if accepted:
            _resolve(directory, config=config, processor=processor)
        else:
            with pytest.raises(ValueError, match=f"{key} up to {limit}, not {value}"):
                _resolve(directory, config=config, processor=processor)


def test_the_tokenizer_clean_up_is_refused(tmp_path):
    """Transformers would rewrite ' .' to '.' after decoding; the runtime decodes without it."""
    with pytest.raises(NotImplementedError, match="clean_up_tokenization_spaces"):
        _resolve(tmp_path / "on", tokenizer_config={**TOKENIZER_CONFIG, "clean_up_tokenization_spaces": True})
    with pytest.raises(ValueError, match="clean_up_tokenization_spaces"):
        _resolve(tmp_path / "text", tokenizer_config={**TOKENIZER_CONFIG, "clean_up_tokenization_spaces": "no"})
    off = _resolve(tmp_path / "off", tokenizer_config={**TOKENIZER_CONFIG, "clean_up_tokenization_spaces": False})
    assert off == _resolve(tmp_path / "absent")


def test_the_tokenizer_must_decode_as_parakeet(tmp_path):
    with pytest.raises(NotImplementedError, match="tokenizer_class"):
        _resolve(tmp_path, tokenizer_config={**TOKENIZER_CONFIG, "tokenizer_class": "PreTrainedTokenizerFast"})
    with pytest.raises(NotImplementedError, match="pad_token"):
        _resolve(tmp_path, tokenizer_config={**TOKENIZER_CONFIG, "pad_token": "<missing>"})


def test_sample_limits(tmp_path):
    cfg = _resolve(tmp_path)
    limits = config_module.sample_limits(cfg)
    assert limits == {"min_samples": 1120, "opt_samples": 240000, "max_samples": 480000,
                      "min_waveform": 1632, "opt_waveform": 240512, "max_waveform": 480513}
    assert cfg.output_frames(1120) == 1 and cfg.output_frames(1119) == 0
    assert config_module.sample_limits(cfg, 16000)["opt_samples"] == 16000
    for bad in (1119, 600 * 16000 + 1, True, 16000.0):
        with pytest.raises(ValueError, match="max_sequence_length"):
            config_module.sample_limits(cfg, bad)


def test_frame_counts_match_the_feature_extractor(tmp_path):
    """Stacked frames and the tail sample, as the official extractor computes them, at many lengths."""
    from transformers import GraniteSpeech5FeatureExtractor

    cfg = _resolve(tmp_path)
    fe = cfg.front_end
    extractor = GraniteSpeech5FeatureExtractor(**{key: value for key, value in PROCESSOR["feature_extractor"].items()
                                                  if key != "feature_extractor_type"})
    rng = np.random.default_rng(0)
    lengths = list(range(1100, 1700)) + [int(value) for value in rng.integers(1700, 200000, 40)] + [373040]
    tails = 0
    for length in lengths:
        features = extractor(rng.standard_normal(length).astype(np.float32), sampling_rate=16000,
                             return_tensors="np")["input_features"]
        assert features.shape == (1, fe.stacked_frames(length), 320), length
        hop_frames = length // fe.hop_length
        stacked = 2 * -(-hop_frames // 2)
        assert fe.needs_tail_sample(length) == (length < (stacked - 1) * fe.hop_length + 1), length
        tails += fe.needs_tail_sample(length)
    assert tails > 0, "no length exercised the tail sample"
    assert cfg.output_frames(373040) == 291
