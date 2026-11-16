# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry resolution, front-end constants and rotary tables, without TensorRT.

The comparisons against librosa, torch and transformers run where those are
installed; the pinned literals next to them were measured from the same
originals, so the CPU lane keeps the checks without the dependencies.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from families.nemotron3_diarization import config as config_module
from families.nemotron3_diarization import frontend, rotary

DATA = Path(__file__).resolve().parent / "data"

# The released checkpoint's config.json and processor_config.json, reduced to
# what the family reads.
CONFIG = {
    "architectures": ["Nemotron3DiarizationForAudioFrameClassification"],
    "model_type": "nemotron3_diarization",
    "audio_config": {
        "hidden_act": "gelu", "hidden_size": 512, "intermediate_size": 2048,
        "max_position_embeddings": 5000, "num_attention_heads": 8, "num_hidden_layers": 31,
        "num_key_value_heads": 8, "num_mel_bins": 128, "subsampling_factor": 8,
        "rope_parameters": {"partial_rotary_factor": 1.0, "rope_theta": 10000.0,
                            "rope_type": "default"},
    },
    "head_config": {"audio_hidden_size": 512, "hidden_size": 192, "num_speakers": 8,
                    "subsampling_factor": 8},
    "streaming_config": {
        "fifo_length": 264, "latest_frames_score_boost": 0.05, "min_positive_scores_rate": 0.5,
        "num_speakers": 8, "prediction_score_threshold": 0.25, "speaker_cache_length": 264,
        "speaker_cache_silence_frames_per_speaker": 1, "speaker_cache_update_period": 222,
        "strong_boost_rate": 0.75, "subsampling_factor": 8, "weak_boost_rate": 1.5,
    },
    "chunk_length": 340, "chunk_right_context": 40, "fifo_length": 40,
    "speaker_cache_update_period": 300,
}
PROCESSOR = {
    "feature_extractor": {
        "feature_extractor_type": "NemotronAsrStreamingFeatureExtractor", "feature_size": 128,
        "hop_length": 160, "n_fft": 512, "padding_value": 0.0, "preemphasis": 0.97,
        "sampling_rate": 16000, "win_length": 400,
    },
    "subsampling_factor": 8,
}


def _shapes(layers: int = 31) -> dict:
    shapes = {
        "model.audio_tower.embedder.projection.weight": (512, 1024),
        "model.audio_tower.layers.0.mlp.fc1.weight": (2048, 512),
        "model.proj.weight": (192, 512),
        "model.upsampler.conv.weight": (1536, 192, 3),
        "classifier.dense.weight": (192, 192),
        "classifier.out_proj.weight": (8, 192),
        "silence_embeds": (512,),
    }
    for index in range(layers):
        shapes[f"model.audio_tower.layers.{index}.self_attn.q_proj.weight"] = (512, 512)
    return shapes


def _write(directory: Path, config: dict, processor: dict) -> Path:
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    return directory


def test_resolve_reads_the_released_geometry(tmp_path):
    cfg = config_module.resolve(_write(tmp_path, CONFIG, PROCESSOR), _shapes())
    assert (cfg["hidden_size"], cfg["num_heads"], cfg["head_dim"], cfg["num_layers"]) == (
        512, 8, 64, 31)
    assert (cfg["head_hidden_size"], cfg["num_speakers"], cfg["subsampling_factor"]) == (192, 8, 8)
    # Offline mode uses the top-level FIFO sizes, not the streaming ones.
    assert (cfg["chunk_length"], cfg["chunk_left_context"], cfg["chunk_right_context"]) == (
        340, 0, 40)
    assert (cfg["fifo_length"], cfg["speaker_cache_update_period"]) == (40, 300)
    # 264 // 8 - 1 = 32 frames per speaker: half, three quarters and 1.5 of it.
    assert (cfg["min_positive_scores"], cfg["strong_boosted_frames"],
            cfg["weak_boosted_frames"]) == (16, 24, 48)
    # Cache, FIFO, chunk and look-ahead: the longest step one engine call sees.
    assert cfg["max_step_rows"] == 264 + 40 + 340 + 40
    assert (cfg["sample_rate"], cfg["n_fft"], cfg["win_length"], cfg["hop_length"]) == (
        16000, 512, 400, 160)
    assert cfg["log_guard"] == 2.0**-24


def test_runtime_config_names_every_speaker(tmp_path):
    cfg = config_module.resolve(_write(tmp_path, CONFIG, PROCESSOR), _shapes())
    runtime = config_module.runtime_config(cfg)
    assert set(runtime) == set(config_module.RUNTIME_KEYS) | {"labels"}
    assert runtime["labels"] == [f"speaker_{index}" for index in range(8)]


def _variant(path: tuple, value):
    config, processor = copy.deepcopy(CONFIG), copy.deepcopy(PROCESSOR)
    target = {"config": config, "processor": processor}[path[0]]
    for key in path[1:-1]:
        target = target[key]
    target[path[-1]] = value
    return config, processor


@pytest.mark.parametrize("path, value", [
    (("config", "audio_config", "rope_parameters", "partial_rotary_factor"), 0.5),
    (("config", "audio_config", "rope_parameters", "rope_type"), "dynamic"),
    (("config", "audio_config", "num_key_value_heads"), 4),
    (("config", "audio_config", "hidden_act"), "relu"),
    (("config", "head_config", "subsampling_factor"), 4),
    (("config", "streaming_config", "speaker_cache_length"), 8),
    (("config", "chunk_right_context"), 340),
    (("processor", "feature_extractor", "mel_filters_dtype"), "bfloat16"),
    (("processor", "feature_extractor", "n_fft"), 500),
])
def test_resolve_refuses_what_the_graph_does_not_implement(tmp_path, path, value):
    config, processor = _variant(path, value)
    with pytest.raises(ValueError):
        config_module.resolve(_write(tmp_path, config, processor), _shapes())


def test_resolve_checks_the_tensors(tmp_path):
    directory = _write(tmp_path, CONFIG, PROCESSOR)
    with pytest.raises(ValueError, match="encoder layer count"):
        config_module.resolve(directory, _shapes(layers=30))
    shapes = _shapes()
    shapes["model.upsampler.conv.weight"] = (1536, 192, 5)
    with pytest.raises(ValueError, match="upsampler"):
        config_module.resolve(directory, shapes)
    (directory / "processor_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        config_module.resolve(directory, _shapes())


def test_mel_filters_are_slaney_triangles_over_the_whole_band():
    filters = frontend.mel_filters(128, 512, 16000)
    assert filters.shape == (128, 257) and filters.dtype == np.float32
    # Measured from librosa.filters.mel(sr=16000, n_fft=512, n_mels=128,
    # fmin=0, fmax=8000, norm="slaney"), which the values equal bit for bit.
    assert float(filters.astype(np.float64).sum()) == pytest.approx(4.09048765, abs=1e-8)
    assert np.nonzero(filters[0])[0].tolist() == [1]
    assert np.nonzero(filters[127])[0].tolist() == list(range(244, 256))
    assert float(filters[10, 8]) == pytest.approx(0.029578963294625282, rel=1e-7)
    librosa = pytest.importorskip("librosa")
    reference = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000.0,
                                    norm="slaney")
    assert np.array_equal(filters, reference)


def test_window_is_the_symmetric_hann():
    window = frontend.symmetric_hann(400)
    assert window.shape == (400,) and window.dtype == np.float32
    # Symmetric, not periodic: zero at both ends, the peak shared by 199 and 200.
    assert window[0] == 0.0 and window[-1] == 0.0
    assert window[199] == window[200] == pytest.approx(0.9999845, abs=1e-7)
    periodic = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    assert np.abs(window - periodic).max() > 1e-3
    torch = pytest.importorskip("torch")
    reference = torch.hann_window(400, periodic=False).numpy()
    # float32 cosines from two libraries: within one unit in the last place.
    assert np.abs(window - reference).max() <= 6e-8


def test_shipped_test_constants_are_the_family_constants():
    # tests/cpp/test_runtime.cpp loads these files as the bundle would.
    filters = np.fromfile(DATA / "mel_filters.f32", dtype=np.float32).reshape(128, 257)
    window = np.fromfile(DATA / "window.f32", dtype=np.float32)
    assert np.array_equal(filters, frontend.mel_filters(128, 512, 16000))
    assert np.array_equal(window, frontend.symmetric_hann(400))


def test_rotary_tables_follow_the_reference_rounding():
    inverse = rotary.inverse_frequencies(64, 10000.0)
    assert inverse.dtype == np.float32 and inverse[0] == 1.0
    assert float(inverse[1]) == 0.7498942017555237
    cos, sin = rotary.rotary_tables(64, 10000.0, 684)
    assert cos.shape == sin.shape == (684, 64)
    assert np.all(cos[0] == 1.0) and np.all(sin[0] == 0.0)
    # rotate_half pairs dimension i with i + 32, so both halves share angles.
    assert np.array_equal(cos[:, :32], cos[:, 32:]) and np.array_equal(sin[:, :32], sin[:, 32:])
    assert float(cos[683, 0]) == pytest.approx(-0.29208114743232727, abs=1e-7)
    torch = pytest.importorskip("torch")
    modeling = pytest.importorskip(
        "transformers.models.nemotron3_diarization.modeling_nemotron3_diarization")
    configuration = pytest.importorskip(
        "transformers.models.nemotron3_diarization.configuration_nemotron3_diarization")
    audio = configuration.Nemotron3DiarizationAudioConfig(
        rope_parameters=CONFIG["audio_config"]["rope_parameters"])
    embedding = modeling.Nemotron3DiarizationRotaryEmbedding(audio)
    assert np.array_equal(inverse, embedding.inv_freq.numpy())
    reference_cos, reference_sin = embedding(torch.zeros(1, 684, 512), torch.arange(684)[None])
    assert np.abs(cos - reference_cos[0].numpy()).max() <= 6e-8
    assert np.abs(sin - reference_sin[0].numpy()).max() <= 6e-8
