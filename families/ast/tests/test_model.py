# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The filter-bank constants and the geometry resolution, without TensorRT.

The expected values here were measured against
``torchaudio.compliance.kaldi``, which is what AST's reference feature
extractor calls. They are pinned as literals rather than recomputed from
torchaudio so the unit lane does not need it installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.ast import config as config_module
from families.ast.fbank import hann_window, mel_filters, mel_scale

_SAMPLE_RATE = 16000.0
_N_FFT = 512
_NUM_MEL_BINS = 128
_FRAME_LENGTH = 400


def test_hann_window_is_symmetric_not_periodic():
    window = hann_window(_FRAME_LENGTH)
    assert window.shape == (_FRAME_LENGTH,)
    # Kaldi's "hanning" includes the endpoint, so both ends are zero. The
    # periodic window used by most spectrogram code is not zero at the end,
    # and swapping the two moves the normalised features by 0.086.
    assert window[0] == pytest.approx(0.0, abs=1e-7)
    assert window[-1] == pytest.approx(0.0, abs=1e-7)
    periodic = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_FRAME_LENGTH) / _FRAME_LENGTH)
    assert periodic[-1] > 0.0
    assert np.abs(window - periodic).max() > 1e-3
    # Symmetric about the centre, which the periodic window is not.
    assert np.abs(window - window[::-1]).max() == 0.0
    assert np.abs(periodic - periodic[::-1]).max() > 1e-4


def test_mel_scale_uses_htk_constants():
    # 1127 * ln(1 + f / 700). Slaney's piecewise scale is linear below 1 kHz
    # and would place every low filter somewhere else.
    assert float(mel_scale(0.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(mel_scale(700.0)) == pytest.approx(1127.0 * np.log(2.0), rel=1e-12)
    assert float(mel_scale(1000.0)) == pytest.approx(999.99070, abs=1e-4)


def test_mel_filters_match_the_measured_kaldi_bank():
    banks = mel_filters(_NUM_MEL_BINS, _N_FFT, _SAMPLE_RATE)
    assert banks.shape == (_NUM_MEL_BINS, _N_FFT // 2 + 1)
    # Measured against torchaudio.compliance.kaldi.get_mel_banks, which agrees
    # to 1.4e-05 elementwise; the total is the compact form of that check.
    assert float(banks.sum()) == pytest.approx(252.61049, abs=1e-3)
    # Kaldi builds its triangles over n_fft // 2 bins and never weights the
    # Nyquist bin, so the padding column is zero by construction.
    assert float(np.abs(banks[:, -1]).max()) == 0.0
    assert float(banks.min()) == 0.0
    # 128 triangles over 20 Hz to Nyquist are narrower than the 31.25 Hz bin
    # spacing at the bottom of the band, so exactly one of them - filter 3 -
    # falls entirely between two FFT bins and collects nothing at all. That is
    # Kaldi's own behaviour, and it moves if the mel scale or either band edge
    # is wrong, so it is pinned rather than smoothed over.
    occupied = banks.max(axis=1) > 0.0
    assert np.flatnonzero(~occupied).tolist() == [3]
    # Every other triangle has one peak, and the peaks climb with the index.
    peaks = banks.argmax(axis=1)[occupied]
    assert np.all(np.diff(peaks) >= 0)
    assert peaks[0] < peaks[-1]
    # Nothing below low_freq gets any weight: 20 Hz is under bin 1 at 31.25 Hz
    # per bin, so the DC bin stays empty.
    assert float(np.abs(banks[:, 0]).max()) == 0.0


def test_mel_filters_refuse_a_band_that_does_not_fit():
    with pytest.raises(ValueError):
        mel_filters(_NUM_MEL_BINS, _N_FFT, _SAMPLE_RATE, low_freq=9000.0)


def _checkpoint(tmp_path, *, config=None, preprocessor=None):
    model = {
        "model_type": "audio-spectrogram-transformer",
        "architectures": ["ASTForAudioClassification"],
        "hidden_size": 768,
        "num_hidden_layers": 2,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "patch_size": 16,
        "frequency_stride": 10,
        "time_stride": 10,
        "num_mel_bins": 128,
        "max_length": 1024,
        "id2label": {"0": "Speech", "1": "Music"},
    }
    model.update(config or {})
    pre = {
        "sampling_rate": 16000,
        "num_mel_bins": 128,
        "max_length": 1024,
        "do_normalize": True,
        "padding_side": "right",
        "padding_value": 0.0,
        "mean": -4.2677393,
        "std": 4.5689974,
    }
    pre.update(preprocessor or {})
    (tmp_path / "config.json").write_text(json.dumps(model), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(pre), encoding="utf-8")
    return tmp_path


def _shapes(*, layers=2, seq_len=1214, hidden=768, patch=16, labels=2):
    shapes = {
        "audio_spectrogram_transformer.embeddings.position_embeddings": (1, seq_len, hidden),
        "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight": (
            hidden, 1, patch, patch),
        "classifier.dense.weight": (labels, hidden),
    }
    for index in range(layers):
        key = f"audio_spectrogram_transformer.encoder.layer.{index}.layernorm_before.weight"
        shapes[key] = (hidden,)
    return shapes


def test_resolve_reads_the_patch_grid_and_the_two_leading_tokens(tmp_path):
    cfg = config_module.resolve(_checkpoint(tmp_path), _shapes())
    # (128 - 16) // 10 + 1 frequency patches by (1024 - 16) // 10 + 1 time patches.
    assert (cfg["frequency_patches"], cfg["time_patches"]) == (12, 101)
    assert cfg["num_patches"] == 1212
    # Two, not one: AST keeps DeiT's distillation token beside the class token.
    assert cfg["seq_len"] == cfg["num_patches"] + 2 == 1214
    assert cfg["labels"] == ["Speech", "Music"]
    assert cfg["num_labels"] == 2
    # 25 ms and 10 ms at 16 kHz, and Kaldi's round-up to a power of two.
    assert (cfg["frame_length"], cfg["frame_shift"], cfg["n_fft"]) == (400, 160, 512)


def test_resolve_folds_the_doubled_deviation_into_one_scale(tmp_path):
    cfg = config_module.resolve(_checkpoint(tmp_path), _shapes())
    # The reference divides by twice the stated deviation. Dividing by the
    # deviation itself would double every feature.
    assert cfg["feature_scale"] == pytest.approx(1.0 / (2.0 * 4.5689974), rel=1e-12)
    assert cfg["feature_mean"] == pytest.approx(-4.2677393, rel=1e-12)


def test_resolve_refuses_a_position_table_that_contradicts_the_strides(tmp_path):
    # A stated time_stride of 16 would cut 64 time patches, not 101, so the
    # trained position table no longer fits. Silently building it would give a
    # network that runs and is wrong.
    with pytest.raises(ValueError, match="contradicts the stated geometry"):
        config_module.resolve(_checkpoint(tmp_path, config={"time_stride": 16}), _shapes())


def test_resolve_refuses_a_gap_in_the_encoder(tmp_path):
    shapes = _shapes(layers=2)
    shapes.pop("audio_spectrogram_transformer.encoder.layer.1.layernorm_before.weight")
    with pytest.raises(ValueError, match="are not 0"):
        config_module.resolve(_checkpoint(tmp_path), shapes)


def test_resolve_refuses_an_unnamed_class(tmp_path):
    checkpoint = _checkpoint(tmp_path, config={"id2label": {"0": "Speech", "1": "  "}})
    with pytest.raises(ValueError, match="no usable label"):
        config_module.resolve(checkpoint, _shapes())


def test_resolve_refuses_a_preprocessor_that_pads_on_the_left(tmp_path):
    # Left padding would put the silence before the audio, which moves every
    # patch in the time direction.
    with pytest.raises(ValueError, match="pads the spectrogram on the right"):
        config_module.resolve(
            _checkpoint(tmp_path, preprocessor={"padding_side": "left"}), _shapes()
        )


def test_resolve_refuses_a_preprocessor_that_contradicts_the_model(tmp_path):
    with pytest.raises(ValueError, match="contradicts config.json"):
        config_module.resolve(
            _checkpoint(tmp_path, preprocessor={"num_mel_bins": 64}), _shapes()
        )
