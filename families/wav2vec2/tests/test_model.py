# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the rules the wav2vec2 graph depends on.

Each rule that is read from the checkpoint rather than stated in the config has
a control here that fails when the rule is broken. The controls matter because
a wav2vec2 graph degrades gracefully: three of the four ways of getting the
positional convolution wrong still produce a readable transcript.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.wav2vec2 import config as config_module, graph as g


def _config(**overrides):
    base = {
        "architectures": ["Wav2Vec2ForCTC"],
        "model_type": "wav2vec2",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "layer_norm_eps": 1e-05,
        "do_stable_layer_norm": False,
        "feat_extract_norm": "group",
        "conv_bias": False,
        "hidden_act": "gelu",
        "conv_dim": [512] * 7,
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "num_conv_pos_embeddings": 128,
        "num_conv_pos_embedding_groups": 16,
        "vocab_size": 4,
    }
    base.update(overrides)
    return base


_VOCAB = {"<pad>": 0, "|": 1, "A": 2, "B": 3}
_PRE = {"feature_size": 1, "padding_value": 0.0, "padding_side": "right",
        "sampling_rate": 16000, "do_normalize": True}


def _write(tmp_path, cfg=None, pre=None, vocab=None):
    (tmp_path / "config.json").write_text(json.dumps(cfg or _config()))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(pre or _PRE))
    (tmp_path / "vocab.json").write_text(json.dumps(vocab if vocab is not None else _VOCAB))
    return tmp_path


def test_resolve_reads_the_supported_variant(tmp_path):
    cfg = config_module.resolve(_write(tmp_path), {"lm_head.weight": (4, 768)})
    assert cfg["tokens"] == ["<pad>", "|", "A", "B"]
    assert cfg["blank_token_id"] == 0
    assert cfg["sample_rate"] == 16000
    assert cfg["do_normalize"] is True


@pytest.mark.parametrize("overrides, fragment", [
    ({"do_stable_layer_norm": True}, "post-norm"),
    ({"feat_extract_norm": "layer"}, "feat_extract_norm"),
    ({"conv_bias": True}, "bias-free"),
    ({"hidden_act": "gelu_new"}, "exact GELU"),
    ({"num_conv_pos_embeddings": 127}, "even num_conv_pos_embeddings"),
    ({"architectures": ["Wav2Vec2ForSequenceClassification"]}, "Wav2Vec2ForCTC"),
])
def test_resolve_refuses_the_variants_it_does_not_build(tmp_path, overrides, fragment):
    # Each of these loads without error and produces plausible output, so the
    # refusal is the only thing standing between them and a silent wrong answer.
    with pytest.raises(NotImplementedError, match=fragment):
        config_module.resolve(_write(tmp_path, cfg=_config(**overrides)), {})


def test_resolve_refuses_a_vocabulary_that_does_not_match(tmp_path):
    with pytest.raises(NotImplementedError, match="vocab.json"):
        config_module.resolve(_write(tmp_path, vocab={"<pad>": 0, "|": 1, "A": 2}), {})


def test_resolve_refuses_a_head_of_the_wrong_width(tmp_path):
    with pytest.raises(NotImplementedError, match="lm_head"):
        config_module.resolve(_write(tmp_path), {"lm_head.weight": (4, 1024)})


def test_resolve_refuses_a_duplicated_token_id(tmp_path):
    with pytest.raises(NotImplementedError, match="two tokens"):
        config_module.resolve(_write(tmp_path, vocab={"<pad>": 0, "|": 1, "A": 2, "B": 2}), {})


def test_weight_normed_kernel_reproduces_the_reference_formula():
    rng = np.random.default_rng(0)
    weight_v = rng.normal(size=(8, 4, 6)).astype(np.float32)
    weight_g = rng.normal(size=(1, 1, 6)).astype(np.float32)
    kernel = g.weight_normed_kernel(weight_g, weight_v)
    assert kernel.shape == weight_v.shape

    # Each kernel tap ends up with the length weight_g states for it. That is
    # the whole content of the reparameterisation, so it is checked directly.
    lengths = np.sqrt((kernel.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(lengths - np.abs(weight_g).reshape(-1)).max() < 1e-5

    # Controls. Using weight_v directly, or taking the norm over the output and
    # tap axes instead, both give a kernel with the wrong tap lengths.
    raw = np.sqrt((weight_v.astype(np.float64) ** 2).sum(axis=(0, 1)))
    assert np.abs(raw - np.abs(weight_g).reshape(-1)).max() > 1e-2
    wrong_axis = weight_v * (weight_g.reshape(1, 1, -1) / np.sqrt(
        (weight_v.astype(np.float64) ** 2).sum(axis=(1, 2), keepdims=True)))
    assert np.abs(np.asarray(wrong_axis) - kernel).max() > 1e-2


def test_sample_range_is_a_real_range():
    assert config_module.MIN_SAMPLES < config_module.OPT_SAMPLES < config_module.MAX_SAMPLES
    # The profile has to cover a useful clip: 30 s at 16 kHz.
    assert config_module.MAX_SAMPLES >= 30 * 16000
