# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU reference checks for the XLSR layout and checkpoint path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from families.wav2vec2 import config as config_module, model

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from families.wav2vec2.convert_checkpoint import convert  # noqa: E402


def _reference(*, stable=True, norm="layer", bias=True):
    torch.manual_seed(23)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=12,
        conv_dim=(4, 8), conv_kernel=(3, 3), conv_stride=(2, 2), conv_bias=bias,
        feat_extract_norm=norm, do_stable_layer_norm=stable,
        num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        hidden_dropout=0, attention_dropout=0, activation_dropout=0, layerdrop=0,
        feat_proj_dropout=0, final_dropout=0, apply_spec_augment=False, vocab_size=4,
    )
    reference = transformers.Wav2Vec2ForCTC(cfg).eval()
    return cfg, reference


def _resolved(tmp_path, cfg, reference):
    cfg.architectures = ["Wav2Vec2ForCTC"]
    cfg.to_json_file(tmp_path / "config.json")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000}))
    (tmp_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "|": 1, "A": 2, "B": 3}))
    weights = {name: value.detach().numpy() for name, value in reference.state_dict().items()}
    return config_module.resolve(tmp_path, {name: value.shape for name, value in weights.items()}), weights


@pytest.fixture
def cpu_graph(monkeypatch):
    functional = torch.nn.functional

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))

    def norm(network, inp, gamma, beta, eps, axis, *, param_axis=None):
        out = functional.layer_norm(inp.movedim(axis, -1), (inp.shape[axis],), eps=eps)
        out = out.movedim(-1, axis)
        param_axis = inp.ndim - 1 if param_axis is None else param_axis
        shape = [1] * inp.ndim
        shape[param_axis] = len(gamma)
        return out * tensor(gamma).reshape(shape) + tensor(beta).reshape(shape)

    def conv(network, inp, weight, *, stride=1, padding=0, groups=1, bias=None):
        return functional.conv1d(inp.squeeze(2), tensor(weight),
                                 None if bias is None else tensor(bias),
                                 stride=stride, padding=padding, groups=groups).unsqueeze(2)

    def attention(network, query, key, value, *, num_heads, head_dim):
        def heads(inp):
            return inp.reshape(-1, num_heads, head_dim).transpose(0, 1)
        out = functional.scaled_dot_product_attention(heads(query), heads(key), heads(value))
        return out.transpose(0, 1).reshape(-1, num_heads * head_dim)

    monkeypatch.setattr(model.g, "add_norm", norm)
    monkeypatch.setattr(model.g, "add_conv1d", conv)
    monkeypatch.setattr(model.g, "add_gelu", lambda net, x: functional.gelu(x, approximate="none"))
    monkeypatch.setattr(model.g, "add_sum", lambda net, a, b: a + b)
    monkeypatch.setattr(model.g, "add_linear", lambda net, x, w, b=None: functional.linear(
        x, tensor(w), None if b is None else tensor(b)))
    monkeypatch.setattr(model.g, "add_attention", attention)


@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize("norm", ["group", "layer"])
@pytest.mark.parametrize("bias", [False, True])
def test_feature_encoder_matches_official_cpu(tmp_path, cpu_graph, stable, norm, bias):
    cfg, reference = _reference(stable=stable, norm=norm, bias=bias)
    resolved, weights = _resolved(tmp_path, cfg, reference)
    waveform = torch.linspace(-0.7, 1.3, 67).reshape(1, -1)
    with torch.no_grad():
        expected = reference.wav2vec2.feature_extractor(waveform)
        actual = model._feature_encoder(None, waveform[:, None, None, :], weights, resolved)
    torch.testing.assert_close(actual.squeeze(2), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("stable", [False, True])
def test_encoder_norm_placement_matches_official_cpu(tmp_path, cpu_graph, monkeypatch, stable):
    cfg, reference = _reference(stable=stable)
    resolved, weights = _resolved(tmp_path, cfg, reference)
    encoder = reference.wav2vec2.encoder
    rows = torch.linspace(-2, 0.7, 72).reshape(9, 8)
    monkeypatch.setattr(model, "_positional", lambda net, x, w, c:
                        x + encoder.pos_conv_embed(x.unsqueeze(0)).squeeze(0))
    with torch.no_grad():
        expected = encoder(rows.unsqueeze(0)).last_hidden_state.squeeze(0)
        actual = model._encoder(None, rows, weights, resolved)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_bin_checkpoint_loads_weights_on_cpu(tmp_path):
    torch.save({"lm_head.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3)},
               tmp_path / "pytorch_model.bin")
    convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")
    weights, shapes = model._load(tmp_path)
    np.testing.assert_array_equal(weights["lm_head.weight"], np.arange(12).reshape(4, 3))
    assert shapes["lm_head.weight"] == (4, 3)


def test_bin_checkpoint_rejects_non_tensor_state(tmp_path):
    torch.save({"lm_head.weight": "invalid"}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="tensor"):
        convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")


def test_official_japanese_geometry_is_resolved(tmp_path):
    path = Path(__file__).parent / "data/xlsr-japanese-config.json"
    (tmp_path / "config.json").write_bytes(path.read_bytes())
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"sampling_rate": 16000}))
    vocab = {"<pad>": 0, "|": 1, **{f"token{i}": i for i in range(2, 2341)}}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    cfg = config_module.resolve(tmp_path, {"lm_head.weight": (2341, 1024)})
    assert cfg["do_stable_layer_norm"] is True
    assert cfg["feat_extract_norm"] == "layer"
    assert cfg["conv_bias"] is True
    assert cfg["num_hidden_layers"] == 24
    assert cfg["num_attention_heads"] == 16


def test_feature_norm_epsilon_is_independent_of_encoder_epsilon(tmp_path, cpu_graph):
    cfg, reference = _reference()
    cfg.layer_norm_eps = 0.1
    resolved, weights = _resolved(tmp_path, cfg, reference)
    waveform = torch.linspace(-0.7, 1.3, 67).reshape(1, -1)
    with torch.no_grad():
        expected = reference.wav2vec2.feature_extractor(waveform)
        actual = model._feature_encoder(None, waveform[:, None, None, :], weights, resolved)
    torch.testing.assert_close(actual.squeeze(2), expected, atol=2e-6, rtol=2e-5)


def test_bin_loader_uses_restricted_cpu_loading(tmp_path, monkeypatch):
    (tmp_path / "pytorch_model.bin").touch()
    observed = {}

    def load(path, **kwargs):
        observed.update(kwargs)
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", load)
    convert(tmp_path / "pytorch_model.bin", tmp_path / "model.safetensors")
    assert observed == {"map_location": "cpu", "weights_only": True}


def test_conversion_preserves_existing_destination(tmp_path):
    destination = tmp_path / "model.safetensors"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        convert(tmp_path / "pytorch_model.bin", destination)
    assert destination.read_bytes() == b"existing"


def test_conversion_rejects_empty_state(tmp_path):
    source = tmp_path / "pytorch_model.bin"
    torch.save({}, source)
    with pytest.raises(ValueError, match="nonempty"):
        convert(source, tmp_path / "model.safetensors")


def test_e2e_preparation_preserves_cached_checkpoint(tmp_path, monkeypatch):
    from families.wav2vec2.tests import test_e2e

    for name in ("config.json", "preprocessor_config.json", "vocab.json"):
        (tmp_path / name).write_text("{}")
    torch.save({"weight": torch.arange(3, dtype=torch.float16)}, tmp_path / "pytorch_model.bin")
    observed = []

    def build(prepared, bundle, manifest):
        weights, shapes = model._load(prepared)
        np.testing.assert_array_equal(weights["weight"], np.arange(3))
        assert shapes["weight"] == (3,)
        assert all((prepared / name).read_text() == "{}" for name in
                   ("config.json", "preprocessor_config.json", "vocab.json"))
        observed.append(prepared)

    monkeypatch.setattr(test_e2e, "_build_prepared", build)
    test_e2e._build(tmp_path, tmp_path / "bundle", {})
    assert len(observed) == 1 and not observed[0].exists()
    assert not (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "pytorch_model.bin").is_file()


def test_missing_xlsr_bias_is_rejected_before_build(tmp_path):
    cfg, reference = _reference()
    resolved, weights = _resolved(tmp_path, cfg, reference)
    weights["wav2vec2.encoder.pos_conv_embed.conv.weight_g"] = np.ones(1)
    weights["wav2vec2.encoder.pos_conv_embed.conv.weight_v"] = np.ones(1)
    del weights["wav2vec2.feature_extractor.conv_layers.1.conv.bias"]
    with pytest.raises(KeyError, match="conv_layers.1.conv.bias"):
        model._require_weights(weights, resolved)
